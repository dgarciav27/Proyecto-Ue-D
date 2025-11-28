import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from transformers import ViTModel
from vit_loaders import load_vit_data, init_vit_model
import argparse
from datetime import datetime
from metrics_lora import compute_best_pr_and_f1
# Arguments configuration
parser = argparse.ArgumentParser()
parser.add_argument('--data_root', default='/SCRATCH/local/biodcase_development_set', type=str)
parser.add_argument('--train_annot', type=str, required=True)
parser.add_argument('--val_annot', type=str, required=True)
parser.add_argument('--batch_size', default=32, type=int) #16 bench
parser.add_argument('--n_classes', default=7, type=int, choices={3, 7, 10})
parser.add_argument('--sample_rate', default=250, type=int) 
parser.add_argument('--duration', default=15, type=int) #5
parser.add_argument('--n_fft', default=512, type=int) #3570 (benchmark)
parser.add_argument('--win_size', default=256, type=int) #250
parser.add_argument('--overlap', default=98, type=int)
parser.add_argument('--fine_tune', default=False, type=bool) #T
parser.add_argument('--n_epochs', default=10, type=int)  
parser.add_argument('--save_dir', default='/Brain/private/d25garci/vit_models', type=str)  
parser.add_argument('--model_path', default='/Brain/public/models/google/vit-base-patch16-224', type=str)  #
parser.add_argument('--patience', default=5, type=int) # early stopping

# Define labels taking into account n_classes
args = parser.parse_args()
if args.n_classes == 3:
    args.labels = ['abz', 'd', 'bp']
elif args.n_classes == 7:
    args.labels = ['bma', 'bmb', 'bmz', 'bmd', 'bpd', 'bp20', 'bp20plus']
elif args.n_classes == 10:
    args.labels = ['bma', 'bmb', 'bmz', 'bmd', 'bpd', 'bp20', 'bp20plus', 'abz', 'd', 'bp']

# Create directory for saving the model
os.makedirs(args.save_dir, exist_ok=True)
print(f"💾 Model save in: {args.save_dir}")

# Load data
train_dataloader, val_dataloader = load_vit_data(args)

# Initialize Vit model
model = init_vit_model(args)

# 0. PARAMETERS
hidden_size = model.config.hidden_size 
num_classes = args.n_classes

# ===== FUNCTION FOR VALIDATION =====
def evaluate_model_with_metrics(model, dataloader, device, labels):
    model.eval()
    criterion = nn.BCEWithLogitsLoss()
    total_loss = 0

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in dataloader:
            inputs, labels_batch = batch
            pixel_values = inputs['pixel_values'].to(device)
            labels_batch = labels_batch.to(device).float()

            outputs = model(pixel_values=pixel_values)
            logits = model.classifier(outputs.last_hidden_state[:, 0, :])
            loss = criterion(logits, labels_batch)
            total_loss += loss.item()

            all_preds.append(torch.sigmoid(logits).cpu())
            all_targets.append(labels_batch.cpu())

    y_hat = torch.cat(all_preds)
    y_true = torch.cat(all_targets)

    # Calcula mejores métricas
    metrics = compute_best_pr_and_f1(y_true, y_hat, labels)
    
    return total_loss / len(dataloader), metrics

# -------Transfer learning-----------------
print("=== TRANSFER LEARNING ===")
# 1. FREEZE THE PRE-TRAINED MODEL (BACKBONE) ❄️
for param in model.parameters():
    param.requires_grad = False

# 2. ADD A NEW, TRAINABLE HEAD 🔥
classification_head = nn.Linear(hidden_size, num_classes)

# 3. TRAINING LOOP (simplified)
optimizer = optim.AdamW(classification_head.parameters(), lr=1e-3)
criterion = nn.BCEWithLogitsLoss()

classification_head.train()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
classification_head.to(device)

# Actual training loop
for batch in tqdm(train_dataloader, desc="Transfer Learning", leave=False):
    inputs, labels = batch
    pixel_values = inputs['pixel_values'].to(device)
    labels = labels.to(device).float() 

    optimizer.zero_grad()

    # 1. Get backbone outputs (features)
    with torch.no_grad():
        outputs = model(pixel_values=pixel_values)
        features = outputs.last_hidden_state[:, 0, :]  # [CLS] token

    # 2. Pass features to the new head
    logits = classification_head(features)

    # 3. Calculate loss
    loss = criterion(logits, labels)

    # 4. Backpropagate
    loss.backward()

    # 5. Update weights
    optimizer.step()

print("Transfer Learning complete!")

# -------Fine-Tuning-----------------
print("=== FINE-TUNING ===")
# 1. DEFINE A NEW, TRAINABLE HEAD 🔥
classification_head_ft = nn.Linear(hidden_size, num_classes)

# 2. COMBINE THE BACKBONE AND HEAD INTO ONE MODEL
class FullModel(nn.Module):
    def __init__(self, backbone, classification_head):
        super(FullModel, self).__init__()
        self.backbone = backbone
        self.head = classification_head

    def forward(self, pixel_values):
        outputs = self.backbone(pixel_values=pixel_values)
        features = outputs.last_hidden_state[:, 0, :]
        logits = self.head(features)
        return logits

# 3. DEFINE THE NEW MODEL
fine_tuned_model = FullModel(model, classification_head_ft)

# >>> Defrost parameters for fine-tuning
for param in fine_tuned_model.backbone.parameters():
    param.requires_grad = True

# 4. TRAINING LOOP 
optimizer_ft = optim.AdamW(fine_tuned_model.parameters(), lr=2e-5)
criterion = nn.BCEWithLogitsLoss()

fine_tuned_model.train()
fine_tuned_model.to(device)

for batch in tqdm(train_dataloader, desc="Fine-Tuning", leave=False):
    inputs, labels = batch
    pixel_values = inputs['pixel_values'].to(device)
    labels = labels.to(device).float()

    optimizer_ft.zero_grad()

    # 1. Forward pass
    logits = fine_tuned_model(pixel_values)

    # 3. Calculate loss
    loss = criterion(logits, labels)

    # 4. Backpropagate
    loss.backward()

    # 5. Update weights
    optimizer_ft.step()

print("Fine-Tuning complete!")

# -------PEFT -- LoRA----------------
print("=== LoRA TRAINING ===")
from peft import LoraConfig, get_peft_model

## 1. CONFIGURE THE LORA MODULE
config = LoraConfig(
    r=4,
    target_modules=["query", "key", "value", "dense"],
    lora_dropout=0,
    lora_alpha=4,
    bias="none",
)

## 2. SET UP YOUR MODEL WITH LORA ADAPTERS
lora_model = get_peft_model(model, config)
lora_model.classifier = nn.Linear(hidden_size, num_classes)

# 4. TRAINING LOOP
optimizer_lora = optim.AdamW(lora_model.parameters(), lr=1e-3)
criterion = nn.BCEWithLogitsLoss()

lora_model.to(device)

# SAVE BEST LORA MODEL
best_lora_loss = float('inf')

for epoch in range(args.n_epochs):
    # Training
    lora_model.train()
    total_loss = 0
    
    for batch in tqdm(train_dataloader, desc=f"LoRA Epoch {epoch+1}"):
        inputs, labels = batch
        pixel_values = inputs['pixel_values'].to(device)
        labels = labels.to(device).float()

        optimizer_lora.zero_grad()
        outputs = lora_model(pixel_values=pixel_values)
        logits = lora_model.classifier(outputs.last_hidden_state[:, 0, :])
        loss = criterion(logits, labels)
        loss.backward()
        optimizer_lora.step()
        
        total_loss += loss.item()
    
    # Validation
    val_loss, val_metrics = evaluate_model_with_metrics(lora_model, val_dataloader, device, args.labels)
    print(f"Epoch {epoch+1} - Train Loss: {total_loss/len(train_dataloader):.4f}, Val Loss: {val_loss:.4f}")
    print(f"-Metrics: Threshold {val_metrics['threshold']:.2f}, Precision {val_metrics['mean_best_prec']:.4f}, Recall {val_metrics['mean_best_recall']:.4f}, F1 {val_metrics['mean_best_f1']:.4f}")
    
    # SAVE BEST MODEL
    if val_loss < best_lora_loss:
        best_lora_loss = val_loss
        lora_model.save_pretrained(os.path.join(args.save_dir, 'best_lora_model'))
        torch.save({
            'epoch': epoch + 1,
            'val_loss': best_lora_loss,
            'classifier_state_dict': lora_model.classifier.state_dict(),
        }, os.path.join(args.save_dir, 'best_lora_classifier.pth'))
        print(f"💾 New best LoRA model saved (val_loss: {val_loss:.4f})")

print("TRAINING COMPLETED")
print(f"Best model_LoRA save with val_loss: {best_lora_loss:.4f}")