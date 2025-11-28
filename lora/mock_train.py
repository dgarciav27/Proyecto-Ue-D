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
import time

# Arguments configuration
parser = argparse.ArgumentParser()
parser.add_argument('--data_root', default='/SCRATCH/local/biodcase_development_set', type=str)
parser.add_argument('--train_annot', type=str, required=True)
parser.add_argument('--val_annot', type=str, required=True)
parser.add_argument('--batch_size', default=32, type=int)
parser.add_argument('--n_classes', default=7, type=int, choices={3, 7, 10})
parser.add_argument('--sample_rate', default=250, type=int)
parser.add_argument('--duration', default=15, type=int)
parser.add_argument('--n_fft', default=512, type=int)
parser.add_argument('--win_size', default=256, type=int)
parser.add_argument('--overlap', default=98, type=int)
parser.add_argument('--fine_tune', default=False, type=bool)
parser.add_argument('--n_epochs', default=10, type=int)
parser.add_argument('--save_dir', default='/Brain/private/d25garci/vit_models', type=str)
parser.add_argument('--model_path', default='/Brain/public/models/google/vit-base-patch16-224', type=str)
parser.add_argument('--patience', default=5, type=int)
args = parser.parse_args()

# Define labels
if args.n_classes == 3:
    args.labels = ['abz', 'd', 'bp']
elif args.n_classes == 7:
    args.labels = ['bma', 'bmb', 'bmz', 'bmd', 'bpd', 'bp20', 'bp20plus']
elif args.n_classes == 10:
    args.labels = ['bma', 'bmb', 'bmz', 'bmd', 'bpd', 'bp20', 'bp20plus', 'abz', 'd', 'bp']

# Create save directory
os.makedirs(args.save_dir, exist_ok=True)
print(f"💾 Model save in: {args.save_dir}")

# Load full data
train_dataloader, val_dataloader = load_vit_data(args)

# Initialize ViT backbone
model = init_vit_model(args)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()  # mock train, just forward pass

# MOCK TRAIN LOOP
print("=== MOCK TRAINING ===")
start_time = time.time()
for batch in tqdm(train_dataloader, desc="Mock Epoch"):
    inputs, labels = batch
    pixel_values = inputs['pixel_values'].to(device)
    
    with torch.no_grad():  # no gradients
        _ = model(pixel_values=pixel_values)  # forward pass only

end_time = time.time()
print(f"Duration of mock epoch: {end_time - start_time:.2f} s")

# MOCK VALIDATION
print("=== MOCK VALIDATION ===")
start_val = time.time()
for batch in tqdm(val_dataloader, desc="Mock Validation"):
    inputs, labels = batch
    pixel_values = inputs['pixel_values'].to(device)
    
    with torch.no_grad():
        _ = model(pixel_values=pixel_values)

end_val = time.time()
print(f"Duration of mock validation: {end_val - start_val:.2f} s")
