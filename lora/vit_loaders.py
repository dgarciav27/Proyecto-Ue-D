import os
import torch
from torch.utils.data import DataLoader
from transformers import ViTModel
from vit_dataset import ViTAudioDataset

def load_vit_data(args, load_info=False, val_only=False):
    print('[INFO]: Loading ViT data')

    train_set = ViTAudioDataset(args, mode='train', load_info=load_info)
    val_set = ViTAudioDataset(args, mode='validation', load_info=load_info)

    # Collate function for Vit
    def vit_collate_fn(batch):
        if load_info:
            infos, inputs, labels = zip(*batch)
            # Stack for inputs
            pixel_values = torch.cat([item['pixel_values'] for item in inputs], dim=0)
            return list(infos), {'pixel_values': pixel_values}, torch.stack(labels)
        else:
            inputs, labels = zip(*batch)
            pixel_values = torch.cat([item['pixel_values'] for item in inputs], dim=0)
            return {'pixel_values': pixel_values}, torch.stack(labels)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, collate_fn=vit_collate_fn, num_workers=12, pin_memory=True, prefetch_factor=2)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, collate_fn=vit_collate_fn, num_workers=12, pin_memory=True, prefetch_factor=2)

    if val_only:
        return val_loader
    else:
        return train_loader, val_loader

def init_vit_model(args):
    print("[INFO]: Loading pre-trained ViT model")
    model = ViTModel.from_pretrained(args.model_path)  
    
    # Freeze or not parameters by fine_tune
    if args.fine_tune:
        print("[INFO]: Fine-tuning all layers...")
        for param in model.parameters():
            param.requires_grad = True
    else:
        print("[INFO]: Freezing backbone layers...")
        for param in model.parameters():
            param.requires_grad = False
    
    return model

def load_vit_from_ckpt(args, model, device, optimizer=None):
    """
    Load a ViT model from checkpoint 
    """
    path = os.path.join(args.log_path, args.modelckpt)
    print(f'[INFO]: Checkpointing from {path}')

    checkpoint = torch.load(path, weights_only=False, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer is None:
        return model
    else:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        best_f1 = checkpoint['best_f1']
        best_val_loss = checkpoint['best_val_loss']
        patience_counter_f1 = checkpoint['patience_counter_f1']
        patience_counter_loss = checkpoint['patience_counter_loss']

        return model, optimizer, epoch, best_f1, best_val_loss, patience_counter_f1, patience_counter_loss