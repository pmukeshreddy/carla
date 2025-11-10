"""
Training and validation functions
"""

import numpy as np
from tqdm import tqdm
import torch

from metrics import dice_score


def train_epoch(model, loader, criterion, optimizer, device):
    """
    Train for one epoch
    
    Args:
        model: Neural network model
        loader: DataLoader for training data
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on (cuda/cpu)
    
    Returns:
        avg_loss: Average training loss
        avg_dice: Average training dice score
    """
    model.train()
    
    total_loss = 0
    total_dice = 0
    
    pbar = tqdm(loader, desc='Training')
    for images, masks in pbar:
        images = images.to(device)
        masks = masks.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, masks)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Calculate metrics
        dice, _ = dice_score(outputs, masks)
        
        total_loss += loss.item()
        total_dice += dice
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}', 
            'dice': f'{dice:.4f}'
        })
    
    avg_loss = total_loss / len(loader)
    avg_dice = total_dice / len(loader)
    
    return avg_loss, avg_dice


def validate(model, loader, criterion, device):
    """
    Validate model
    
    Args:
        model: Neural network model
        loader: DataLoader for validation data
        criterion: Loss function
        device: Device to validate on (cuda/cpu)
    
    Returns:
        avg_loss: Average validation loss
        avg_dice: Average validation dice score
        avg_dice_per_class: Average dice score for each class
    """
    model.eval()
    
    total_loss = 0
    total_dice = 0
    dice_per_class_all = [[], [], []]
    
    with torch.no_grad():
        pbar = tqdm(loader, desc='Validation')
        for images, masks in pbar:
            images = images.to(device)
            masks = masks.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            # Calculate metrics
            dice, dice_per_class = dice_score(outputs, masks)
            
            total_loss += loss.item()
            total_dice += dice
            
            # Accumulate per-class dice scores
            for i, d in enumerate(dice_per_class):
                dice_per_class_all[i].append(d)
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}', 
                'dice': f'{dice:.4f}'
            })
    
    avg_loss = total_loss / len(loader)
    avg_dice = total_dice / len(loader)
    avg_dice_per_class = [np.mean(d) for d in dice_per_class_all]
    
    return avg_loss, avg_dice, avg_dice_per_class
