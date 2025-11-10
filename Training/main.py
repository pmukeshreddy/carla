"""
Main training script for lane segmentation
"""

import time
import json
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from config import CONFIG
from dataset import LaneSegmentationDataset
from model import UNet
from train import train_epoch, validate
from utils import match_images_to_labels, plot_training_curves
from evaluate import visualize_predictions


def setup_data_loaders(config):
    """
    Setup training and validation data loaders
    
    Args:
        config: Configuration dictionary
    
    Returns:
        train_loader: Training DataLoader
        val_loader: Validation DataLoader
    """
    print("\n" + "="*70)
    print("LOADING DATASET")
    print("="*70)
    
    # Setup paths
    train_img_path = Path(config["train_images"])
    train_label_path = Path(config["train_labels"])
    val_img_path = Path(config["val_images"])
    val_label_path = Path(config["val_labels"])
    
    # Match images to labels
    print("Matching train images to labels...")
    train_pairs = match_images_to_labels(train_img_path, train_label_path)
    train_images = [img for img, _ in train_pairs]
    train_labels = [label for _, label in train_pairs]
    
    print("Matching val images to labels...")
    val_pairs = match_images_to_labels(val_img_path, val_label_path)
    val_images = [img for img, _ in val_pairs]
    val_labels = [label for _, label in val_pairs]
    
    print(f"\nTrain: {len(train_images)} matched pairs")
    print(f"Val:   {len(val_images)} matched pairs")
    print("="*70 + "\n")
    
    # Create datasets
    train_dataset = LaneSegmentationDataset(
        train_images, 
        train_labels, 
        img_size=config["img_size"],
        augment=config["use_augmentation"]
    )
    
    val_dataset = LaneSegmentationDataset(
        val_images, 
        val_labels, 
        img_size=config["img_size"],
        augment=False
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        pin_memory=config["pin_memory"]
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        pin_memory=config["pin_memory"]
    )
    
    return train_loader, val_loader, val_dataset


def main():
    """
    Main training function
    """
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load data
    train_loader, val_loader, val_dataset = setup_data_loaders(CONFIG)
    
    # Initialize model
    print("="*70)
    print("INITIALIZING MODEL")
    print("="*70)
    
    model = UNet(
        in_channels=3, 
        num_classes=CONFIG['num_classes']
    ).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print("="*70 + "\n")
    
    # Setup training
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=CONFIG['learning_rate'],
        weight_decay=CONFIG['weight_decay']
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, verbose=True
    )
    
    # Training history
    history = {
        'train_loss': [],
        'train_dice': [],
        'val_loss': [],
        'val_dice': [],
        'val_dice_per_class': []
    }
    
    best_val_dice = 0
    patience_counter = 0
    patience = CONFIG['patience']
    
    print("="*70)
    print("STARTING TRAINING")
    print("="*70 + "\n")
    
    # Training loop
    for epoch in range(CONFIG['num_epochs']):
        print(f"\nEpoch {epoch+1}/{CONFIG['num_epochs']}")
        print("-" * 70)
        
        start_time = time.time()
        
        # Train
        train_loss, train_dice = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        
        # Validate
        val_loss, val_dice, val_dice_per_class = validate(
            model, val_loader, criterion, device
        )
        
        # Scheduler step
        scheduler.step(val_dice)
        
        epoch_time = time.time() - start_time
        
        # Record history
        history['train_loss'].append(train_loss)
        history['train_dice'].append(train_dice)
        history['val_loss'].append(val_loss)
        history['val_dice'].append(val_dice)
        history['val_dice_per_class'].append(val_dice_per_class)
        
        # Print summary
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  Train: Loss={train_loss:.4f}, Dice={train_dice:.4f}")
        print(f"  Val:   Loss={val_loss:.4f}, Dice={val_dice:.4f}")
        print(f"  Val Dice per class:")
        print(f"    Background: {val_dice_per_class[0]:.4f}")
        print(f"    Left Lane:  {val_dice_per_class[1]:.4f}")
        print(f"    Right Lane: {val_dice_per_class[2]:.4f}")
        print(f"  Time: {epoch_time:.1f}s | LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save best model
        if val_dice > best_val_dice:
            best_val_dice = val_dice
            patience_counter = 0
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_dice': val_dice,
                'val_dice_per_class': val_dice_per_class,
            }, 'best_lane_segmentation_model.pth')
            
            print(f"  ✓ Best model saved (Dice: {best_val_dice:.4f})")
        else:
            patience_counter += 1
            print(f"  No improvement ({patience_counter}/{patience})")
        
        # Early stopping
        if patience_counter >= patience:
            print(f"\n⚠️  Early stopping triggered")
            break
    
    print("\n" + "="*70)
    print(f"TRAINING COMPLETE - Best Dice: {best_val_dice:.4f}")
    print("="*70)
    
    # Plot training curves
    plot_training_curves(history)
    
    # Load best model and visualize predictions
    print("\nLoading best model for visualization...")
    checkpoint = torch.load('best_lane_segmentation_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print("Generating prediction visualizations...")
    visualize_predictions(model, val_dataset, device, num_samples=5)
    
    # Save final model
    torch.save(model.state_dict(), 'lane_segmentation_final.pth')
    
    # Save configuration
    deployment_config = {
        'model': 'UNet',
        'img_size': CONFIG['img_size'],
        'num_classes': CONFIG['num_classes'],
        'best_dice': float(best_val_dice),
        'class_names': CONFIG['class_names']
    }
    
    with open('deployment_config.json', 'w') as f:
        json.dump(deployment_config, f, indent=2)
    
    print("\n✓ Model saved: lane_segmentation_final.pth")
    print("✓ Config saved: deployment_config.json")
    print("\n🎉 ALL DONE! 🎉")


if __name__ == "__main__":
    main()
