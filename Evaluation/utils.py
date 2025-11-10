"""
Utility functions for lane segmentation
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image


def match_images_to_labels(img_dir, label_dir):
    """
    Correctly match images to their corresponding labels
    
    Image:  Town04_Clear_Noon_09_09_2020_14_57_22_frame_10.png
    Label:  Town04_Clear_Noon_09_09_2020_14_57_22_frame_10_label.png
    
    Args:
        img_dir: Path to image directory
        label_dir: Path to label directory
    
    Returns:
        matched_pairs: List of (image_path, label_path) tuples
    """
    all_images = sorted(list(img_dir.glob("*.png")))
    
    matched_pairs = []
    unmatched = []
    
    for img_path in all_images:
        # Expected label name: image_name + "_label.png"
        img_stem = img_path.stem  # Filename without extension
        label_name = f"{img_stem}_label.png"
        label_path = label_dir / label_name
        
        if label_path.exists():
            matched_pairs.append((img_path, label_path))
        else:
            unmatched.append(img_path.name)
    
    if unmatched:
        print(f"⚠️  Warning: {len(unmatched)} images without matching labels")
        if len(unmatched) <= 5:
            for name in unmatched:
                print(f"    - {name}")
    
    return matched_pairs


def visualize_sample(img_path, label_path, title=""):
    """
    Visualize image and segmentation mask
    
    Args:
        img_path: Path to image
        label_path: Path to label
        title: Plot title
    """
    img = Image.open(img_path).convert('RGB')
    label = Image.open(label_path)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    axes[0].imshow(img)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Segmentation mask
    axes[1].imshow(label, cmap='tab10')
    axes[1].set_title('Lane Mask (0=BG, 1=Left, 2=Right)')
    axes[1].axis('off')
    
    # Overlay
    label_np = np.array(label)
    overlay = np.array(img).copy()
    overlay[label_np == 1] = [255, 0, 0]  # Left lane = Red
    overlay[label_np == 2] = [0, 0, 255]  # Right lane = Blue
    axes[2].imshow(overlay)
    axes[2].set_title('Overlay (Red=Left, Blue=Right)')
    axes[2].axis('off')
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


def plot_training_curves(history, save_path='training_curves.png'):
    """
    Plot training and validation curves
    
    Args:
        history: Dictionary with training history
        save_path: Path to save the plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss
    axes[0].plot(epochs, history['train_loss'], 'b-', label='Train', linewidth=2)
    axes[0].plot(epochs, history['val_loss'], 'r-', label='Val', linewidth=2)
    axes[0].set_title('Loss', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Cross Entropy Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Dice Score
    axes[1].plot(epochs, history['train_dice'], 'b-', label='Train', linewidth=2)
    axes[1].plot(epochs, history['val_dice'], 'r-', label='Val', linewidth=2)
    axes[1].set_title('Dice Score', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Dice Score')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"✓ Training curves saved to {save_path}")
