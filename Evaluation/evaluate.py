"""
Evaluation and visualization functions
"""

import numpy as np
import matplotlib.pyplot as plt
import torch


def visualize_predictions(model, dataset, device, num_samples=5):
    """
    Visualize model predictions on random samples
    
    Args:
        model: Trained model
        dataset: Dataset to sample from
        device: Device (cuda/cpu)
        num_samples: Number of samples to visualize
    """
    model.eval()
    indices = np.random.choice(len(dataset), num_samples, replace=False)
    
    for idx in indices:
        image, mask = dataset[idx]
        
        # Get prediction
        with torch.no_grad():
            output = model(image.unsqueeze(0).to(device))
            pred = torch.argmax(output, dim=1).squeeze().cpu().numpy()
        
        # Denormalize image for display
        img_display = image.cpu().numpy().transpose(1, 2, 0)
        img_display = img_display * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]
        img_display = np.clip(img_display, 0, 1)
        
        mask_np = mask.cpu().numpy()
        
        # Create visualization
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        
        # Original image
        axes[0].imshow(img_display)
        axes[0].set_title('Image')
        axes[0].axis('off')
        
        # Ground truth
        axes[1].imshow(mask_np, cmap='tab10', vmin=0, vmax=2)
        axes[1].set_title('Ground Truth')
        axes[1].axis('off')
        
        # Prediction
        axes[2].imshow(pred, cmap='tab10', vmin=0, vmax=2)
        axes[2].set_title('Prediction')
        axes[2].axis('off')
        
        # Overlay prediction on image
        overlay = (img_display * 255).astype(np.uint8).copy()
        overlay[pred == 1] = [255, 0, 0]  # Left lane = Red
        overlay[pred == 2] = [0, 0, 255]  # Right lane = Blue
        axes[3].imshow(overlay)
        axes[3].set_title('Overlay (Red=Left, Blue=Right)')
        axes[3].axis('off')
        
        plt.tight_layout()
        plt.show()


def evaluate_model(model, loader, device, num_classes=3):
    """
    Comprehensive model evaluation
    
    Args:
        model: Trained model
        loader: DataLoader for test data
        device: Device (cuda/cpu)
        num_classes: Number of classes
    
    Returns:
        results: Dictionary with evaluation metrics
    """
    from metrics import dice_score, pixel_accuracy, iou_score
    
    model.eval()
    
    all_dice = []
    all_iou = []
    all_acc = []
    dice_per_class = [[], [], []]
    iou_per_class = [[], [], []]
    
    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(images)
            
            # Calculate metrics
            dice, dice_class = dice_score(outputs, masks, num_classes)
            iou, iou_class = iou_score(outputs, masks, num_classes)
            acc = pixel_accuracy(outputs, masks)
            
            all_dice.append(dice)
            all_iou.append(iou)
            all_acc.append(acc)
            
            for i in range(num_classes):
                dice_per_class[i].append(dice_class[i])
                iou_per_class[i].append(iou_class[i])
    
    results = {
        'dice': np.mean(all_dice),
        'iou': np.mean(all_iou),
        'accuracy': np.mean(all_acc),
        'dice_per_class': [np.mean(d) for d in dice_per_class],
        'iou_per_class': [np.mean(i) for i in iou_per_class]
    }
    
    return results


def print_evaluation_results(results, class_names):
    """
    Print evaluation results in a formatted way
    
    Args:
        results: Dictionary with evaluation metrics
        class_names: List of class names
    """
    print("\n" + "="*70)
    print("EVALUATION RESULTS")
    print("="*70)
    print(f"Overall Metrics:")
    print(f"  Dice Score:      {results['dice']:.4f}")
    print(f"  IoU Score:       {results['iou']:.4f}")
    print(f"  Pixel Accuracy:  {results['accuracy']:.4f}")
    print(f"\nPer-Class Metrics:")
    
    for i, class_name in enumerate(class_names):
        print(f"  {class_name}:")
        print(f"    Dice: {results['dice_per_class'][i]:.4f}")
        print(f"    IoU:  {results['iou_per_class'][i]:.4f}")
    
    print("="*70 + "\n")
