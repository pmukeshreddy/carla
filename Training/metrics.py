"""
Evaluation metrics for segmentation
"""

import numpy as np
import torch


def dice_score(pred, target, num_classes=3, smooth=1e-6):
    """
    Calculate Dice score for multi-class segmentation
    
    Args:
        pred: (B, C, H, W) logits
        target: (B, H, W) class indices
        num_classes: Number of classes
        smooth: Smoothing factor to avoid division by zero
    
    Returns:
        dice: Mean dice score across all classes
        dice_per_class: List of dice scores for each class
    """
    pred = torch.softmax(pred, dim=1)  # (B, C, H, W)
    pred = torch.argmax(pred, dim=1)   # (B, H, W)
    
    dice_per_class = []
    
    for c in range(num_classes):
        pred_c = (pred == c).float()
        target_c = (target == c).float()
        
        intersection = (pred_c * target_c).sum()
        union = pred_c.sum() + target_c.sum()
        
        dice_c = (2.0 * intersection + smooth) / (union + smooth)
        dice_per_class.append(dice_c.item())
    
    return np.mean(dice_per_class), dice_per_class


def pixel_accuracy(pred, target):
    """
    Calculate pixel-wise accuracy
    
    Args:
        pred: (B, C, H, W) logits
        target: (B, H, W) class indices
    
    Returns:
        accuracy: Pixel accuracy
    """
    pred = torch.argmax(pred, dim=1)
    correct = (pred == target).sum().item()
    total = target.numel()
    return correct / total


def iou_score(pred, target, num_classes=3, smooth=1e-6):
    """
    Calculate IoU (Intersection over Union) score
    
    Args:
        pred: (B, C, H, W) logits
        target: (B, H, W) class indices
        num_classes: Number of classes
        smooth: Smoothing factor
    
    Returns:
        mean_iou: Mean IoU across all classes
        iou_per_class: List of IoU for each class
    """
    pred = torch.argmax(pred, dim=1)
    
    iou_per_class = []
    
    for c in range(num_classes):
        pred_c = (pred == c).float()
        target_c = (target == c).float()
        
        intersection = (pred_c * target_c).sum()
        union = pred_c.sum() + target_c.sum() - intersection
        
        iou_c = (intersection + smooth) / (union + smooth)
        iou_per_class.append(iou_c.item())
    
    return np.mean(iou_per_class), iou_per_class
