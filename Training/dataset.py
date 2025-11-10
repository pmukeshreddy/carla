"""
Dataset class for lane segmentation
"""

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class LaneSegmentationDataset(Dataset):
    """
    Dataset for lane segmentation with CARLA driving simulator images
    """
    
    def __init__(self, image_paths, label_paths, img_size=(256, 256), augment=False):
        """
        Args:
            image_paths: List of paths to images
            label_paths: List of paths to labels
            img_size: Target image size (H, W)
            augment: Whether to apply data augmentation
        """
        self.image_paths = image_paths
        self.label_paths = label_paths
        self.img_size = img_size
        self.augment = augment
        
        # Image transformations (normalization with ImageNet stats)
        self.img_transforms = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], 
                               [0.229, 0.224, 0.225])
        ])
        
        # Mask transformations
        self.mask_transforms = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor()
        ])
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image and mask
        image = Image.open(self.image_paths[idx]).convert("RGB")
        mask = Image.open(self.label_paths[idx])
        
        # Apply augmentation
        if self.augment:
            # Random horizontal flip
            if np.random.random() > 0.5:
                image = transforms.functional.hflip(image)
                mask = transforms.functional.hflip(mask)
            
            # Random color jitter
            if np.random.random() > 0.5:
                image = transforms.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.2,
                    hue=0.1
                )(image)
        
        # Apply transformations
        image = self.img_transforms(image)
        mask = self.mask_transforms(mask)
        
        # Convert mask to long tensor with class indices
        mask = (mask.squeeze(0) * 255).long()
        
        return image, mask
