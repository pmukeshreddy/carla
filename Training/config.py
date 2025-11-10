"""
Configuration file for lane segmentation training
"""

CONFIG = {
    # Data paths
    'train_images': '/kaggle/input/lane-detection-for-carla-driving-simulator/train',
    'train_labels': '/kaggle/input/lane-detection-for-carla-driving-simulator/train_label',
    'val_images': '/kaggle/input/lane-detection-for-carla-driving-simulator/val',
    'val_labels': '/kaggle/input/lane-detection-for-carla-driving-simulator/val_label',
    
    # Model parameters
    'img_size': (256, 256),  # (H, W) - adjust based on GPU memory
    'num_classes': 3,  # Background, Left Lane, Right Lane
    
    # Training parameters
    'batch_size': 16,
    'num_epochs': 50,
    'learning_rate': 1e-3,
    'weight_decay': 1e-5,
    'patience': 10,  # Early stopping patience
    
    # Augmentation
    'use_augmentation': True,
    
    # Device
    'num_workers': 2,
    'pin_memory': True,
    
    # Class names
    'class_names': ['Background', 'Left Lane', 'Right Lane']
}
