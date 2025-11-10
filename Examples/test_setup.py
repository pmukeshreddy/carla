"""
Test script to verify installation and setup
Run this to check if everything is working correctly
"""

import torch
import numpy as np
from PIL import Image

print("="*70)
print("TESTING LANE SEGMENTATION SETUP")
print("="*70)

# Test 1: Import modules
print("\n1. Testing imports...")
try:
    from config import CONFIG
    from model import UNet
    from dataset import LaneSegmentationDataset
    from metrics import dice_score
    from predict import LaneSegmentationPredictor
    print("   ✓ All modules imported successfully")
except Exception as e:
    print(f"   ✗ Import failed: {e}")
    exit(1)

# Test 2: Check PyTorch and CUDA
print("\n2. Checking PyTorch and CUDA...")
print(f"   PyTorch version: {torch.__version__}")
print(f"   CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"   CUDA version: {torch.version.cuda}")
    print(f"   GPU device: {torch.cuda.get_device_name(0)}")

# Test 3: Create model
print("\n3. Testing model creation...")
try:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(in_channels=3, num_classes=3).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   ✓ Model created successfully")
    print(f"   Total parameters: {total_params:,}")
except Exception as e:
    print(f"   ✗ Model creation failed: {e}")
    exit(1)

# Test 4: Test forward pass
print("\n4. Testing forward pass...")
try:
    # Create dummy input
    dummy_input = torch.randn(2, 3, 256, 256).to(device)
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        output = model(dummy_input)
    
    print(f"   ✓ Forward pass successful")
    print(f"   Input shape: {dummy_input.shape}")
    print(f"   Output shape: {output.shape}")
    
    # Check output shape
    expected_shape = (2, 3, 256, 256)
    assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"
    print(f"   ✓ Output shape correct")
except Exception as e:
    print(f"   ✗ Forward pass failed: {e}")
    exit(1)

# Test 5: Test metrics
print("\n5. Testing metrics...")
try:
    dummy_pred = torch.randn(2, 3, 256, 256).to(device)
    dummy_target = torch.randint(0, 3, (2, 256, 256)).to(device)
    
    dice, dice_per_class = dice_score(dummy_pred, dummy_target, num_classes=3)
    
    print(f"   ✓ Metrics calculation successful")
    print(f"   Dice score: {dice:.4f}")
    print(f"   Per-class: {[f'{d:.4f}' for d in dice_per_class]}")
except Exception as e:
    print(f"   ✗ Metrics test failed: {e}")
    exit(1)

# Test 6: Test predictor
print("\n6. Testing predictor...")
try:
    predictor = LaneSegmentationPredictor(model, device=device)
    
    # Create dummy image
    dummy_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    dummy_image = Image.fromarray(dummy_image)
    
    # Predict
    mask = predictor.predict(dummy_image)
    features = predictor.extract_lane_features(mask)
    steering = predictor.get_steering_signal(mask)
    
    print(f"   ✓ Predictor working correctly")
    print(f"   Mask shape: {mask.shape}")
    print(f"   Steering signal: {steering:.3f}")
except Exception as e:
    print(f"   ✗ Predictor test failed: {e}")
    exit(1)

# Test 7: Check configuration
print("\n7. Checking configuration...")
try:
    required_keys = ['train_images', 'train_labels', 'val_images', 'val_labels',
                     'img_size', 'num_classes', 'batch_size', 'num_epochs']
    
    for key in required_keys:
        assert key in CONFIG, f"Missing key: {key}"
    
    print(f"   ✓ Configuration valid")
    print(f"   Image size: {CONFIG['img_size']}")
    print(f"   Batch size: {CONFIG['batch_size']}")
    print(f"   Num epochs: {CONFIG['num_epochs']}")
except Exception as e:
    print(f"   ✗ Configuration check failed: {e}")
    exit(1)

print("\n" + "="*70)
print("ALL TESTS PASSED! ✓")
print("="*70)
print("\nYou're ready to train!")
print("Run: python main.py")
print("="*70 + "\n")
