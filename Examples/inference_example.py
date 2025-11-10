"""
Quick inference example
Usage: python inference_example.py --image path/to/image.png --model path/to/model.pth
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch

from model import UNet
from predict import LaneSegmentationPredictor, load_model_for_inference


def visualize_result(image, mask):
    """Visualize the segmentation result"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    axes[0].imshow(image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Segmentation mask
    axes[1].imshow(mask, cmap='tab10', vmin=0, vmax=2)
    axes[1].set_title('Segmentation (0=BG, 1=Left, 2=Right)')
    axes[1].axis('off')
    
    # Overlay
    overlay = np.array(image).copy()
    overlay[mask == 1] = [255, 0, 0]  # Left lane = Red
    overlay[mask == 2] = [0, 0, 255]  # Right lane = Blue
    axes[2].imshow(overlay)
    axes[2].set_title('Overlay (Red=Left, Blue=Right)')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Lane Segmentation Inference')
    parser.add_argument('--image', type=str, required=True, 
                       help='Path to input image')
    parser.add_argument('--model', type=str, default='lane_segmentation_final.pth',
                       help='Path to model weights')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to run inference on (cuda/cpu)')
    parser.add_argument('--save', type=str, default=None,
                       help='Path to save result (optional)')
    
    args = parser.parse_args()
    
    # Check device availability
    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = 'cpu'
    
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading model from {args.model}...")
    model = load_model_for_inference(args.model, UNet, device=device, num_classes=3)
    
    # Create predictor
    predictor = LaneSegmentationPredictor(model, device=device)
    
    # Load image
    print(f"Loading image from {args.image}...")
    image = Image.open(args.image).convert('RGB')
    
    # Predict
    print("Running inference...")
    mask = predictor.predict(image)
    
    # Extract features
    features = predictor.extract_lane_features(mask)
    steering = predictor.get_steering_signal(mask)
    
    # Print results
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"Lane Features:")
    print(f"  Has left lane:      {features['has_left_lane']}")
    print(f"  Has right lane:     {features['has_right_lane']}")
    print(f"  Left lane pixels:   {features['left_lane_pixels']}")
    print(f"  Right lane pixels:  {features['right_lane_pixels']}")
    if features['lane_center_x'] is not None:
        print(f"  Lane center X:      {features['lane_center_x']:.2f}")
        print(f"  Lane width:         {features['lane_width']:.2f}")
    print(f"\nSteering signal: {steering:.3f}")
    print("="*60 + "\n")
    
    # Visualize
    visualize_result(image, mask)
    
    # Save if requested
    if args.save:
        # Create overlay
        overlay = np.array(image).copy()
        overlay[mask == 1] = [255, 0, 0]
        overlay[mask == 2] = [0, 0, 255]
        
        result_image = Image.fromarray(overlay)
        result_image.save(args.save)
        print(f"✓ Result saved to {args.save}")


if __name__ == "__main__":
    main()
