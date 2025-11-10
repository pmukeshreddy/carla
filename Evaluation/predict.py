"""
Lane Segmentation Predictor for deployment
"""

import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms


class LaneSegmentationPredictor:
    """
    Real-time lane segmentation predictor for deployment
    
    Usage in CARLA:
        predictor = LaneSegmentationPredictor(model, device)
        lane_mask = predictor.predict(camera_frame)
    """
    
    def __init__(self, model, device='cuda', img_size=(256, 256)):
        """
        Initialize predictor
        
        Args:
            model: Trained segmentation model
            device: Device to run inference on (cuda/cpu)
            img_size: Target image size (H, W)
        """
        self.model = model
        self.model.eval()
        self.device = device
        
        self.transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], 
                               [0.229, 0.224, 0.225])
        ])
    
    @torch.no_grad()
    def predict(self, image):
        """
        Predict lane segmentation mask
        
        Args:
            image: PIL Image or numpy array (H, W, 3)
        
        Returns:
            mask: numpy array (H, W) with values {0, 1, 2}
                  0 = Background
                  1 = Left Lane
                  2 = Right Lane
        """
        # Convert numpy array to PIL if needed
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        # Transform and add batch dimension
        img_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Predict
        output = self.model(img_tensor)
        pred = torch.argmax(output, dim=1).squeeze().cpu().numpy()
        
        return pred
    
    def extract_lane_features(self, mask):
        """
        Extract lane features for RL agent from segmentation mask
        
        Args:
            mask: Segmentation mask (H, W) with values {0, 1, 2}
        
        Returns:
            features: Dictionary with lane features
        """
        left_lane = (mask == 1)
        right_lane = (mask == 2)
        
        features = {
            'left_lane_pixels': int(left_lane.sum()),
            'right_lane_pixels': int(right_lane.sum()),
            'has_left_lane': bool(left_lane.any()),
            'has_right_lane': bool(right_lane.any()),
        }
        
        # Calculate lane center if both lanes are visible
        if features['has_left_lane'] and features['has_right_lane']:
            left_x = np.where(left_lane)[1].mean()
            right_x = np.where(right_lane)[1].mean()
            features['lane_center_x'] = float((left_x + right_x) / 2)
            features['lane_width'] = float(right_x - left_x)
        else:
            features['lane_center_x'] = None
            features['lane_width'] = None
        
        return features
    
    def get_steering_signal(self, mask, image_width=256):
        """
        Calculate steering signal based on lane position
        
        Args:
            mask: Segmentation mask (H, W)
            image_width: Width of the image
        
        Returns:
            steering: Steering signal in [-1, 1]
                     Negative = turn left, Positive = turn right
        """
        features = self.extract_lane_features(mask)
        
        if features['lane_center_x'] is not None:
            # Calculate deviation from image center
            image_center = image_width / 2
            deviation = features['lane_center_x'] - image_center
            
            # Normalize to [-1, 1]
            steering = -deviation / (image_width / 2)
            steering = np.clip(steering, -1, 1)
        else:
            # No lanes detected
            steering = 0.0
        
        return float(steering)


def load_model_for_inference(model_path, model_class, device='cuda', num_classes=3):
    """
    Load a trained model for inference
    
    Args:
        model_path: Path to saved model weights
        model_class: Model class (e.g., UNet)
        device: Device to load model on
        num_classes: Number of output classes
    
    Returns:
        model: Loaded model ready for inference
    """
    model = model_class(in_channels=3, num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    return model
