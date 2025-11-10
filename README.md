# Lane Segmentation for CARLA Driving Simulator

A deep learning-based lane segmentation system using UNet architecture for autonomous driving applications in the CARLA simulator.

## Project Structure

```
lane-segmentation/
├── config.py              # Configuration settings
├── dataset.py             # Dataset class and data loading
├── model.py               # UNet architecture
├── metrics.py             # Evaluation metrics (Dice, IoU, Accuracy)
├── train.py               # Training and validation functions
├── evaluate.py            # Evaluation and visualization
├── predict.py             # Inference/deployment predictor
├── utils.py               # Helper functions
├── main.py                # Main training script
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## Features

- **UNet Architecture**: Encoder-decoder architecture with skip connections for precise segmentation
- **Multi-class Segmentation**: Segments background, left lane, and right lane
- **Data Augmentation**: Horizontal flip and color jitter for robust training
- **Comprehensive Metrics**: Dice score, IoU, and pixel accuracy
- **Early Stopping**: Automatic training termination to prevent overfitting
- **Deployment Ready**: Predictor class for real-time inference

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/lane-segmentation.git
cd lane-segmentation

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Training

1. **Update Configuration**:
   Edit `config.py` to set your data paths and hyperparameters:
   ```python
   CONFIG = {
       'train_images': '/path/to/train/images',
       'train_labels': '/path/to/train/labels',
       'val_images': '/path/to/val/images',
       'val_labels': '/path/to/val/labels',
       # ... other settings
   }
   ```

2. **Run Training**:
   ```bash
   python main.py
   ```

### Inference

```python
from model import UNet
from predict import LaneSegmentationPredictor, load_model_for_inference

# Load model
model = load_model_for_inference('lane_segmentation_final.pth', UNet)

# Create predictor
predictor = LaneSegmentationPredictor(model, device='cuda')

# Predict on image
from PIL import Image
image = Image.open('test_image.png')
mask = predictor.predict(image)

# Extract lane features for RL
features = predictor.extract_lane_features(mask)
print(features)

# Get steering signal
steering = predictor.get_steering_signal(mask)
print(f"Steering: {steering}")
```

## Model Architecture

The UNet model consists of:
- **Encoder**: 4 downsampling blocks (64 → 128 → 256 → 512)
- **Bottleneck**: 1024 channels
- **Decoder**: 4 upsampling blocks with skip connections
- **Output**: 3-channel segmentation map (background, left lane, right lane)

Total parameters: ~31M

## Configuration

Key hyperparameters in `config.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `img_size` | (256, 256) | Input image size |
| `batch_size` | 16 | Training batch size |
| `num_epochs` | 50 | Maximum training epochs |
| `learning_rate` | 1e-3 | Initial learning rate |
| `patience` | 10 | Early stopping patience |
| `use_augmentation` | True | Enable data augmentation |

## Evaluation Metrics

- **Dice Score**: Measures overlap between prediction and ground truth
- **IoU (Intersection over Union)**: Ratio of intersection to union
- **Pixel Accuracy**: Percentage of correctly classified pixels
- **Per-class Metrics**: Individual scores for each lane class

## Dataset Format

Images and labels should be organized as follows:

```
train/
  ├── image1.png
  ├── image2.png
  └── ...
train_label/
  ├── image1_label.png
  ├── image2_label.png
  └── ...
```

Label format:
- 0: Background
- 1: Left lane
- 2: Right lane

## Integration with CARLA

Example integration with CARLA simulator:

```python
import carla
from predict import LaneSegmentationPredictor

# Setup CARLA
client = carla.Client('localhost', 2000)
world = client.get_world()

# Create predictor
predictor = LaneSegmentationPredictor(model, device='cuda')

# In your control loop
def process_camera_image(image):
    # Convert CARLA image to numpy
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = array.reshape((image.height, image.width, 4))[:, :, :3]
    
    # Predict lanes
    mask = predictor.predict(array)
    
    # Get steering signal
    steering = predictor.get_steering_signal(mask)
    
    return steering
```

## Results

Expected performance on validation set:
- Dice Score: ~0.85-0.90
- IoU: ~0.75-0.82
- Pixel Accuracy: ~0.92-0.95

## Next Steps

1. **RL Integration**: Use lane features as state input for reinforcement learning agent
2. **Real-time Optimization**: Implement model quantization for faster inference
3. **Multi-weather Support**: Train on diverse weather conditions
4. **Temporal Consistency**: Add recurrent connections for video sequences

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{lane-segmentation-carla,
  author = {Your Name},
  title = {Lane Segmentation for CARLA Driving Simulator},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/yourusername/lane-segmentation}
}
```

## License

MIT License

## Contact

For questions or issues, please open an issue on GitHub or contact [mukeshreddy662369@gmail.com]
