# Real-time Object Detection and Tracking for Autonomous Surveillance

A lightweight, real-time object detection and tracking system for security surveillance applications, optimized for macOS.

## Features

- Real-time object detection using YOLOv8
- Object tracking with unique IDs using a simplified DeepSORT approach
- Dashboard with live video feed and tracking logs
- Support for custom model training
- Optimized for macOS (Apple Silicon)

## Requirements

- Python 3.8 or higher
- macOS (tested on Apple Silicon)
- Webcam or video file for testing

## Installation

1. Clone this repository or download the files

2. Create a virtual environment (recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On macOS/Linux
   ```

3. Install the required packages
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Basic Usage (Desktop Window)

```bash
python main.py --source 0  # Use webcam
# or
python main.py --source path/to/video.mp4  # Use video file
```

### Web Dashboard

```bash
python main.py --source 0 --web  # Use webcam with web dashboard
```
Then open a browser and go to: http://localhost:5000

### Custom Training

1. Prepare your dataset in YOLO format:
   ```
   dataset/
   ├── train/
   │   ├── images/
   │   │   ├── img1.jpg
   │   │   ├── img2.jpg
   │   │   └── ...
   │   └── labels/
   │       ├── img1.txt
   │       ├── img2.txt
   │       └── ...
   └── val/
       ├── images/
       │   ├── img_val1.jpg
       │   ├── img_val2.jpg
       │   └── ...
       └── labels/
           ├── img_val1.txt
           ├── img_val2.txt
           └── ...
   ```

2. Create a `dataset.yaml` file in your dataset folder:
   ```yaml
   path: /path/to/dataset
   train: /path/to/dataset/train/images
   val: /path/to/dataset/val/images
   names:
     0: person
     1: car
     # Add your classes
   ```

3. Run the training:
   ```bash
   python main.py --train --dataset path/to/dataset
   ```

4. Once training is complete, use the custom model:
   ```bash
   python main.py --source 0 --model runs/train/custom_model/weights/best.pt
   ```

## Project Structure

- `main.py`: Main application file
- `detection.py`: Object detection module using YOLOv8
- `tracking.py`: Object tracking module with simplified DeepSORT
- `utils.py`: Utility functions for visualization and logging
- `dataset.py`: Custom dataset handling and training utilities
- `templates/index.html`: Web dashboard template

## Performance Notes

- The system is optimized for macOS with Apple Silicon using MPS (Metal Performance Shaders)
- For better performance on older Macs, reduce the resolution or use a smaller YOLOv8 model
- Expected performance:
  - M1/M2 Mac: 15-25 FPS at 640x480 resolution
  - Intel Mac: 5-15 FPS at 640x480 resolution

## Troubleshooting

- If you encounter errors related to PyTorch, try reinstalling with:
  ```bash
  pip uninstall torch
  pip install torch torchvision
  ```

- If you see "MPS not available" warnings, you may be using an older version of macOS or PyTorch that doesn't support Metal. The system will fall back to CPU.

- For custom training issues, ensure your dataset follows the YOLO format exactly.

## Limitations

- This is a simplified implementation for educational purposes
- The tracking algorithm is a simplified version of DeepSORT
- The system may not handle extremely crowded scenes well
- Custom training requires properly labeled data in YOLO format
