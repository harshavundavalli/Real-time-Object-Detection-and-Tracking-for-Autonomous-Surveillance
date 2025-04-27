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

   ```

## Project Structure

- `main.py`: Main application file
- `detection.py`: Object detection module using YOLOv8
- `tracking.py`: Object tracking module with simplified DeepSORT
- `utils.py`: Utility functions for visualization and logging
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

