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

1. Create a virtual environment (recommended)
   ```bash
   python -m venv myenv
   source myenv/bin/activate  # On macOS/Linux
   ```

2. Install the required packages
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

- If you see "MPS not available" warnings, you may be using an older version of macOS or PyTorch that doesn't support Metal. The system will fall back to CPU.

- For custom training issues, ensure your dataset follows the YOLO format exactly.


