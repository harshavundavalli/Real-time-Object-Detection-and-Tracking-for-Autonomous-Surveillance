# detection.py - Object detection module

import torch
import numpy as np
from ultralytics import YOLO

# Detection class for YOLOv8 object detection
class Detection:
    def __init__(self, model_path="yolov8n.pt", conf_threshold=0.5, device=None):
        
        self.conf_threshold = conf_threshold
        
        # Determine device (CPU or GPU)
        if device is None:
            self.device = 'mps' if torch.backends.mps.is_available() else 'cpu'
        else:
            self.device = device
        
        # Load YOLOv8 model
        self.model = YOLO(model_path)
        print(f"Model loaded on {self.device}")
    
    # Detect objects in a given frame
    def detect(self, frame):
        
        # Run inference
        results = self.model(frame, conf=self.conf_threshold, verbose=False)[0]
        
        # Process results
        detections = []
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            confidence = box.conf[0].cpu().numpy()
            class_id = int(box.cls[0].cpu().numpy())
            class_name = results.names[class_id]
            
            detection = [int(x1), int(y1), int(x2), int(y2), float(confidence), class_id, class_name]
            detections.append(detection)
        
        return detections
