# utils.py - Utility functions

import cv2
import numpy as np
import logging
import os
from datetime import datetime

# Utility function to draw bounding boxes and tracking information on a frame
def draw_boxes(frame, tracks):
    
    # Define colors for different classes (for better visualization)
    colors = {
        0: (0, 255, 0),    # Person - Green
        1: (255, 0, 0),    # Bicycle - Blue
        2: (0, 0, 255),    # Car - Red
        3: (255, 255, 0),  # Motorcycle - Yellow
        4: (0, 255, 255),  # Bus - Cyan
        5: (255, 0, 255),  # Truck - Magenta
        6: (128, 0, 0),    # Traffic Light - Dark Blue
        7: (0, 128, 0),    # Fire Hydrant - Dark Green
        8: (128, 128, 0),  # Other - Dark Yellow
    }
    
    # Draw boxes for each track
    for track in tracks:
        # Get bounding box
        x1, y1, x2, y2 = track.box
        
        # Get color based on class ID
        color = colors.get(track.class_id, (255, 255, 255))  
        
        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Draw track ID and class name
        text = f"ID: {track.track_id} - {track.class_name} ({track.confidence:.2f})"
        cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Draw centroid
        centroid_x, centroid_y = int(track.centroid[0]), int(track.centroid[1])
        cv2.circle(frame, (centroid_x, centroid_y), 4, color, -1)
    
    return frame

# setup_logger function to configure logging for the application
def setup_logger():
    logger = logging.getLogger('surveillance')
    logger.setLevel(logging.INFO)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    
    # Add handler to logger
    logger.addHandler(console_handler)
    
    # Create log directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    
    # Create file handler
    log_file = os.path.join('logs', f'surveillance_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    
    # Add file handler to logger
    logger.addHandler(file_handler)
    
    return logger
