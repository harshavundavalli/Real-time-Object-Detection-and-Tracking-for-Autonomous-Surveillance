# tracking.py - Object tracking module

import numpy as np
import cv2
from scipy.optimize import linear_sum_assignment

class TrackingObject:
    """Object tracking state representation"""
    def __init__(self, bbox, confidence, class_id, class_name):
        self.bbox = bbox  # [x1, y1, x2, y2]
        self.confidence = confidence
        self.class_id = class_id
        self.class_name = class_name
        self.track_id = -1
        self.time_since_update = 0
        self.hits = 0
        self.age = 0
        
        # Calculate centroid
        self.centroid = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
        
        # Initialize Kalman filter state
        self.kf = KalmanFilter()
        self.kf.initialize_state(self.centroid[0], self.centroid[1], 0, 0)
    
    def update(self, bbox, confidence, class_id, class_name):
        """Update tracking object with new detection"""
        self.bbox = bbox
        self.confidence = confidence
        self.class_id = class_id
        self.class_name = class_name
        self.time_since_update = 0
        self.hits += 1
        
        # Update centroid
        self.centroid = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
        
        # Update Kalman filter with measurement
        self.kf.update(self.centroid[0], self.centroid[1])
    
    def predict(self):
        """Predict new position using Kalman filter"""
        x, y = self.kf.predict()
        return x, y
    
    def to_tlbr(self):
        """Convert bounding box to top-left bottom-right format"""
        return self.bbox


class KalmanFilter:
    """Simple Kalman filter implementation for 2D tracking"""
    def __init__(self):
        # State: [x, y, vx, vy]
        self.state = np.zeros(4)
        
        # State transition matrix
        self.F = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        
        # Measurement matrix (we only measure position)
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])
        
        # Process noise covariance
        self.Q = np.eye(4) * 0.1
        
        # Measurement noise covariance
        self.R = np.eye(2) * 1.0
        
        # Error covariance matrix
        self.P = np.eye(4) * 10
    
    def initialize_state(self, x, y, vx=0, vy=0):
        """Initialize filter state"""
        self.state = np.array([x, y, vx, vy])
    
    def predict(self):
        """Predict next state"""
        # Predict state
        self.state = self.F @ self.state
        
        # Update error covariance
        self.P = self.F @ self.P @ self.F.T + self.Q
        
        return self.state[0], self.state[1]
    
    def update(self, x, y):
        """Update state with measurement"""
        # Measurement
        z = np.array([x, y])
        
        # Innovation
        y = z - self.H @ self.state
        
        # Innovation covariance
        S = self.H @ self.P @ self.H.T + self.R
        
        # Kalman gain
        K = self.P @ self.H.T @ np.linalg.inv(S)
        
        # Update state
        self.state = self.state + K @ y
        
        # Update error covariance
        I = np.eye(4)
        self.P = (I - K @ self.H) @ self.P


class Tracking:
    """Object tracking class"""
    def __init__(self, max_age=30, min_hits=3, iou_threshold=0.3):
        """
        Initialize tracker
        
        Args:
            max_age (int): Maximum frames to keep track without detection
            min_hits (int): Minimum hits to confirm a track
            iou_threshold (float): IOU threshold for matching
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.tracks = []
        self.next_id = 1
    
    def iou(self, bbox1, bbox2):
        """Calculate IOU between two bounding boxes"""
        # Extract coordinates
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Calculate area of bounding boxes
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        
        # Calculate intersection coordinates
        x_left = max(x1_1, x1_2)
        y_top = max(y1_1, y1_2)
        x_right = min(x2_1, x2_2)
        y_bottom = min(y2_1, y2_2)
        
        # Calculate intersection area
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        
        # Calculate union area
        union_area = area1 + area2 - intersection_area
        
        # Calculate IOU
        iou = intersection_area / union_area if union_area > 0 else 0
        
        return iou
    
    def update(self, detections, frame=None):
        """
        Update tracks with new detections
        
        Args:
            detections (list): List of detections [x1, y1, x2, y2, confidence, class_id, class_name]
            frame (numpy.ndarray): Current frame (unused, kept for compatibility)
            
        Returns:
            List of active tracks
        """
        # Create detection objects
        detection_objects = []
        for det in detections:
            x1, y1, x2, y2, confidence, class_id, class_name = det
            bbox = [x1, y1, x2, y2]
            detection_objects.append(TrackingObject(bbox, confidence, class_id, class_name))
        
        # Update track age
        for track in self.tracks:
            track.age += 1
            track.time_since_update += 1
            
            # Predict new position
            track.predict()
        
        # Match detections to tracks
        if len(self.tracks) > 0 and len(detection_objects) > 0:
            # Create cost matrix using IOU
            cost_matrix = np.zeros((len(self.tracks), len(detection_objects)))
            for i, track in enumerate(self.tracks):
                for j, det in enumerate(detection_objects):
                    cost_matrix[i, j] = 1 - self.iou(track.bbox, det.bbox)
            
            # Solve assignment problem
            row_indices, col_indices = linear_sum_assignment(cost_matrix)
            
            # Update matched tracks
            for row, col in zip(row_indices, col_indices):
                if cost_matrix[row, col] < 1 - self.iou_threshold:  # Convert IOU threshold to cost
                    track = self.tracks[row]
                    det = detection_objects[col]
                    track.update(det.bbox, det.confidence, det.class_id, det.class_name)
                    # Mark as matched
                    detection_objects[col] = None
            
            # Remove matched detections
            detection_objects = [det for det in detection_objects if det is not None]
        
        # Create new tracks for unmatched detections
        for det in detection_objects:
            track = TrackingObject(det.bbox, det.confidence, det.class_id, det.class_name)
            track.track_id = self.next_id
            self.next_id += 1
            self.tracks.append(track)
        
        # Remove old tracks
        self.tracks = [track for track in self.tracks if track.time_since_update <= self.max_age]
        
        # Return active tracks (tracks with enough hits)
        active_tracks = [track for track in self.tracks if track.hits >= self.min_hits]
        
        return active_tracks
