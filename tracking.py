# tracking.py - Object tracking module

import numpy as np
import cv2
from scipy.optimize import linear_sum_assignment

class TrackingObject:
    """Object tracking state representation"""
    def __init__(self, box, confidence, class_id, class_name):
        self.box = box  # [x1, y1, x2, y2]
        self.confidence = confidence
        self.class_id = class_id
        self.class_name = class_name
        self.track_id = -1
        self.time_since_update = 0
        self.hits = 0
        self.age = 0
        
        # Calculate centroid
        self.centroid = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
        
        # Initialize Kalman filter state
        self.kf = KalmanFilter()
        self.kf.initialize_state(self.centroid[0], self.centroid[1], 0, 0)
    
    def update(self, box, confidence, class_id, class_name):
        """Update tracking object with new detection"""
        self.box = box
        self.confidence = confidence
        self.class_id = class_id
        self.class_name = class_name
        self.time_since_update = 0
        self.hits += 1
        
        # Update centroid
        self.centroid = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
        
        # Update Kalman filter with measurement
        self.kf.update(self.centroid[0], self.centroid[1])
    
    def predict(self):
        """Predict new position using Kalman filter"""
        x, y = self.kf.predict()
        return x, y
    
    def to_tlbr(self):
        """Convert bounding box to top-left bottom-right format"""
        return self.box


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


# Tracking class for managing multiple object tracks
class Tracking:
    def __init__(self, max_age=30, min_hits=3, iou_threshold=0.3):
        
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.tracks = []
        self.next_id = 1

        # Add these lines for ID switch tracking
        self.prev_tracks = []  # Tracks from previous frame
        self.id_switches = 0   # Counter for ID switches
    
    # Method to calculate Intersection over Union (IOU) between two bounding boxes
    def iou(self, box1, box2):
        # Extract coordinates
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
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
    
    # Method to detect ID switches between frames
    def detect_id_switches(self, current_tracks):

        if not self.prev_tracks:
            self.prev_tracks = current_tracks.copy()
            return 0
            
        switches = 0
        
        # Check each previous track against current tracks
        for prev_track in self.prev_tracks:
            for curr_track in current_tracks:
                # If high IoU but different IDs -> ID switch detected
                iou_value = self.iou(prev_track.box, curr_track.box)
                if (iou_value > 0.8 and  # Higher threshold for ID switch detection
                    prev_track.track_id != curr_track.track_id and
                    prev_track.class_id == curr_track.class_id):
                    switches += 1
                    break
        
        # Store current tracks for next comparison
        self.prev_tracks = current_tracks.copy()
        return switches
    
    # Update method to process detections and manage tracks
    def update(self, detections, frame=None):
        
        # Create detection objects
        detection_objects = []
        for det in detections:
            x1, y1, x2, y2, confidence, class_id, class_name = det
            box = [x1, y1, x2, y2]
            detection_objects.append(TrackingObject(box, confidence, class_id, class_name))
        
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
                    cost_matrix[i, j] = 1 - self.iou(track.box, det.box)
            
            # Solve assignment problem
            row_indices, col_indices = linear_sum_assignment(cost_matrix)
            
            # Update matched tracks
            for row, col in zip(row_indices, col_indices):
                if cost_matrix[row, col] < 1 - self.iou_threshold:  # Convert IOU threshold to cost
                    track = self.tracks[row]
                    det = detection_objects[col]
                    track.update(det.box, det.confidence, det.class_id, det.class_name)
                    # Mark as matched
                    detection_objects[col] = None
            
            # Remove matched detections
            detection_objects = [det for det in detection_objects if det is not None]
        
        # Create new tracks for unmatched detections
        for detection_object_track in detection_objects:
            track = detection_object_track
            track.track_id = self.next_id
            self.next_id += 1
            self.tracks.append(track)
        
        # Remove old tracks
        self.tracks = [track for track in self.tracks if track.time_since_update <= self.max_age]
        
        # Return active tracks (tracks with enough hits)
        active_tracks = [track for track in self.tracks if track.hits >= self.min_hits]
        # Add this line to detect ID switches
        switches = self.detect_id_switches(active_tracks)
        self.id_switches += switches
        return active_tracks
