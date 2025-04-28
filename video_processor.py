
import cv2
import queue
from threading import Thread
import os
import time
import datetime
from detection import Detection
from tracking import Tracking
from utils import draw_boxes, setup_logger

logger = setup_logger()


# VideoProcessor class to handle video processing, detection, and tracking
class VideoProcessor:
    def __init__(self, source=0, output_folder="output"):
        
        self.source = source
        self.output_folder = output_folder
        self.frame_queue = queue.Queue(maxsize=10)
        self.result_queue = queue.Queue(maxsize=10)
        self.stopped = False
        self.tracking_enabled = True  
        
        # Creating output directory if it doesn't exist
        os.makedirs(output_folder, exist_ok=True)
        
        # Initializing up detector and tracker
        self.detector = Detection(model_path="yolov8n.pt")
        self.tracker = Tracking()
        
        # Initialize camera
        self.cap = cv2.VideoCapture(self.source)
        if not self.cap.isOpened():
            raise ValueError("Could not open video source")
        
        # Get video properties
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        
        # Initialize performance metrics
        self.frame_count = 0
        self.start_time = time.time()
        self.log_file = os.path.join(output_folder, 'tracking_log.csv')
        
        # Create log file header
        with open(self.log_file, 'w') as f:
            f.write("timestamp,track_id,class,confidence,x,y,width,height\n")
    
    # Start the video processing threads
    def start(self):
        self.stopped = False
        Thread(target=self.read_frames, daemon=True).start()
        Thread(target=self.process_frames, daemon=True).start()
        return self
    
    # Read frames from the video source and put them in the queue
    def read_frames(self):
        while not self.stopped:
            if not self.frame_queue.full():
                ret, frame = self.cap.read()
                if not ret:
                    self.stopped = True
                    break
                self.frame_queue.put(frame)
            else:
                time.sleep(0.01)  # Sleep briefly to avoid CPU hogging
    
    # Process frames from the queue using the detector and tracker
    def process_frames(self):
        while not self.stopped:
            if not self.frame_queue.empty() and not self.result_queue.full():
                frame = self.frame_queue.get()
                
                # Perform detection
                detections = self.detector.detect(frame)
                
                # Update tracker if tracking is enabled
                if self.tracking_enabled:
                    tracks = self.tracker.update(detections, frame)
                    # Draw bounding boxes with tracking
                    result_frame = draw_boxes(frame.copy(), tracks)
                    # Log tracking information
                    self.log_tracks(tracks)
                else:
                    tracks = []
                    # Draw only detection boxes without tracking
                    result_frame = frame.copy()
                    for det in detections:
                        x1, y1, x2, y2, conf, class_id, class_name = det
                        cv2.rectangle(result_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(result_frame, f"{class_name} ({conf:.2f})", (x1, y1 - 10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # Calculate FPS
                self.frame_count += 1
                elapsed_time = time.time() - self.start_time
                current_fps = self.frame_count / elapsed_time if elapsed_time > 0 else 0
                
                # Add FPS to the frame
                cv2.putText(result_frame, f"FPS: {current_fps:.2f}", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Add tracking status to the frame
                status_text = "Tracking: ON" if self.tracking_enabled else "Tracking: OFF"
                cv2.putText(result_frame, status_text, (10, 70), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Put the result in the result queue
                self.result_queue.put(result_frame)
            else:
                time.sleep(0.01)  # Sleep briefly to avoid CPU hogging

    # Toggle tracking on or off
    def toggle_tracking(self, enabled=None):
        if enabled is not None:
            self.tracking_enabled = enabled
        else:
            self.tracking_enabled = not self.tracking_enabled
        
        logger.info(f"Tracking {'enabled' if self.tracking_enabled else 'disabled'}")
        return self.tracking_enabled
    
    # Log tracking information to a CSV file
    def log_tracks(self, tracks):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
        with open(self.log_file, 'a') as f:
            for track in tracks:
                track_id = track.track_id
                class_name = track.class_name
                confidence = track.confidence
                box = track.to_tlbr()  # Get bounding box in format [top, left, bottom, right]
                
                # Convert to x, y, width, height format
                x = int(box[0])
                y = int(box[1])
                w = int(box[2] - box[0])
                h = int(box[3] - box[1])
                
                # Write to log file
                f.write(f"{timestamp},{track_id},{class_name},{confidence:.2f},{x},{y},{w},{h}\n")
    
    # Get the latest processed frame from the result queue
    def get_frame(self):
        if not self.result_queue.empty():
            return self.result_queue.get()
        return None
    
    # Stop the video processor and release resources
    def stop(self):
        self.stopped = True
        if self.cap.isOpened():
            self.cap.release()

    # Get the current frames per second    
    def get_fps(self):
        elapsed_time = time.time() - self.start_time
        current_fps = self.frame_count / elapsed_time if elapsed_time > 0 else 0
        return round(current_fps, 2)

    # Add to VideoProcessor class in video_processor.py
    def get_id_switches(self):
        return self.tracker.id_switches
