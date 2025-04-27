# main.py - Main application file

import os
import cv2
import time
import numpy as np
import argparse
from pathlib import Path
import torch
from threading import Thread
import queue
import logging
from flask import Flask, Response, render_template, request, jsonify
import datetime

# Import custom modules
from detection import Detection
from tracking import Tracking
from utils import draw_boxes, setup_logger


# Set up logging
logger = setup_logger()

class VideoProcessor:
    def __init__(self, source=0, output_folder="output"):
        """
        Initialize the video processor
        
        Args:
            source (int or str): Camera index or video file path
            output_folder (str): Folder to save output frames
        """
        self.source = source
        self.output_folder = output_folder
        self.frame_queue = queue.Queue(maxsize=10)
        self.result_queue = queue.Queue(maxsize=10)
        self.stopped = False
        self.tracking_enabled = True  # Add tracking enabled flag
        
        # Create output directory if it doesn't exist
        os.makedirs(output_folder, exist_ok=True)
        
        # Set up detector and tracker
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
    
    def start(self):
        """Start the threads for processing video frames"""
        self.stopped = False
        Thread(target=self.read_frames, daemon=True).start()
        Thread(target=self.process_frames, daemon=True).start()
        return self
    
    def read_frames(self):
        """Read frames from the video source and put them in the queue"""
        while not self.stopped:
            if not self.frame_queue.full():
                ret, frame = self.cap.read()
                if not ret:
                    self.stopped = True
                    break
                self.frame_queue.put(frame)
            else:
                time.sleep(0.01)  # Sleep briefly to avoid CPU hogging
    
    def process_frames(self):
        """Process frames from the queue using detector and tracker"""
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

    def toggle_tracking(self, enabled=None):
        """Toggle or set tracking status"""
        if enabled is not None:
            self.tracking_enabled = enabled
        else:
            self.tracking_enabled = not self.tracking_enabled
        
        logger.info(f"Tracking {'enabled' if self.tracking_enabled else 'disabled'}")
        return self.tracking_enabled
    
    def log_tracks(self, tracks):
        """Log tracking information to CSV file"""
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
        with open(self.log_file, 'a') as f:
            for track in tracks:
                track_id = track.track_id
                class_name = track.class_name
                confidence = track.confidence
                bbox = track.to_tlbr()  # Get bounding box in format [top, left, bottom, right]
                
                # Convert to x, y, width, height format
                x = int(bbox[0])
                y = int(bbox[1])
                w = int(bbox[2] - bbox[0])
                h = int(bbox[3] - bbox[1])
                
                # Write to log file
                f.write(f"{timestamp},{track_id},{class_name},{confidence:.2f},{x},{y},{w},{h}\n")
    
    def get_frame(self):
        """Get the latest processed frame"""
        if not self.result_queue.empty():
            return self.result_queue.get()
        return None
    
    def stop(self):
        """Stop the video processor"""
        self.stopped = True
        if self.cap.isOpened():
            self.cap.release()
    
    def get_fps(self):
        """Get the current frames per second"""
        elapsed_time = time.time() - self.start_time
        current_fps = self.frame_count / elapsed_time if elapsed_time > 0 else 0
        return round(current_fps, 2)

# Flask web application for the dashboard
app = Flask(__name__)
video_processor = None

@app.route('/')
def index():
    """Render the dashboard template"""
    return render_template('index.html')

def generate_frames():
    """Generate frames for the video stream"""
    while not video_processor.stopped:
        frame = video_processor.get_frame()
        if frame is not None:
            # Convert frame to JPEG format
            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        else:
            time.sleep(0.01)

@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
  



@app.route('/logs')
def logs():
    """Return the latest log entries"""
    if os.path.exists(video_processor.log_file):
        with open(video_processor.log_file, 'r') as f:
            log_content = f.readlines()[-100:]  # Get last 100 lines
        return {'logs': log_content}
    return {'logs': []}

@app.route('/toggle_tracking')
def toggle_tracking():
    """Toggle tracking on/off"""
    enabled = request.args.get('enabled')
    if enabled is not None:
        # Convert string to boolean
        enabled = enabled.lower() == 'true'
        status = video_processor.toggle_tracking(enabled)
    else:
        status = video_processor.toggle_tracking()
    
    return jsonify({'status': 'enabled' if status else 'disabled'})

@app.route('/stats')
def stats():
    """Return current statistics"""
    if video_processor:
        # Get active tracks count
        active_tracks = 0
        for track in video_processor.tracker.tracks:
            if track.hits >= video_processor.tracker.min_hits:
                active_tracks += 1
                
        return jsonify({
            'fps': video_processor.get_fps(),
            'active_tracks': active_tracks,
            'uptime_seconds': int(time.time() - video_processor.start_time)
        })
    return jsonify({
        'fps': 0,
        'active_tracks': 0,
        'uptime_seconds': 0
    })

def main():
    """Main function to start the application"""
    global video_processor
    
    parser = argparse.ArgumentParser(description='Real-time Object Detection and Tracking')
    parser.add_argument('--source', type=str, default='0', help='Video source (0 for webcam, or file path)')
    parser.add_argument('--output', type=str, default='output', help='Output folder')
    parser.add_argument('--web', action='store_true', help='Run web dashboard')
    args = parser.parse_args()
    
    
    # Initialize video processor
    video_source = int(args.source) if args.source.isdigit() else args.source
    video_processor = VideoProcessor(source=video_source, output_folder=args.output)
    video_processor.start()

    if args.web:
        # Run Flask web server
        logger.info("Starting web dashboard on http://localhost:8080")
        app.run(host='0.0.0.0', port=8080, debug=False)
    else:
        # Run desktop version
        try:
            while not video_processor.stopped:
                frame = video_processor.get_frame()
                if frame is not None:
                    cv2.imshow('Real-time Surveillance', frame)
                    
                    # Break if 'q' is pressed
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                else:
                    time.sleep(0.01)
        except KeyboardInterrupt:
            pass
        finally:
            video_processor.stop()
            cv2.destroyAllWindows()

if __name__ == '__main__':
    main()