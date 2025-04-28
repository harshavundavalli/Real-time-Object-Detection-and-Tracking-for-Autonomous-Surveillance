#imports

import os
import cv2
import time
import argparse
from flask import Flask, Response, render_template, request, jsonify
from utils import draw_boxes, setup_logger
from video_processor import VideoProcessor

# Setting up logger
logger = setup_logger()

# Function to generate frames for the video stream
def generate_frames():
    while not video_processor.stopped:
        frame = video_processor.get_frame()
        
        if frame is not None:    
            # Converting frame to JPEG format
            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            
            # Yielding each frame in the required format for streaming
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        else:
            # If no frame is available, wait briefly to avoid overloading the CPU
            time.sleep(0.01)


# Flask web application for the dashboard
app = Flask(__name__)
video_processor = None

# Default route for the web application serving the index page
@app.route('/')
def index():
    return render_template('index.html')


# Api route to fetch and stream the video feed for real-time object detection and tracking
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
  


# Api route to fetch the latest log entries
@app.route('/logs')
def logs():
    if os.path.exists(video_processor.log_file):
        # Fetching the last 100 lines of the log file
        with open(video_processor.log_file, 'r') as f:
            log_content = f.readlines()[-100:]  
        return {'logs': log_content}
    return {'logs': []}

# Api route to toggle tracking on/off
@app.route('/toggle_tracking')
def toggle_tracking():
    enabled = request.args.get('enabled')
    if enabled is not None:
        enabled = enabled.lower() == 'true'
        status = video_processor.toggle_tracking(enabled)
    else:
        status = video_processor.toggle_tracking()
    
    return jsonify({'status': 'enabled' if status else 'disabled'})

# Api route to fetch the current statistics of the video processor like fp, axcitve streams and as
@app.route('/stats')
def stats():
    
    if video_processor:
        # Get active tracks count
        active_tracks = 0
        for track in video_processor.tracker.tracks:
            if track.hits >= video_processor.tracker.min_hits:
                active_tracks += 1
                
        return jsonify({
            'fps': video_processor.get_fps(),
            'active_tracks': active_tracks,
            'uptime_seconds': int(time.time() - video_processor.start_time),
            'id_switches': video_processor.get_id_switches()
        })
    return jsonify({
        'fps': 0,
        'active_tracks': 0,
        'uptime_seconds': 0,
        'id_switches': 0
    })



def main():
    """Main function to start the application"""
    global video_processor
    
    parser = argparse.ArgumentParser(description='Real-time Object Detection and Tracking')
    parser.add_argument('--source', type=str, default='0', help='Video source (0 for webcam, or file path)')
    parser.add_argument('--output', type=str, default='output', help='Output folder')
    parser.add_argument('--web', action='store_true', help='Run web dashboard')
   
    args = parser.parse_args()
    
    
    # Initializing the video processor
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