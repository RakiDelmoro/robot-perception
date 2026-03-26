#!/usr/bin/env python3
"""
Webcam Streaming Server for Docker Container Access
Run this on your Windows HOST machine (not in the container)
Streams webcam to http://localhost:8080/video_feed
"""

import cv2
from flask import Flask, Response
import threading

app = Flask(__name__)
camera = None
lock = threading.Lock()


def get_camera():
    """Initialize webcam"""
    global camera
    if camera is None or not camera.isOpened():
        camera = cv2.VideoCapture(0)
        if camera.isOpened():
            camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            camera.set(cv2.CAP_PROP_FPS, 30)
            print("Webcam initialized successfully!")
        else:
            print("ERROR: Could not open webcam")
            return None
    return camera


def generate_frames():
    """Generate MJPEG stream"""
    cam = get_camera()
    if cam is None:
        return

    while True:
        success, frame = cam.read()
        if not success:
            break

        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ret:
            continue

        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/')
def index():
    """Simple status page"""
    return """
    <h1>Webcam Server Running</h1>
    <p>Webcam stream available at: <code>http://host.docker.internal:8080/video_feed</code></p>
    <p>Container can access this URL for real-time classification</p>
    <img src="/video_feed" width="640" height="480">
    """


if __name__ == '__main__':
    print("=" * 50)
    print("Webcam Streaming Server")
    print("=" * 50)
    print("This server streams your webcam to the container")
    print("")
    print("URLs:")
    print("  - Status: http://localhost:8080")
    print("  - Stream: http://localhost:8080/video_feed")
    print("")
    print("Container should connect to: http://host.docker.internal:8080/video_feed")
    print("=" * 50)

    cam = get_camera()
    if cam is None:
        print("\nERROR: No webcam detected!")
        exit(1)

    print("\nStarting server...")
    print("Keep this window open while running the classification container")
    print("Press Ctrl+C to stop\n")

    try:
        app.run(host='0.0.0.0', port=8080, debug=False, threaded=True)
    except KeyboardInterrupt:
        print("\nShutting down...")
        if camera:
            camera.release()
        print("Goodbye!")
