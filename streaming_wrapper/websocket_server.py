import time
import threading
import json
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime
from collections import deque

try:
    from flask import Flask, Response, render_template, jsonify
    from flask_socketio import SocketIO, emit
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False
    SocketIO = None

import numpy as np


def json_safe(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [json_safe(x) for x in obj]
    return obj


@dataclass
class VideoFrame:
    frame_id: int
    timestamp: float
    image_data: Optional[bytes]
    caption: Optional[str]
    objects: List[Dict]


class WebSocketServer:
    def __init__(self, pipeline, host: str = '0.0.0.0', port: int = 5000):
        if not FLASK_AVAILABLE:
            raise ImportError("Flask and flask-socketio required")
        
        self.pipeline = pipeline
        self.host = host
        self.port = port
        
        self.app = Flask(__name__, template_folder='../templates', static_folder='../templates')
        self.app.config['SECRET_KEY'] = 'robot-perception-secret'
        
        self.socketio = SocketIO(self.app, cors_allowed_origins="*", async_mode='threading', ping_timeout=60, ping_interval=25)
        
        self._frame_lock = threading.Lock()
        self._latest_frame: Optional[VideoFrame] = None
        self._event_history: deque = deque(maxlen=100)
        self._running = False
        self._server_thread: Optional[threading.Thread] = None
        self._clients = {'events': 0, 'video': 0}
        self._client_lock = threading.Lock()
        
        self._setup_routes()
        self._setup_socket_events()
    
    def _setup_routes(self):
        @self.app.route('/')
        def index():
            return render_template('streaming.html')
        
        @self.app.route('/video_feed')
        def video_feed():
            return Response(self._generate_mjpeg(), mimetype='multipart/x-mixed-replace; boundary=frame')
        
        @self.app.route('/api/state')
        def get_state():
            result = self.pipeline.get_scene_state()
            return jsonify(json_safe(result.to_dict())) if result else jsonify({'error': 'No state'})
        
        @self.app.route('/api/events')
        def get_events():
            return jsonify([json_safe(e.to_dict()) for e in self._event_history])
        
        @self.app.route('/api/caption')
        def get_caption():
            return jsonify({'caption': self.pipeline.get_latest_caption() or 'Waiting...'})
        
        @self.app.route('/api/stats')
        def get_stats():
            return jsonify(json_safe({
                'frame_count': self.pipeline.frame_count,
                'caption_count': self.pipeline.caption_count,
                'latency': self.pipeline.get_latency_stats(),
                'clients': self._clients
            }))
    
    def _setup_socket_events(self):
        @self.socketio.on('connect', namespace='/')
        def handle_connect():
            with self._client_lock:
                self._clients['events'] += 1
            emit('connected', {'status': 'ok'})
        
        @self.socketio.on('disconnect', namespace='/')
        def handle_disconnect():
            with self._client_lock:
                self._clients['events'] = max(0, self._clients['events'] - 1)
        
        @self.socketio.on('request_state', namespace='/')
        def handle_state_request():
            result = self.pipeline.get_scene_state()
            if result:
                emit('state_update', result.to_dict())
        
        @self.socketio.on('request_events', namespace='/')
        def handle_events_request():
            emit('events_batch', {'events': [e.to_dict() for e in self._event_history]})
    
    def _generate_mjpeg(self):
        while self._running:
            with self._frame_lock:
                if self._latest_frame is None or self._latest_frame.image_data is None:
                    time.sleep(0.01)
                    continue
                frame_data = self._latest_frame.image_data
            if frame_data:
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame_data + b'\r\n')
            time.sleep(0.033)
        yield b''
    
    def start(self):
        if self._running:
            return
        self._running = True
        self.pipeline.on_event(self._handle_event)
        self.pipeline.on_caption(self._handle_caption)
        self.pipeline.start()
        self._server_thread = threading.Thread(target=self._run_server, daemon=True, name="WebSocketServer")
        self._server_thread.start()
        self._frame_thread = threading.Thread(target=self._frame_update_loop, daemon=True, name="FrameUpdater")
        self._frame_thread.start()
        print(f"[WebSocket] Server started on http://{self.host}:{self.port}")
    
    def _run_server(self):
        self.socketio.run(self.app, host=self.host, port=self.port, debug=False, use_reloader=False, allow_unsafe_werkzeug=True)
    
    def _frame_update_loop(self):
        import cv2
        while self._running:
            try:
                self._update_frame_periodic()
            except Exception as e:
                print(f"[WebSocket] Frame update error: {e}")
            time.sleep(0.033)
    
    def stop(self):
        self._running = False
        self.pipeline.stop()
        print("[WebSocket] Server stopped")
    
    def _handle_event(self, event):
        self._event_history.append(event)
        self.socketio.emit('event', event.to_dict(), namespace='/')
    
    def _handle_caption(self, caption, scene_context, result):
        self.socketio.emit('caption', {'caption': caption, 'scene': scene_context.to_dict()}, namespace='/')
    
    def _update_frame_periodic(self):
        import cv2
        raw_frame = self.pipeline.get_latest_frame()
        if raw_frame is None:
            return
        result = self.pipeline.get_scene_state()
        
        with self._frame_lock:
            frame_id = result.frame_id if result else 0
            if self._latest_frame and self._latest_frame.frame_id == frame_id:
                self._latest_frame.caption = self.pipeline.get_latest_caption()
                return
            
            h, w = raw_frame.shape[:2]
            scale = 640 / max(w, h)
            frame = cv2.resize(raw_frame, (int(w*scale), int(h*scale))) if scale < 1 else raw_frame.copy()
            
            action_phase_map = {}
            if result and result.scene_context:
                for obj in result.scene_context.active_objects:
                    action_phase_map[obj['id']] = obj.get('action_phase', 'stable')
            
            objects_list = []
            if result:
                for track in result.tracks[:self.pipeline.max_tracked]:
                    state = result.states.get(track.track_id)
                    action_phase = action_phase_map.get(track.track_id, 'stable')
                    color = (0, 0, 255) if action_phase == 'in_progress' else (0, 255, 0)
                    
                    if scale < 1:
                        x, y, bw, bh = [int(v * scale) for v in track.bbox]
                    else:
                        x, y, bw, bh = track.bbox
                    
                    x, y = max(0, x), max(0, y)
                    bw, bh = min(frame.shape[1]-x, bw), min(frame.shape[0]-y, bh)
                    
                    cv2.rectangle(frame, (x, y), (x+bw, y+bh), color, 2)
                    label = f"ID{track.track_id}:{track.class_name[:3]} {'?' if action_phase == 'in_progress' else '✓'}"
                    cv2.putText(frame, label, (x, max(y-5, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                    
                    obj_dict = track.to_dict()
                    obj_dict['action_phase'] = action_phase
                    objects_list.append(obj_dict)
            
            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 50])
            if ret:
                self._latest_frame = VideoFrame(
                    frame_id=frame_id, timestamp=time.time(), image_data=buffer.tobytes(),
                    caption=self.pipeline.get_latest_caption(), objects=objects_list
                )
    
    @property
    def is_running(self) -> bool:
        return self._running
