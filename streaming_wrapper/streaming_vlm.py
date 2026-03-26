import time
import queue
import threading
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime

from .detector import ObjectDetector, create_robot_detector
from .tracker import SortTracker, Track
from .state_estimator import StateEstimator, ObjectState
from .predictor import TrajectoryPredictor, TrajectoryPrediction
from .event_generator import EventGenerator, Event, EventType, SceneContext
from .prompts import PromptBuilder, PromptContext


@dataclass
class PipelineResult:
    timestamp: float
    frame_id: int
    events: List[Event]
    caption: Optional[str]
    scene_context: SceneContext
    tracks: List[Track]
    states: Dict[int, ObjectState]
    predictions: Dict[int, TrajectoryPrediction]
    latency_ms: float
    
    def to_dict(self) -> Dict:
        return {
            'timestamp': datetime.fromtimestamp(self.timestamp).isoformat(),
            'frame_id': int(self.frame_id),
            'event_count': len(self.events),
            'caption': self.caption,
            'scene': self.scene_context.to_dict(),
            'tracks': [t.to_dict() for t in self.tracks],
            'objects': {int(tid): {'state': s.to_dict(), 'prediction': p.to_dict() if p else None}
                        for tid, s, p in [(t.track_id, self.states.get(t.track_id), self.predictions.get(t.track_id)) for t in self.tracks]},
            'latency_ms': float(self.latency_ms)
        }


@dataclass 
class EventOutput:
    event_type: str
    event_id: str
    timestamp: str
    data: Dict[str, Any]
    
    def to_dict(self) -> Dict:
        return {'event_type': self.event_type, 'event_id': self.event_id, 'timestamp': self.timestamp, **self.data}


class StreamingVLM:
    def __init__(self, device: str = 'cuda', max_tracked: int = 10, prediction_horizon: float = 2.0,
                 vlm_max_tokens: int = 32, frame_skip: int = 1, vlm_enabled: bool = True):
        self.device = device
        self.max_tracked = max_tracked
        self.prediction_horizon = prediction_horizon
        self.vlm_max_tokens = vlm_max_tokens
        self.frame_skip = frame_skip
        self._vlm_enabled = vlm_enabled
        
        self.detector: Optional[ObjectDetector] = None
        self.tracker: Optional[SortTracker] = None
        self.state_estimator: Optional[StateEstimator] = None
        self.predictor: Optional[TrajectoryPredictor] = None
        self.event_generator: Optional[EventGenerator] = None
        self.prompt_builder: Optional[PromptBuilder] = None
        self.vlm_model = None
        
        self._running = False
        self._processing_thread = None
        self._vlm_thread = None
        
        self._frame_queue = queue.Queue(maxsize=5)
        self._event_queue = queue.Queue(maxsize=100)
        self._caption_queue = queue.Queue(maxsize=5)
        
        self._latest_result: Optional[PipelineResult] = None
        self._latest_frame: Optional[np.ndarray] = None
        self._latest_caption: Optional[str] = None
        self._caption_lock = threading.Lock()
        
        self._frame_count = 0
        self._caption_count = 0
        self._vlm_callbacks: List[callable] = []
        self._event_callbacks: List[callable] = []
        self._detector_latency = 0.0
        self._tracker_latency = 0.0
        self._vlm_latency = 0.0
    
    def load(self) -> bool:
        print("=" * 60)
        print("Loading StreamingVLM Pipeline")
        print("=" * 60)
        
        print("\n[1/5] Loading object detector...")
        self.detector = create_robot_detector(device=self.device)
        if not self.detector.load():
            print("[ERROR] Failed to load detector")
            return False
        print(f"       Detector ready ({self.detector.latency_ms:.1f}ms)")
        
        print("\n[2/5] Initializing tracker...")
        self.tracker = SortTracker(max_age=30, min_hits=1, iou_threshold=0.3)
        print("       Tracker ready")
        
        print("\n[3/5] Initializing state estimator...")
        self.state_estimator = StateEstimator(position_scale=100.0, smoothing=0.3, vel_threshold=5.0)
        print("       State estimator ready")
        
        print("\n[4/5] Initializing trajectory predictor...")
        self.predictor = TrajectoryPredictor(horizon_s=self.prediction_horizon, dt_s=0.2)
        print(f"       Predictor ready ({self.prediction_horizon}s horizon)")
        
        print("\n[5/5] Initializing event generator...")
        self.event_generator = EventGenerator(scene_update_interval=30, max_tracked=self.max_tracked)
        self.prompt_builder = PromptBuilder(max_objects=self.max_tracked)
        
        if self._vlm_enabled:
            if not self._load_vlm():
                print("       VLM not available - detection/tracking only")
        else:
            print("       VLM disabled by user")
        
        print("\n" + "=" * 60)
        print("Pipeline loaded successfully" if self.detector.is_ready else "Pipeline loaded (detector only)")
        print("=" * 60)
        return self.detector.is_ready
    
    def _load_vlm(self) -> bool:
        try:
            from model_inference import ModelInference
            self.vlm_model = ModelInference(n_ctx=2048)
            if self.vlm_model.load():
                self._vlm_enabled = True
                return True
            self.vlm_model = None
            self._vlm_enabled = False
            return False
        except ImportError:
            print("       model_inference not found - VLM unavailable")
            self._vlm_enabled = False
            return False
        except Exception as e:
            print(f"       VLM load error: {e}")
            self._vlm_enabled = False
            return False
    
    def start(self):
        if self._running:
            return
        self._running = True
        self._processing_thread = threading.Thread(target=self._process_loop, daemon=True, name="PipelineProcess")
        self._processing_thread.start()
        if self.vlm_model:
            self._vlm_thread = threading.Thread(target=self._vlm_loop, daemon=True, name="VLMProcess")
            self._vlm_thread.start()
        print("[Pipeline] Started")
    
    def stop(self):
        self._running = False
        if self._processing_thread:
            self._processing_thread.join(timeout=5)
        if self._vlm_thread:
            self._vlm_thread.join(timeout=5)
        print("[Pipeline] Stopped")
    
    def submit_frame(self, frame: np.ndarray) -> bool:
        if not self._running:
            return False
        self._frame_count += 1
        if self._frame_count % self.frame_skip != 0:
            return True
        try:
            self._frame_queue.put_nowait(frame)
            return True
        except queue.Full:
            return False
    
    def _process_loop(self):
        while self._running:
            try:
                frame = self._frame_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            
            start_time = time.time()
            self._latest_frame = frame.copy()
            detections = self.detector.detect(frame)
            det_time = time.time()
            tracks = self.tracker.update(detections)
            track_time = time.time()
            states = self.state_estimator.update(tracks)
            states_time = time.time()
            predictions = self.predictor.predict(states)
            pred_time = time.time()
            events, scene_context = self.event_generator.process(tracks, states, predictions)
            
            self._detector_latency = (det_time - start_time) * 1000
            self._tracker_latency = (track_time - det_time) * 1000
            
            result = PipelineResult(
                timestamp=time.time(), frame_id=self._frame_count, events=events, caption=None,
                scene_context=scene_context, tracks=tracks, states=states, predictions=predictions,
                latency_ms=(time.time() - start_time) * 1000
            )
            self._latest_result = result
            
            for event in events:
                self._emit_event(event)
            
            caption_worthy = {EventType.NEW_OBJECT, EventType.OBJECT_LEFT, EventType.ACTION_COMPLETE}
            if self._vlm_enabled and scene_context.active_objects and any(e.event_type in caption_worthy for e in events):
                try:
                    self._caption_queue.put_nowait((frame.copy(), scene_context, result))
                except queue.Full:
                    pass
    
    def _vlm_loop(self):
        while self._running:
            try:
                frame, scene_context, result = self._caption_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            
            start_time = time.time()
            caption = self._generate_caption(frame, scene_context)
            self._vlm_latency = (time.time() - start_time) * 1000
            
            with self._caption_lock:
                self._latest_caption = caption
            self._caption_count += 1
            
            for callback in self._vlm_callbacks:
                try:
                    callback(caption, scene_context, result)
                except Exception as e:
                    print(f"[VLM] Callback error: {e}")
            result.caption = caption
    
    def _generate_caption(self, frame: np.ndarray, scene_context: SceneContext) -> str:
        if not self.vlm_model or not self.vlm_model.is_ready:
            return self._generate_fallback_caption(scene_context)
        
        try:
            prompt_context = PromptContext(
                scene_description=scene_context.summary,
                active_objects=scene_context.active_objects,
                new_objects=[o['id'] for o in scene_context.active_objects if self._is_new_object(o['id'])],
                recent_events=[e.event_type.value for e in self.event_generator.get_recent_events(3)]
            )
            prompt = self.prompt_builder.build_scene_prompt(prompt_context)
            
            import base64
            from io import BytesIO
            from PIL import Image
            
            frame_rgb = frame[:, :, ::-1]
            pil_image = Image.fromarray(frame_rgb)
            buffer = BytesIO()
            pil_image.save(buffer, format='JPEG', quality=60)
            image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            image_data_uri = f"data:image/jpeg;base64,{image_base64}"
            
            messages = [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": image_data_uri}}, {"type": "text", "text": prompt}]}]
            
            response = self.vlm_model.model.create_chat_completion(
                messages=messages, max_tokens=self.vlm_max_tokens, temperature=0.0, stop=["<|im_end|>", "</s>"]
            )
            
            if response and "choices" in response:
                return response["choices"][0]["message"]["content"].strip()
            return self._generate_fallback_caption(scene_context)
        except Exception as e:
            print(f"[VLM] Caption error: {e}")
            return self._generate_fallback_caption(scene_context)
    
    def _generate_fallback_caption(self, scene_context: SceneContext) -> str:
        if not scene_context.active_objects:
            return "No objects detected"
        parts = [f"{obj.get('class', 'object')} {obj.get('motion_state', 'present')}" for obj in scene_context.active_objects[:3]]
        return f"Detected: {', '.join(parts)}"
    
    def _is_new_object(self, obj_id: int) -> bool:
        for event in self.event_generator.get_recent_events(30):
            if event.event_type == EventType.NEW_OBJECT and event.data.get('object', {}).get('id') == obj_id:
                return True
        return False
    
    def _emit_event(self, event: Event):
        output = EventOutput(event_type=event.event_type.value, event_id=event.event_id,
                           timestamp=datetime.fromtimestamp(event.timestamp).isoformat(), data=event.data)
        for callback in self._event_callbacks:
            try:
                callback(output)
            except Exception as e:
                print(f"[Event] Callback error: {e}")
        try:
            self._event_queue.put_nowait(output)
        except queue.Full:
            pass
    
    def on_event(self, callback: callable):
        self._event_callbacks.append(callback)
    
    def on_caption(self, callback: callable):
        self._vlm_callbacks.append(callback)
    
    def get_events(self, max_events: int = 10) -> List[EventOutput]:
        events = []
        while len(events) < max_events:
            try:
                events.append(self._event_queue.get_nowait())
            except queue.Empty:
                break
        return events
    
    def get_latest_caption(self) -> Optional[str]:
        with self._caption_lock:
            return self._latest_caption
    
    def get_scene_state(self) -> Optional[PipelineResult]:
        return self._latest_result
    
    @property
    def is_running(self) -> bool:
        return self._running
    
    @property
    def frame_count(self) -> int:
        return self._frame_count
    
    @property
    def caption_count(self) -> int:
        return self._caption_count
    
    def get_latency_stats(self) -> Dict[str, float]:
        return {'detector_ms': self._detector_latency, 'tracker_ms': self._tracker_latency, 'vlm_ms': self._vlm_latency}
    
    def get_latest_frame(self) -> Optional[np.ndarray]:
        return self._latest_frame
