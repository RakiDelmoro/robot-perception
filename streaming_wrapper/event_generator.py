import time
import uuid
import numpy as np
from typing import Dict, List, Set, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum


class EventType(Enum):
    NEW_OBJECT = "new_object"
    OBJECT_LEFT = "object_left"
    ACTION_COMPLETE = "action_complete"
    CAPTION_UPDATE = "caption_update"


@dataclass
class Event:
    event_type: EventType
    event_id: str
    timestamp: float
    frame_id: int
    data: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            'event_type': self.event_type.value,
            'event_id': self.event_id,
            'timestamp': float(self.timestamp),
            'frame_id': int(self.frame_id),
            **{k: int(v) if isinstance(v, np.integer) else float(v) if isinstance(v, np.floating) else v for k, v in self.data.items()}
        }


@dataclass
class SceneContext:
    active_objects: List[Dict] = field(default_factory=list)
    summary: str = ""
    
    def to_dict(self) -> Dict:
        return {
            'active_objects': [{
                'id': int(obj['id']),
                'class': obj['class'],
                'motion_state': obj['motion_state'],
                'speed_ms': float(obj['speed_ms']),
                'intent': obj['intent']
            } for obj in self.active_objects],
            'summary': self.summary
        }


class EventGenerator:
    def __init__(self, scene_update_interval: int = 30, max_tracked: int = 10):
        self.scene_update_interval = scene_update_interval
        self.max_tracked = max_tracked
        self._frame_id = 0
        self._prev_track_ids: Set[int] = set()
        self._lost_tracks: Dict[int, int] = {}
        self._event_history: List[Event] = []
        self._max_event_history = 100
        self._prev_motion_states: Dict[int, str] = {}
        self._action_phases: Dict[int, str] = {}
        self._action_stable_frames: Dict[int, int] = {}
    
    def process(self, tracks: List['Track'], 
                states: Dict[int, 'ObjectState'],
                predictions: Dict[int, 'TrajectoryPrediction']) -> Tuple[List[Event], SceneContext]:
        self._frame_id += 1
        current_time = time.time()
        events = []
        current_track_ids = {t.track_id for t in tracks}
        
        for track_id in current_track_ids - self._prev_track_ids:
            track = next((t for t in tracks if t.track_id == track_id), None)
            if track:
                events.append(Event(
                    event_type=EventType.NEW_OBJECT,
                    event_id=f"EVT-{uuid.uuid4().hex[:8]}",
                    timestamp=current_time,
                    frame_id=self._frame_id,
                    data={'object': {'id': track_id, 'class': track.class_name, 'bbox': list(track.bbox), 'confidence': float(track.confidence)}}
                ))
        
        for track_id in self._prev_track_ids - current_track_ids:
            self._lost_tracks[track_id] = self._lost_tracks.get(track_id, 0) + 1
            if self._lost_tracks[track_id] == 1:
                events.append(Event(
                    event_type=EventType.OBJECT_LEFT,
                    event_id=f"EVT-{uuid.uuid4().hex[:8]}",
                    timestamp=current_time,
                    frame_id=self._frame_id,
                    data={'object_id': track_id}
                ))
        
        for track_id, state in states.items():
            if track_id in self._prev_motion_states:
                prev_motion = self._prev_motion_states[track_id]
                current_phase = self._action_phases.get(track_id, 'stable')
                
                if prev_motion != state.motion_state:
                    if state.motion_state == 'moving' and current_phase == 'stable':
                        self._action_phases[track_id] = 'in_progress'
                        self._action_stable_frames[track_id] = 0
                    elif state.motion_state == 'stationary' and current_phase == 'in_progress':
                        self._action_stable_frames[track_id] = self._action_stable_frames.get(track_id, 0) + 1
                        if self._action_stable_frames[track_id] >= 3:
                            events.append(Event(
                                event_type=EventType.ACTION_COMPLETE,
                                event_id=f"EVT-{uuid.uuid4().hex[:8]}",
                                timestamp=current_time,
                                frame_id=self._frame_id,
                                data={'object_id': track_id, 'action': 'completed movement', 'from_state': prev_motion, 'to_state': state.motion_state}
                            ))
                            self._action_phases[track_id] = 'stable'
                            self._action_stable_frames[track_id] = 0
                    elif state.motion_state == 'stationary':
                        self._action_stable_frames[track_id] = self._action_stable_frames.get(track_id, 0) + 1
                else:
                    if current_phase == 'in_progress':
                        self._action_stable_frames[track_id] = self._action_stable_frames.get(track_id, 0) + 1
            
            self._prev_motion_states[track_id] = state.motion_state
        
        if self._frame_id % self.scene_update_interval == 0:
            scene_context = self._build_scene_context(tracks, states, predictions)
            events.append(Event(
                event_type=EventType.CAPTION_UPDATE,
                event_id=f"EVT-{uuid.uuid4().hex[:8]}",
                timestamp=current_time,
                frame_id=self._frame_id,
                data={'scene': scene_context.to_dict()}
            ))
        
        self._prev_track_ids = current_track_ids
        
        for track_id in self._prev_track_ids - current_track_ids:
            if track_id in self._lost_tracks:
                del self._lost_tracks[track_id]
        
        self._event_history.extend(events)
        if len(self._event_history) > self._max_event_history:
            self._event_history = self._event_history[-self._max_event_history:]
        
        return events, self._build_scene_context(tracks, states, predictions)
    
    def _build_scene_context(self, tracks: List['Track'],
                            states: Dict[int, 'ObjectState'],
                            predictions: Dict[int, 'TrajectoryPrediction']) -> SceneContext:
        active_objects = []
        for track in tracks[:self.max_tracked]:
            state = states.get(track.track_id)
            pred = predictions.get(track.track_id)
            if state is None:
                continue
            active_objects.append({
                'id': track.track_id,
                'class': track.class_name,
                'bbox': list(track.bbox),
                'center': list(track.center),
                'motion_state': state.motion_state,
                'action_phase': self._action_phases.get(track.track_id, 'stable'),
                'speed_ms': state.speed_ms,
                'intent': pred.intent if pred else 'unknown'
            })
        return SceneContext(active_objects=active_objects, summary=self._generate_summary(active_objects))
    
    def _generate_summary(self, objects: List[Dict]) -> str:
        if not objects:
            return "Empty scene"
        classes = {}
        for obj in objects:
            cls = obj['class']
            classes[cls] = classes.get(cls, 0) + 1
        parts = [f"{count} {cls}{'s' if count > 1 else ''}" for cls, count in classes.items()]
        return f"Scene contains {', '.join(parts)}"
    
    def get_recent_events(self, n: int = 10) -> List[Event]:
        return self._event_history[-n:]
    
    def reset(self):
        self._frame_id = 0
        self._prev_track_ids.clear()
        self._lost_tracks.clear()
        self._event_history.clear()
        self._prev_motion_states.clear()
