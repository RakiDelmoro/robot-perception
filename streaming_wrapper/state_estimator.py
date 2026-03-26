import time
import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass
from collections import deque


@dataclass
class ObjectState:
    track_id: int
    position: Tuple[float, float]
    velocity: Tuple[float, float]
    acceleration: Tuple[float, float]
    speed: float
    speed_ms: float
    motion_state: str
    heading: float
    confidence: float
    timestamp: float
    
    def to_dict(self) -> Dict:
        return {
            'position': {'x': float(self.position[0]), 'y': float(self.position[1])},
            'velocity': {'vx': float(self.velocity[0]), 'vy': float(self.velocity[1])},
            'acceleration': {'ax': float(self.acceleration[0]), 'ay': float(self.acceleration[1])},
            'speed': float(self.speed),
            'speed_ms': float(self.speed_ms),
            'motion_state': self.motion_state,
            'heading_rad': float(self.heading)
        }


class StateEstimator:
    def __init__(self, position_scale: float = 100.0, smoothing: float = 0.3,
                 vel_threshold: float = 5.0, accel_threshold: float = 20.0):
        self.position_scale = position_scale
        self.smoothing = smoothing
        self.vel_threshold = vel_threshold
        self.accel_threshold = accel_threshold
        self._states: Dict[int, Dict] = {}
        self._pos_hist: Dict[int, deque] = {}
        self._time_hist: Dict[int, deque] = {}
        self._max_hist = 30
    
    def update(self, tracks: List['Track']) -> Dict[int, ObjectState]:
        current_time = time.time()
        current_ids = {t.track_id for t in tracks}
        
        for track in tracks:
            tid = track.track_id
            pos = track.center
            
            if tid not in self._pos_hist:
                self._pos_hist[tid] = deque(maxlen=self._max_hist)
                self._time_hist[tid] = deque(maxlen=self._max_hist)
                self._states[tid] = {'velocity': (0., 0.), 'acceleration': (0., 0.), 'last_vel': (0., 0.)}
            
            self._pos_hist[tid].append(pos)
            self._time_hist[tid].append(current_time)
            
            if len(self._pos_hist[tid]) >= 2:
                self._states[tid] = self._estimate(tid)
        
        for tid in list(self._states.keys()):
            if tid not in current_ids:
                del self._pos_hist[tid]
                del self._time_hist[tid]
                del self._states[tid]
        
        return self._build_states(tracks, current_time)
    
    def _estimate(self, tid: int) -> Dict:
        pos = list(self._pos_hist[tid])
        t = list(self._time_hist[tid])
        dt = max(t[-1] - t[-2], 0.001)
        
        raw_vx = (pos[-1][0] - pos[-2][0]) / dt
        raw_vy = (pos[-1][1] - pos[-2][1]) / dt
        
        prev = self._states.get(tid, {})
        prev_v = prev.get('velocity', (raw_vx, raw_vy))
        last_v = prev.get('last_vel', (raw_vx, raw_vy))
        
        smooth_vx = self.smoothing * raw_vx + (1 - self.smoothing) * prev_v[0]
        smooth_vy = self.smoothing * raw_vy + (1 - self.smoothing) * prev_v[1]
        
        raw_ax = (raw_vx - last_v[0]) / dt
        raw_ay = (raw_vy - last_v[1]) / dt
        prev_a = prev.get('acceleration', (0., 0.))
        smooth_ax = self.smoothing * raw_ax + (1 - self.smoothing) * prev_a[0]
        smooth_ay = self.smoothing * raw_ay + (1 - self.smoothing) * prev_a[1]
        
        return {'velocity': (smooth_vx, smooth_vy), 'acceleration': (smooth_ax, smooth_ay), 'last_vel': (raw_vx, raw_vy)}
    
    def _build_states(self, tracks: List['Track'], current_time: float) -> Dict[int, ObjectState]:
        states = {}
        for track in tracks:
            tid = track.track_id
            if tid not in self._states:
                continue
            s = self._states[tid]
            vel, acc = s['velocity'], s['acceleration']
            speed = np.sqrt(vel[0]**2 + vel[1]**2)
            speed_ms = speed / self.position_scale
            heading = np.arctan2(vel[1], vel[0]) if speed > 0.1 else 0.0
            accel_mag = np.sqrt(acc[0]**2 + acc[1]**2)
            
            if speed < self.vel_threshold:
                motion_state = 'stationary'
            elif accel_mag > self.accel_threshold:
                motion_state = 'accelerating' if np.dot(vel, acc) > 0 else 'decelerating'
            else:
                motion_state = 'moving'
            
            confidence = min(1.0, len(self._pos_hist.get(tid, [])) / 5.0)
            states[tid] = ObjectState(
                track_id=tid, position=track.center, velocity=vel, acceleration=acc,
                speed=speed, speed_ms=speed_ms, motion_state=motion_state,
                heading=heading, confidence=confidence, timestamp=current_time
            )
        return states
    
    def reset(self):
        self._states.clear()
        self._pos_hist.clear()
        self._time_hist.clear()
