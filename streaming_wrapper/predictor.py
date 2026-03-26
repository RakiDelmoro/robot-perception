import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass


@dataclass
class TrajectoryPrediction:
    track_id: int
    trajectory: List[Tuple[float, float]]
    horizon_s: float
    intent: str
    confidence: float
    
    def to_dict(self) -> Dict:
        return {
            'trajectory': [[float(x), float(y)] for x, y in self.trajectory],
            'horizon_s': self.horizon_s,
            'intent': self.intent,
            'confidence': float(self.confidence)
        }


class TrajectoryPredictor:
    def __init__(self, horizon_s: float = 2.0, dt_s: float = 0.2, position_scale: float = 100.0):
        self.horizon_s = horizon_s
        self.dt_s = dt_s
        self.position_scale = position_scale
        self._n_points = int(horizon_s / dt_s)
    
    def predict(self, states: Dict[int, 'ObjectState']) -> Dict[int, TrajectoryPrediction]:
        predictions = {}
        for track_id, state in states.items():
            predictions[track_id] = TrajectoryPrediction(
                track_id=track_id,
                trajectory=self._constant_velocity_predict(state),
                horizon_s=self.horizon_s,
                intent=self._classify_intent(state),
                confidence=self._compute_confidence(state)
            )
        return predictions
    
    def _constant_velocity_predict(self, state: 'ObjectState') -> List[Tuple[float, float]]:
        x0, y0 = state.position
        vx, vy = state.velocity
        return [(x0 + vx * i * self.dt_s, y0 + vy * i * self.dt_s) for i in range(1, self._n_points + 1)]
    
    def _classify_intent(self, state: 'ObjectState') -> str:
        speed = state.speed
        if speed < 5.0:
            return 'stationary'
        heading = state.heading
        lateral = abs(np.sin(heading) * speed)
        forward = abs(np.cos(heading) * speed)
        if lateral > forward * 0.7:
            return 'crossing'
        elif forward > lateral * 2:
            return 'approaching' if speed > 50 else 'departing'
        return 'moving'
    
    def _compute_confidence(self, state: 'ObjectState') -> float:
        base_confidence = state.confidence
        angular_speed = np.sqrt(state.acceleration[0]**2 + state.acceleration[1]**2) / max(state.speed, 1.0)
        turning_penalty = min(1.0, angular_speed / 0.5)
        return max(0.1, min(1.0, base_confidence * (1.0 - 0.2 * turning_penalty)))
