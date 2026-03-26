import time
import numpy as np
from typing import List, Tuple, Dict
from collections import OrderedDict
from dataclasses import dataclass


@dataclass
class Track:
    track_id: int
    class_id: int
    class_name: str
    bbox: Tuple[int, int, int, int]
    confidence: float
    age: int = 0
    hits: int = 0
    time_since_update: int = 0
    state: str = 'tentative'
    last_update: float = 0
    
    @property
    def center(self) -> Tuple[float, float]:
        x, y, w, h = self.bbox
        return (x + w / 2, y + h / 2)
    
    def to_dict(self) -> Dict:
        return {
            'id': int(self.track_id), 'class': self.class_name, 'class_id': int(self.class_id),
            'bbox': [int(x) for x in self.bbox], 'center': [float(x) for x in self.center],
            'confidence': float(self.confidence), 'age': int(self.age), 'state': self.state
        }


class KalmanBoxTracker:
    count = 0
    
    def __init__(self, bbox):
        from filterpy.kalman import KalmanFilter
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self._init_kalman(bbox)
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.last_update = time.time()
    
    def _init_kalman(self, bbox):
        x, y, w, h = bbox
        self.kf.x = np.array([[x + w/2], [y + h/2], [w], [h], [0], [0], [0]])
        self.kf.F = np.array([[1,0,0,0,1,0,0], [0,1,0,0,0,1,0], [0,0,1,0,0,0,1], [0,0,0,1,0,0,0], [0,0,0,0,1,0,0], [0,0,0,0,0,1,0], [0,0,0,0,0,0,1]])
        self.kf.H = np.array([[1,0,0,0,0,0,0], [0,1,0,0,0,0,0], [0,0,1,0,0,0,0], [0,0,0,1,0,0,0]])
        self.kf.P[4:, 4:] *= 1000.
        self.kf.P *= 10.
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01
        self.kf.R[2:, 2:] *= 10.
    
    def update(self, bbox):
        self.kf.update(self._measurement(bbox))
        self.time_since_update = 0
        self.last_update = time.time()
    
    def predict(self):
        self.kf.predict()
        self.time_since_update += 1
        return self.get_state()
    
    def get_state(self):
        return self.kf.x[:4].flatten()
    
    @staticmethod
    def _measurement(bbox):
        x, y, w, h = bbox
        return np.array([[x + w/2], [y + h/2], [w], [h]])


class SortTracker:
    def __init__(self, max_age: int = 30, min_hits: int = 3, iou_threshold: float = 0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.tracks: OrderedDict[int, KalmanBoxTracker] = OrderedDict()
        self.frame_count = 0
    
    def update(self, detections: List['Detection']) -> List[Track]:
        self.frame_count += 1
        if len(detections) == 0:
            return self._get_active()
        
        det_boxes = np.array([d.bbox for d in detections])
        det_classes = np.array([d.class_id for d in detections])
        det_confs = np.array([d.confidence for d in detections])
        det_names = [d.class_name for d in detections]
        
        predicted = np.array([self._pred_to_box(self.tracks[k].predict()) for k in list(self.tracks.keys())]) if self.tracks else np.empty((0, 4))
        
        iou_matrix = self._compute_iou(det_boxes, predicted)
        matched, unmatched_dets, unmatched_tracks = self._associate(iou_matrix)
        
        for di, ti in matched:
            tid = list(self.tracks.keys())[ti]
            self.tracks[tid].update(det_boxes[di])
            self.tracks[tid].hits += 1
            self.tracks[tid].confidence = det_confs[di]
            self.tracks[tid].age = self.frame_count
        
        for di in unmatched_dets:
            self._create_track(det_boxes[di], det_classes[di], det_names[di], det_confs[di])
        
        for ti in unmatched_tracks:
            self.tracks[list(self.tracks.keys())[ti]].time_since_update += 1
        
        self._prune()
        return self._get_active()
    
    def _pred_to_box(self, pred: np.ndarray) -> np.ndarray:
        x, y, w, h = pred
        return np.array([x - w/2, y - h/2, w, h])
    
    def _compute_iou(self, boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
        iou = np.zeros((len(boxes1), len(boxes2)))
        for i, b1 in enumerate(boxes1):
            for j, b2 in enumerate(boxes2):
                iou[i, j] = self._iou(b1, b2)
        return iou
    
    @staticmethod
    def _iou(box1: np.ndarray, box2: np.ndarray) -> float:
        x1_1, y1_1, w1, h1 = box1
        x1_2, y1_2, w2, h2 = box2
        x2_1, y2_1 = x1_1 + w1, y1_1 + h1
        x2_2, y2_2 = x1_2 + w2, y1_2 + h2
        xi1, yi1 = max(x1_1, x1_2), max(y1_1, y1_2)
        xi2, yi2 = min(x2_1, x2_2), min(y2_1, y2_2)
        inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        return inter / (w1 * h1 + w2 * h2 - inter + 1e-6)
    
    def _associate(self, iou_matrix: np.ndarray):
        from scipy.optimize import linear_sum_assignment
        if iou_matrix.size == 0:
            return np.array([]), np.arange(iou_matrix.shape[0] if len(iou_matrix.shape) > 0 else 0), np.array([], dtype=int)
        rows, cols = linear_sum_assignment(-iou_matrix)
        matched = [[r, c] for r, c in zip(rows, cols) if iou_matrix[r, c] >= self.iou_threshold]
        unmatched = [i for i in range(len(iou_matrix)) if i not in rows]
        unmatched_t = [i for i in range(len(iou_matrix[0])) if i not in cols]
        return np.array(matched), np.array(unmatched), np.array(unmatched_t)
    
    def _create_track(self, bbox, class_id, class_name, conf):
        self.tracks[KalmanBoxTracker.count] = KalmanBoxTracker(bbox.astype(int))
        tid = KalmanBoxTracker.count
        self.tracks[tid].class_id = class_id
        self.tracks[tid].class_name = class_name
        self.tracks[tid].confidence = conf
        self.tracks[tid].hits = 1
        self.tracks[tid].age = self.frame_count
        self.tracks[tid].state = 'tentative'
    
    def _prune(self):
        deleted = [tid for tid, t in self.tracks.items() if t.time_since_update > self.max_age]
        for tid in deleted:
            del self.tracks[tid]
    
    def _get_active(self) -> List[Track]:
        return [Track(tid, t.class_id, t.class_name, tuple(int(x) for x in self._pred_to_box(t.get_state())),
                t.confidence, t.age, t.hits, t.time_since_update, t.state, t.last_update)
               for tid, t in self.tracks.items()]
    
    def reset(self):
        self.tracks.clear()
        self.frame_count = 0
        KalmanBoxTracker.count = 0
