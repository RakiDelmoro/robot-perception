import time
import numpy as np
from typing import List, Optional, Tuple, Dict


class Detection:
    def __init__(self, bbox: Tuple[int, int, int, int], class_id: int, class_name: str, confidence: float):
        self.bbox = bbox
        self.class_id = class_id
        self.class_name = class_name
        self.confidence = confidence
    
    @property
    def center(self) -> Tuple[float, float]:
        x, y, w, h = self.bbox
        return (x + w / 2, y + h / 2)
    
    def to_dict(self) -> Dict:
        return {
            'bbox': [int(x) for x in self.bbox],
            'class_id': int(self.class_id),
            'class_name': self.class_name,
            'confidence': float(self.confidence),
            'center': [float(x) for x in self.center]
        }


class ObjectDetector:
    ROBOT_CLASSES = [0, 1, 2, 3, 5, 7, 9, 10, 11, 15, 16, 56, 57, 59, 60, 61, 62, 63, 64, 65, 66, 67, 72, 73, 74, 75]
    
    def __init__(self, model_name: str = 'yolov8n.pt', confidence: float = 0.25, device: str = 'cuda', classes: Optional[List[int]] = None):
        self.model_name = model_name
        self.confidence = confidence
        self.device = device
        self.classes = classes
        self.model = None
        self._inference_time = 0
    
    def load(self) -> bool:
        try:
            from ultralytics import YOLO
            import torch
            print(f"[DETECTOR] Loading YOLO: {self.model_name}")
            self.model = YOLO(self.model_name)
            if self.device == 'cuda' and torch.cuda.is_available():
                self.model.to('cuda')
            print(f"[DETECTOR] Using {'CUDA' if self.device == 'cuda' else 'CPU'}")
            return True
        except ImportError:
            print("[DETECTOR] ERROR: ultralytics not installed")
            return False
        except Exception as e:
            print(f"[DETECTOR] ERROR: {e}")
            return False
    
    def detect(self, frame: np.ndarray) -> List[Detection]:
        if self.model is None:
            return []
        start = time.time()
        try:
            results = self.model(frame, verbose=False, device=self.device)[0]
            detections = []
            if results.boxes is not None:
                for box in results.boxes:
                    conf = float(box.conf[0])
                    if conf < self.confidence:
                        continue
                    class_id = int(box.cls[0])
                    if self.classes and class_id not in self.classes:
                        continue
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    detections.append(Detection(bbox=(int(x1), int(y1), int(x2-x1), int(y2-y1)), class_id=class_id, class_name=self.model.names[class_id], confidence=conf))
            self._inference_time = time.time() - start
            return detections
        except Exception as e:
            print(f"[DETECTOR] Error: {e}")
            return []
    
    @property
    def latency_ms(self) -> float:
        return self._inference_time * 1000
    
    @property
    def is_ready(self) -> bool:
        return self.model is not None


def create_robot_detector(device: str = 'cuda') -> ObjectDetector:
    return ObjectDetector(model_name='yolov8n.pt', confidence=0.5, device=device, classes=ObjectDetector.ROBOT_CLASSES)
