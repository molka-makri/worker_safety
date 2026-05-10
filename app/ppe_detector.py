import os
import cv2
from typing import Any, Dict, Tuple
from app.hf_model_store import ensure_model_file

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PPE_MODEL_PATH = ensure_model_file('ppe.pt')

CONF_THRESHOLD = 0.40

YOLO_AVAILABLE = False
yolo_model = None

try:
    from ultralytics import YOLO
    if os.path.exists(PPE_MODEL_PATH):
        yolo_model = YOLO(PPE_MODEL_PATH)
        YOLO_AVAILABLE = True
        print(f"[PPEDetector] OK: Model loaded: {PPE_MODEL_PATH}")
        print("[PPEDetector] Model classes:")
        for cls_id, cls_name in yolo_model.names.items():
            print(f"   class {cls_id} -> '{cls_name}'")
    else:
        print(f"[PPEDetector] WARNING: Model not found: {PPE_MODEL_PATH}")
except ImportError as exc:
    print(f"[PPEDetector] WARNING: ultralytics not installed: {exc}")


class PPEDetector:
    def __init__(self):
        self.model = yolo_model
        self.available = YOLO_AVAILABLE

    def detect_ppe(self, frame: cv2.typing.MatLike, camera: str = 'default') -> Tuple[bool, float, Dict[str, Any]]:
        if not self.available or self.model is None:
            return False, 0.0, {
                'model': 'YOLO_ppe.pt',
                'model_available': False,
                'note': f'Modele YOLO introuvable: {PPE_MODEL_PATH}'
            }

        try:
            results = self.model(frame, conf=CONF_THRESHOLD, verbose=False)
            return self._parse_results(results, frame.shape)
        except Exception as exc:
            print(f"[PPEDetector] YOLO error: {exc}")
            return False, 0.0, {'model_available': False, 'error': str(exc)}

    def _parse_results(self, results, frame_shape) -> Tuple[bool, float, Dict[str, Any]]:
        detections = []
        has_human = False
        has_helmet = False
        has_vest = False
        has_boots = False  # ADDED BOOTS
        best_confidence = 0.0

        for result in results:
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                continue
                
            for box in boxes:
                cls_id = int(box.cls[0])
                confidence = float(box.conf[0])
                label = result.names.get(cls_id, str(cls_id))
                x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                
                detections.append({
                    'class_id': cls_id,
                    'label': label,
                    'confidence': round(confidence, 3),
                    'bbox': [x1, y1, x2, y2],
                })
                
                # Track what was detected to check for violations
                lbl = label.lower().strip()
                if lbl in ['human', 'person', 'worker']: has_human = True
                if lbl in ['helmet', 'hardhat']: has_helmet = True
                if lbl in ['vest', 'safety-vest']: has_vest = True
                if lbl in ['boots', 'shoes', 'safety-boots', 'boot']: has_boots = True  # ADDED BOOTS
                
                if confidence > best_confidence:
                    best_confidence = confidence

        # A VIOLATION occurs if a human is found but missing helmet, vest, OR boots
        violation_detected = has_human and (not has_helmet or not has_vest or not has_boots)  # ADDED BOOTS
        
        details = {
            'model': 'YOLO_ppe.pt',
            'processing_method': 'yolo_detection',
            'model_available': True,
            'detections': detections,
            'ppe_violation': violation_detected,
            'has_human': has_human,
            'has_helmet': has_helmet,
            'has_vest': has_vest,
            'has_boots': has_boots,  # ADDED BOOTS
        }

        return violation_detected, min(best_confidence, 1.0), details

ppe_detector = PPEDetector()

def detect_ppe_in_frame(frame: cv2.typing.MatLike, camera: str = 'default') -> Tuple[bool, float, Dict[str, Any]]:
    return ppe_detector.detect_ppe(frame, camera=camera)
