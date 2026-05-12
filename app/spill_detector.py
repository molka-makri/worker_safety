import os
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
from app.hf_model_store import ensure_model_file

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SPILL_MODEL_PATH = ensure_model_file('spill_detection_model.pt')

CONF_THRESHOLD = 0.20
MIN_SPILL_AREA_RATIO = 0.0002
TEMPORAL_HOLD_FRAMES = 6
BBOX_SMOOTHING_ALPHA = 0.65
CONFIDENCE_DECAY = 0.88
SPILL_LABEL_HINTS = {
    'spill', 'spillage', 'chemical', 'liquid', 'leak', 'oil', 'hazard',
    'contamination', 'puddle', 'fluid', 'water', 'deversement',
}

YOLO_AVAILABLE = False
yolo_model = None

try:
    from ultralytics import YOLO

    if os.path.exists(SPILL_MODEL_PATH):
        yolo_model = YOLO(SPILL_MODEL_PATH)
        YOLO_AVAILABLE = True
        print(f"[SpillDetector] OK: Model loaded: {SPILL_MODEL_PATH}")
        print("[SpillDetector] Model classes:")
        for cls_id, cls_name in yolo_model.names.items():
            print(f"   class {cls_id} -> '{cls_name}'")
    else:
        print(f"[SpillDetector] WARNING: Model not found: {SPILL_MODEL_PATH}")
except ImportError as exc:
    print(f"[SpillDetector] WARNING: ultralytics not installed: {exc}")


def _is_spill_label(label: str) -> bool:
    normalized = label.lower().strip()
    return any(hint in normalized for hint in SPILL_LABEL_HINTS)


def _simplify_polygon(points: np.ndarray) -> List[List[int]]:
    if points is None or len(points) < 3:
        return []
    contour = points.astype(np.int32).reshape((-1, 1, 2))
    epsilon = max(1.0, 0.006 * cv2.arcLength(contour, True))
    simplified = cv2.approxPolyDP(contour, epsilon, True).reshape((-1, 2))
    return [[int(x), int(y)] for x, y in simplified[:80]]


def _smooth_bbox(previous_bbox, current_bbox):
    if not previous_bbox or not current_bbox:
        return current_bbox
    return [
        int(round(previous * BBOX_SMOOTHING_ALPHA + current * (1 - BBOX_SMOOTHING_ALPHA)))
        for previous, current in zip(previous_bbox, current_bbox)
    ]


class SpillDetector:
    def __init__(self):
        self.model = yolo_model
        self.available = YOLO_AVAILABLE
        self._spill_streak = 0
        self._states = {}

    def detect_spill(self, frame: np.ndarray, camera: str = 'default') -> Tuple[bool, float, Dict[str, Any]]:
        if not self.available or self.model is None:
            return self._fallback_detection()
        try:
            results = self.model(frame, conf=CONF_THRESHOLD, verbose=False)
            detected, confidence, details = self._parse_results(results, frame.shape)
            return self._apply_temporal_smoothing(camera, detected, confidence, details)
        except Exception as exc:
            print(f"[SpillDetector] YOLO error: {exc}")
            return self._fallback_detection()

    def _apply_temporal_smoothing(
        self,
        camera: str,
        detected: bool,
        confidence: float,
        details: Dict[str, Any],
    ) -> Tuple[bool, float, Dict[str, Any]]:
        state = self._states.get(camera, {'misses': 0, 'confidence': 0.0, 'details': None})

        if detected:
            previous_details = state.get('details') or {}
            smoothed_bbox = _smooth_bbox(previous_details.get('bbox'), details.get('bbox'))
            if smoothed_bbox:
                details['bbox'] = smoothed_bbox
            details['temporal_hold'] = False
            details['missed_frames'] = 0
            details['processing_method'] = 'yolo_segmentation_temporal'
            self._states[camera] = {
                'misses': 0,
                'confidence': confidence,
                'details': details,
            }
            return True, confidence, details

        last_details = state.get('details')
        misses = int(state.get('misses', 0)) + 1
        if last_details and misses <= TEMPORAL_HOLD_FRAMES:
            held_details = dict(last_details)
            held_confidence = max(CONF_THRESHOLD, float(state.get('confidence', 0.0)) * (CONFIDENCE_DECAY ** misses))
            held_details.update({
                'processing_method': 'temporal_hold',
                'raw_spill_found': False,
                'temporal_hold': True,
                'missed_frames': misses,
                'detections': details.get('detections', []),
            })
            self._states[camera] = {
                'misses': misses,
                'confidence': state.get('confidence', held_confidence),
                'details': last_details,
            }
            return True, min(held_confidence, 1.0), held_details

        self._states[camera] = {'misses': 0, 'confidence': 0.0, 'details': None}
        details['temporal_hold'] = False
        details['missed_frames'] = misses
        return False, 0.0, details

    def _parse_results(self, results, frame_shape) -> Tuple[bool, float, Dict[str, Any]]:
        frame_h, frame_w = frame_shape[:2]
        frame_area = max(1, frame_h * frame_w)
        detections = []
        polygons = []
        best_bbox = None
        best_confidence = 0.0
        spill_area = 0.0

        for result in results:
            boxes = result.boxes
            masks = result.masks
            if boxes is None or len(boxes) == 0:
                continue

            mask_polygons = masks.xy if masks is not None and masks.xy is not None else []

            for index, box in enumerate(boxes):
                cls_id = int(box.cls[0])
                confidence = float(box.conf[0])
                label = result.names.get(cls_id, str(cls_id))
                x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                bbox_area = max(0, x2 - x1) * max(0, y2 - y1)
                polygon = _simplify_polygon(mask_polygons[index]) if index < len(mask_polygons) else []
                area = float(cv2.contourArea(np.array(polygon, dtype=np.float32))) if polygon else float(bbox_area)
                is_spill = confidence >= CONF_THRESHOLD

                detections.append({
                    'class_id': cls_id,
                    'label': label,
                    'confidence': round(confidence, 3),
                    'bbox': [x1, y1, x2, y2],
                    'area_ratio': round(area / frame_area, 5),
                })

                if is_spill:
                    spill_area += area
                    if polygon:
                        polygons.append(polygon)
                    if confidence > best_confidence:
                        best_confidence = confidence
                        best_bbox = [x1, y1, x2, y2]

        area_ratio = spill_area / frame_area
        raw_spill_detected = best_confidence >= CONF_THRESHOLD and area_ratio >= MIN_SPILL_AREA_RATIO

        if raw_spill_detected:
            self._spill_streak += 1
        else:
            self._spill_streak = 0

        spill_detected = raw_spill_detected
        details = {
            'model': 'YOLO_spill_detection_model.pt',
            'processing_method': 'yolo_segmentation',
            'model_available': True,
            'detections': detections,
            'bbox': best_bbox,
            'polygons': polygons[:8],
            'spill_area_ratio': round(area_ratio, 5),
            'raw_spill_found': raw_spill_detected,
            'spill_streak': self._spill_streak,
            'accepted_labels': sorted(SPILL_LABEL_HINTS),
        }
        return spill_detected, min(best_confidence, 1.0) if spill_detected else 0.0, details

    def _fallback_detection(self) -> Tuple[bool, float, Dict[str, Any]]:
        return False, 0.0, {
            'model': 'YOLO_spill_detection_model.pt',
            'processing_method': 'model_unavailable',
            'model_available': False,
            'bbox': None,
            'polygons': [],
            'note': f'Modele YOLO introuvable: {SPILL_MODEL_PATH}',
        }


spill_detector = SpillDetector()


def detect_spill_in_frame(frame: np.ndarray, camera: str = 'default') -> Tuple[bool, float, Dict[str, Any]]:
    return spill_detector.detect_spill(frame, camera=camera)
