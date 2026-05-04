import glob
import os
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.abspath(os.path.join(BASE_DIR, '..', 'models'))
MANHOLE_MODEL_PATH = os.path.join(MODELS_DIR, 'manhole_seg.pt')

CONF_THRESHOLD = 0.05
MIN_AREA_RATIO = 0.002
MAX_AREA_RATIO = 0.40
TEMPORAL_HOLD_FRAMES = 10
CONFIDENCE_DECAY = 0.93
BBOX_SMOOTHING_ALPHA = 0.78
POLYGON_SMOOTHING_ALPHA = 0.82
POLYGON_POINTS = 64

YOLO_AVAILABLE = False
yolo_model = None

try:
    from ultralytics import YOLO

    if os.path.exists(MANHOLE_MODEL_PATH):
        yolo_model = YOLO(MANHOLE_MODEL_PATH)
        YOLO_AVAILABLE = True
        print(f"[ManholeDetector] OK: Model loaded: {MANHOLE_MODEL_PATH}")
        print("[ManholeDetector] Model classes:")
        for cls_id, cls_name in yolo_model.names.items():
            print(f"   class {cls_id} -> '{cls_name}'")
    else:
        print(f"[ManholeDetector] WARNING: Model not found: {MANHOLE_MODEL_PATH}")
except ImportError as exc:
    print(f"[ManholeDetector] WARNING: ultralytics not installed: {exc}")


def _simplify_polygon(points: np.ndarray) -> List[List[int]]:
    if points is None or len(points) < 3:
        return []
    contour = points.astype(np.int32).reshape((-1, 1, 2))
    epsilon = max(1.0, 0.006 * cv2.arcLength(contour, True))
    simplified = cv2.approxPolyDP(contour, epsilon, True).reshape((-1, 2))
    return [[int(x), int(y)] for x, y in simplified[:80]]


def _resample_polygon(polygon, count: int = POLYGON_POINTS) -> List[List[float]]:
    if not polygon or len(polygon) < 3:
        return []
    points = np.asarray(polygon, dtype=np.float32)
    closed = np.vstack([points, points[0]])
    segments = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    perimeter = float(segments.sum())
    if perimeter <= 0:
        return [[float(x), float(y)] for x, y in points[:count]]
    targets = np.linspace(0, perimeter, count, endpoint=False)
    cumulative = np.concatenate([[0.0], np.cumsum(segments)])
    sampled = []
    segment_index = 0
    for target in targets:
        while segment_index < len(segments) - 1 and cumulative[segment_index + 1] < target:
            segment_index += 1
        segment_start = closed[segment_index]
        segment_end = closed[segment_index + 1]
        segment_length = max(segments[segment_index], 1e-6)
        ratio = (target - cumulative[segment_index]) / segment_length
        point = segment_start + (segment_end - segment_start) * ratio
        sampled.append([float(point[0]), float(point[1])])
    return sampled


def _smooth_polygon(previous_polygon, current_polygon):
    current = _resample_polygon(current_polygon)
    if not previous_polygon:
        return [[int(round(x)), int(round(y))] for x, y in current]
    previous = _resample_polygon(previous_polygon)
    if len(previous) != len(current) or not current:
        return [[int(round(x)), int(round(y))] for x, y in current]

    previous_arr = np.asarray(previous, dtype=np.float32)
    current_arr = np.asarray(current, dtype=np.float32)
    offsets = np.linalg.norm(np.roll(previous_arr[None, :, :], np.arange(len(current)), axis=1) - current_arr, axis=2).sum(axis=1)
    best_offset = int(np.argmin(offsets))
    aligned_previous = np.roll(previous_arr, best_offset, axis=0)
    smoothed = aligned_previous * POLYGON_SMOOTHING_ALPHA + current_arr * (1 - POLYGON_SMOOTHING_ALPHA)
    return [[int(round(x)), int(round(y))] for x, y in smoothed]


def _smooth_bbox(previous_bbox, current_bbox):
    if not previous_bbox or not current_bbox:
        return current_bbox
    return [
        int(round(previous * BBOX_SMOOTHING_ALPHA + current * (1 - BBOX_SMOOTHING_ALPHA)))
        for previous, current in zip(previous_bbox, current_bbox)
    ]


def _candidate_metrics(polygon, bbox, frame_shape) -> Dict[str, float]:
    frame_h, frame_w = frame_shape[:2]
    frame_area = max(1.0, frame_h * frame_w)
    x1, y1, x2, y2 = bbox
    width = max(1.0, x2 - x1)
    height = max(1.0, y2 - y1)
    area = float(cv2.contourArea(np.asarray(polygon, dtype=np.float32))) if polygon else width * height
    perimeter = float(cv2.arcLength(np.asarray(polygon, dtype=np.float32).reshape((-1, 1, 2)), True)) if polygon else (2.0 * (width + height))
    circularity = float(4 * np.pi * area / (perimeter * perimeter)) if perimeter > 0 else 0.0
    aspect_ratio = width / height
    center_y_ratio = ((y1 + y2) / 2.0) / max(1.0, frame_h)
    return {
        'area_ratio': area / frame_area,
        'circularity': circularity,
        'aspect_ratio': aspect_ratio,
        'center_y_ratio': center_y_ratio,
    }


class DepthEstimator:
    def __init__(self):
        self.available = False
        self.model = None
        self.input_size = 518
        self.model_name = 'heuristic_fallback'
        self._load_depth_anything()

    def _load_depth_anything(self):
        try:
            from depth_anything_v2.dpt import DepthAnythingV2  # type: ignore
        except Exception:
            return

        weight_candidates = (
            glob.glob(os.path.join(MODELS_DIR, 'depth_anything_v2*.pth')) +
            glob.glob(os.path.join(MODELS_DIR, 'depth_anything*.pth'))
        )
        if not weight_candidates:
            return

        try:
            import torch

            checkpoint = weight_candidates[0]
            encoder = 'vitb'
            lower = os.path.basename(checkpoint).lower()
            if 'vitl' in lower:
                encoder = 'vitl'
            elif 'vits' in lower:
                encoder = 'vits'

            model_configs = {
                'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
                'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
                'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
            }
            self.model = DepthAnythingV2(**model_configs[encoder])
            state_dict = torch.load(checkpoint, map_location='cpu')
            self.model.load_state_dict(state_dict)
            self.model.eval()
            self.available = True
            self.model_name = f'depth_anything_v2_{encoder}'
            print(f"[DepthEstimator] OK: Loaded weights: {checkpoint}")
        except Exception as exc:
            print(f"[DepthEstimator] WARNING: Depth Anything V2 unavailable: {exc}")
            self.model = None
            self.available = False

    def estimate(self, frame: np.ndarray, polygon=None, bbox=None, state='unknown') -> Dict[str, Any]:
        if self.available and self.model is not None:
            try:
                depth_map = self.model.infer_image(frame, self.input_size)
                if depth_map is not None:
                    return self._from_depth_map(depth_map, polygon, bbox, state, source=self.model_name)
            except Exception as exc:
                print(f"[DepthEstimator] WARNING: Inference failed: {exc}")

        return self._heuristic_depth(frame, polygon, bbox, state)

    def _from_depth_map(self, depth_map, polygon, bbox, state, source) -> Dict[str, Any]:
        depth = np.asarray(depth_map, dtype=np.float32)
        if depth.size == 0:
            return self._heuristic_depth(np.zeros((1, 1, 3), dtype=np.uint8), polygon, bbox, state)
        normalized = depth - depth.min()
        max_value = float(normalized.max())
        if max_value > 0:
            normalized /= max_value

        mask = np.zeros(depth.shape[:2], dtype=np.uint8)
        if polygon and len(polygon) >= 3:
            cv2.fillPoly(mask, [np.asarray(polygon, dtype=np.int32)], 255)
        elif bbox:
            x1, y1, x2, y2 = [int(v) for v in bbox]
            mask[max(0, y1):max(0, y2), max(0, x1):max(0, x2)] = 255
        else:
            mask[:] = 255

        selected = normalized[mask > 0]
        mean_depth = float(selected.mean()) if selected.size else 0.0
        severity = 'high' if state == 'open' and mean_depth > 0.58 else 'medium' if state == 'open' else 'low'
        return {
            'depth_available': True,
            'depth_source': source,
            'depth_score': round(mean_depth, 4),
            'depth_level': 'deep' if mean_depth > 0.58 else 'moderate' if mean_depth > 0.36 else 'shallow',
            'risk_level': severity,
        }

    def _heuristic_depth(self, frame: np.ndarray, polygon=None, bbox=None, state='unknown') -> Dict[str, Any]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mask = np.zeros(gray.shape[:2], dtype=np.uint8)
        if polygon and len(polygon) >= 3:
            cv2.fillPoly(mask, [np.asarray(polygon, dtype=np.int32)], 255)
        elif bbox:
            x1, y1, x2, y2 = [int(v) for v in bbox]
            mask[max(0, y1):max(0, y2), max(0, x1):max(0, x2)] = 255
        else:
            mask[:] = 255

        masked_pixels = gray[mask > 0]
        darkness = 1.0 - (float(masked_pixels.mean()) / 255.0) if masked_pixels.size else 0.0
        bbox_height_ratio = 0.0
        center_y_ratio = 0.0
        if bbox:
            x1, y1, x2, y2 = bbox
            bbox_height_ratio = max(0.0, (y2 - y1) / max(1.0, frame.shape[0]))
            center_y_ratio = ((y1 + y2) / 2.0) / max(1.0, frame.shape[0])

        depth_score = float(np.clip(0.55 * darkness + 0.25 * center_y_ratio + 0.20 * bbox_height_ratio, 0.0, 1.0))
        if state == 'closed':
            depth_score *= 0.55

        return {
            'depth_available': False,
            'depth_source': self.model_name,
            'depth_score': round(depth_score, 4),
            'depth_level': 'deep' if depth_score > 0.58 else 'moderate' if depth_score > 0.36 else 'shallow',
            'risk_level': 'critical' if state == 'open' and depth_score > 0.62 else 'high' if state == 'open' else 'low',
        }


depth_estimator = DepthEstimator()


class ManholeDetector:
    def __init__(self):
        self.model = yolo_model
        self.available = YOLO_AVAILABLE
        self._states: Dict[str, Dict[str, Any]] = {}

    def detect_manhole(self, frame: np.ndarray, camera: str = 'cam5') -> Tuple[bool, float, Dict[str, Any]]:
        if not self.available or self.model is None:
            return self._fallback_detection()
        try:
            results = self.model(frame, conf=CONF_THRESHOLD, verbose=False)
            detected, confidence, details = self._parse_results(results, frame.shape, frame)
            return self._apply_temporal_smoothing(camera, detected, confidence, details)
        except Exception as exc:
            print(f"[ManholeDetector] YOLO error: {exc}")
            return self._fallback_detection()

    def _parse_results(self, results, frame_shape, frame) -> Tuple[bool, float, Dict[str, Any]]:
        candidates = []
        all_detections = []

        for result in results:
            boxes = result.boxes
            masks = result.masks
            if boxes is None or len(boxes) == 0:
                continue
            mask_polygons = masks.xy if masks is not None and masks.xy is not None else []

            for index, box in enumerate(boxes):
                cls_id = int(box.cls[0])
                confidence = float(box.conf[0])
                label = str(result.names.get(cls_id, str(cls_id)))
                bbox = [int(v) for v in box.xyxy[0].tolist()]
                polygon = _simplify_polygon(mask_polygons[index]) if index < len(mask_polygons) else []
                metrics = _candidate_metrics(polygon, bbox, frame_shape)
                state = 'open' if 'open' in label.lower() else 'closed'
                score = confidence
                score += min(metrics['circularity'], 1.0) * 0.35
                score += max(0.0, 1.0 - abs(metrics['aspect_ratio'] - 1.0)) * 0.20
                if state == 'open':
                    score += 0.08
                if metrics['area_ratio'] < MIN_AREA_RATIO or metrics['area_ratio'] > MAX_AREA_RATIO:
                    score -= 0.40
                if metrics['aspect_ratio'] < 0.35 or metrics['aspect_ratio'] > 2.8:
                    score -= 0.25

                payload = {
                    'class_id': cls_id,
                    'label': label,
                    'state': state,
                    'confidence': round(confidence, 3),
                    'bbox': bbox,
                    'polygon': polygon,
                    'metrics': {key: round(value, 4) for key, value in metrics.items()},
                    'candidate_score': round(score, 4),
                }
                all_detections.append(payload)

                if confidence >= CONF_THRESHOLD and polygon:
                    candidates.append(payload)

        if not candidates:
            return False, 0.0, {
                'model': 'YOLO_manhole_seg.pt',
                'processing_method': 'yolo_segmentation',
                'model_available': True,
                'detections': all_detections,
                'bbox': None,
                'polygons': [],
                'manhole_state': 'unknown',
                'depth_available': False,
                'depth_source': depth_estimator.model_name,
                'depth_score': 0.0,
                'depth_level': 'unknown',
                'risk_level': 'low',
            }

        best = max(candidates, key=lambda item: item['candidate_score'])
        details = {
            'model': 'YOLO_manhole_seg.pt',
            'processing_method': 'yolo_segmentation',
            'model_available': True,
            'detections': all_detections,
            'bbox': best['bbox'],
            'polygons': [best['polygon']],
            'manhole_state': best['state'],
            'shape_metrics': best['metrics'],
            'candidate_score': best['candidate_score'],
            'risk_level': 'high' if best['state'] == 'open' else 'low',
        }
        return True, float(best['confidence']), details

    def _apply_temporal_smoothing(self, camera, detected, confidence, details):
        state = self._states.get(camera, {'misses': 0, 'confidence': 0.0, 'details': None})
        if detected:
            previous_details = state.get('details') or {}
            smoothed_bbox = _smooth_bbox(previous_details.get('bbox'), details.get('bbox'))
            previous_polygons = previous_details.get('polygons') or []
            current_polygons = details.get('polygons') or []
            if smoothed_bbox:
                details['bbox'] = smoothed_bbox
            if previous_polygons and current_polygons:
                details['polygons'] = [_smooth_polygon(previous_polygons[0], current_polygons[0])]
            details['temporal_hold'] = False
            details['missed_frames'] = 0
            details['processing_method'] = 'yolo_segmentation_temporal'
            self._states[camera] = {'misses': 0, 'confidence': confidence, 'details': details}
            return True, confidence, details

        last_details = state.get('details')
        misses = int(state.get('misses', 0)) + 1
        if last_details and misses <= TEMPORAL_HOLD_FRAMES:
            held_details = dict(last_details)
            held_confidence = max(CONF_THRESHOLD, float(state.get('confidence', 0.0)) * (CONFIDENCE_DECAY ** misses))
            held_details.update({
                'processing_method': 'temporal_hold',
                'temporal_hold': True,
                'missed_frames': misses,
                'detections': details.get('detections', []),
            })
            self._states[camera] = {'misses': misses, 'confidence': state.get('confidence', held_confidence), 'details': last_details}
            return True, min(held_confidence, 1.0), held_details

        self._states[camera] = {'misses': 0, 'confidence': 0.0, 'details': None}
        details['temporal_hold'] = False
        details['missed_frames'] = misses
        return False, 0.0, details

    def _fallback_detection(self):
        return False, 0.0, {
            'model': 'YOLO_manhole_seg.pt',
            'processing_method': 'model_unavailable',
            'model_available': False,
            'bbox': None,
            'polygons': [],
            'manhole_state': 'unknown',
            'depth_available': False,
            'depth_source': depth_estimator.model_name,
            'depth_score': 0.0,
            'depth_level': 'unknown',
            'risk_level': 'low',
            'note': f'Modele YOLO introuvable: {MANHOLE_MODEL_PATH}',
        }


manhole_detector = ManholeDetector()


def estimate_manhole_depth(
    frame: np.ndarray,
    polygon: Optional[List[List[int]]] = None,
    bbox: Optional[List[int]] = None,
    state: str = 'unknown',
) -> Dict[str, Any]:
    return depth_estimator.estimate(frame, polygon=polygon, bbox=bbox, state=state)


def detect_manhole_in_frame(
    frame: np.ndarray,
    camera: str = 'cam5',
    include_depth: bool = False,
) -> Tuple[bool, float, Dict[str, Any]]:
    detected, confidence, details = manhole_detector.detect_manhole(frame, camera=camera)
    if include_depth and detected:
        polygon = (details.get('polygons') or [None])[0]
        bbox = details.get('bbox')
        state = details.get('manhole_state', 'unknown')
        details.update(estimate_manhole_depth(frame, polygon=polygon, bbox=bbox, state=state))
    else:
        details.update({
            'depth_available': False,
            'depth_source': 'deferred_report',
            'depth_score': None,
            'depth_level': 'pending',
        })
    return detected, confidence, details
