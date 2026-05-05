import os
from typing import Any, Dict, List, Tuple

import cv2

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FIRE_SMOKE_MODEL_PATH = os.path.abspath(
    os.path.join(BASE_DIR, "..", "models", "fire_smoke_detection.pt")
)

CONF_THRESHOLD = 0.25

FIRE_LABEL_HINTS = {"fire", "flame", "feu", "incendie"}
SMOKE_LABEL_HINTS = {"smoke", "fumee", "fumée", "fumee", "fume"}

YOLO_AVAILABLE = False
yolo_model = None

try:
    from ultralytics import YOLO

    if os.path.exists(FIRE_SMOKE_MODEL_PATH):
        yolo_model = YOLO(FIRE_SMOKE_MODEL_PATH)
        YOLO_AVAILABLE = True
        print(f"[FireSmokeDetector] OK: Model loaded: {FIRE_SMOKE_MODEL_PATH}")
    else:
        print(f"[FireSmokeDetector] WARNING: Model not found: {FIRE_SMOKE_MODEL_PATH}")
except ImportError as exc:
    print(f"[FireSmokeDetector] WARNING: ultralytics not installed: {exc}")


def _classify_label(label: str) -> str:
    normalized = (label or "").lower().strip()
    if any(hint in normalized for hint in FIRE_LABEL_HINTS):
        return "fire"
    if any(hint in normalized for hint in SMOKE_LABEL_HINTS):
        return "smoke"
    return "other"


class FireSmokeDetector:
    def __init__(self):
        self.model = yolo_model
        self.available = YOLO_AVAILABLE

    def detect_fire_smoke(
        self, frame: cv2.typing.MatLike, camera: str = "cam12"
    ) -> Tuple[bool, float, Dict[str, Any]]:
        if not self.available or self.model is None:
            return False, 0.0, {
                "model": "YOLO_fire_smoke_detection.pt",
                "model_available": False,
                "camera": camera,
                "note": f"Modele YOLO introuvable: {FIRE_SMOKE_MODEL_PATH}",
            }

        try:
            results = self.model(frame, conf=CONF_THRESHOLD, verbose=False)
            return self._parse_results(results, camera=camera)
        except Exception as exc:
            print(f"[FireSmokeDetector] YOLO error: {exc}")
            return False, 0.0, {
                "model": "YOLO_fire_smoke_detection.pt",
                "model_available": False,
                "camera": camera,
                "error": str(exc),
            }

    def _parse_results(self, results, camera: str) -> Tuple[bool, float, Dict[str, Any]]:
        detections: List[Dict[str, Any]] = []
        fire_detected = False
        smoke_detected = False
        max_confidence = 0.0
        primary_bbox = None
        primary_type = "other"

        for result in results:
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                continue

            for box in boxes:
                cls_id = int(box.cls[0])
                confidence = float(box.conf[0])
                label = result.names.get(cls_id, str(cls_id))
                x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                detection_type = _classify_label(label)

                detections.append(
                    {
                        "class_id": cls_id,
                        "label": label,
                        "type": detection_type,
                        "confidence": round(confidence, 3),
                        "bbox": [x1, y1, x2, y2],
                    }
                )

                if detection_type == "fire":
                    fire_detected = True
                elif detection_type == "smoke":
                    smoke_detected = True

                if confidence > max_confidence:
                    max_confidence = confidence
                    primary_bbox = [x1, y1, x2, y2]
                    primary_type = detection_type

        any_detection = len(detections) > 0
        incident_detected = fire_detected or smoke_detected or any_detection

        details = {
            "model": "YOLO_fire_smoke_detection.pt",
            "processing_method": "yolo_detection",
            "model_available": True,
            "camera": camera,
            "detections": detections,
            "bbox": primary_bbox,
            "primary_type": primary_type,
            "fire_detected": fire_detected,
            "smoke_detected": smoke_detected,
        }

        return incident_detected, min(max_confidence, 1.0), details


fire_smoke_detector = FireSmokeDetector()


def detect_fire_smoke_in_frame(
    frame: cv2.typing.MatLike, camera: str = "cam12"
) -> Tuple[bool, float, Dict[str, Any]]:
    return fire_smoke_detector.detect_fire_smoke(frame, camera=camera)
