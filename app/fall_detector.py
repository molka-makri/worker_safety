#!/usr/bin/env python
"""
fall_detector.py — SafeVision AI
==================================
Détection de chute avec YOLO (fall_detection.pt).

IMPORTANT — classes de votre modèle :
  Le modèle retourne plusieurs classes. On filtre uniquement les personnes
  en état de chute en se basant sur :
    1. Le NOM de la classe (label contenant 'fall', 'down', 'chute', 'fallen')
    2. L'aspect ratio de la bounding box (personne allongée = largeur > hauteur)
    3. La position verticale dans le frame (personne au sol = bas de l'image)
"""

import cv2
import numpy as np
from app.hf_model_store import ensure_model_file
import os
from typing import Tuple, Dict, Any

# ── Chemin vers le modèle ──────────────────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
FALL_MODEL_PATH = ensure_model_file('fall_detection.pt')
DEBUG_FALL_DETECTOR = os.getenv("DEBUG_FALL_DETECTOR", "0").strip().lower() in {"1", "true", "yes", "on"}

CONF_THRESHOLD = 0.45

# ── Labels qui indiquent une chute (insensible à la casse) ────────────────────
FALL_LABELS = {'fall', 'fallen', 'down', 'chute', 'tombé', 'fall_down', 'falling'}

# ── Chargement YOLO ────────────────────────────────────────────────────────────
YOLO_AVAILABLE = False
yolo_model     = None

try:
    from ultralytics import YOLO

    if os.path.exists(FALL_MODEL_PATH):
        yolo_model     = YOLO(FALL_MODEL_PATH)
        YOLO_AVAILABLE = True
        print(f"[FallDetector] OK: Model loaded: {FALL_MODEL_PATH}")

        # ── SHOW ALL MODEL CLASSES AT STARTUP ──
        print("[FallDetector] Model classes:")
        for cls_id, cls_name in yolo_model.names.items():
            print(f"   class {cls_id} -> '{cls_name}'")
    else:
        print(f"[FallDetector] WARNING: Model not found: {FALL_MODEL_PATH}")
except ImportError as e:
    print(f"[FallDetector] WARNING: ultralytics not installed: {e}")


def _is_fall_label(label: str) -> bool:
    """Retourne True si le label correspond à une chute."""
    return label.lower().strip() in FALL_LABELS


def _is_fallen_by_shape(x1: int, y1: int, x2: int, y2: int, frame_h: int) -> bool:
    """
    Heuristique géométrique : une personne allongée a une bbox plus large que haute
    ET se trouve dans la moitié basse du frame.
    """
    w = x2 - x1
    h = y2 - y1
    if h == 0:
        return False
    aspect_ratio   = w / h
    center_y_ratio = (y1 + h / 2) / frame_h

    return aspect_ratio > 1.1 and center_y_ratio > 0.30


def _bbox_iou(a, b) -> float:
    if not a or not b:
        return 0.0
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1, (bx2 - bx1) * (by2 - by1))
    return inter / (area_a + area_b - inter)


class FallDetector:

    def __init__(self):
        self.model     = yolo_model
        self.available = YOLO_AVAILABLE
        self._fall_streak = 0
        self._last_fall_bbox = None

    def detect_fall(self, frame: np.ndarray) -> Tuple[bool, float, Dict[str, Any]]:
        if not self.available or self.model is None:
            return self._fallback_detection(frame)
        try:
            results = self.model(frame, conf=CONF_THRESHOLD, verbose=False)
            return self._parse_results(results, frame.shape)
        except Exception as e:
            print(f"[FallDetector] Erreur YOLO : {e}")
            return self._fallback_detection(frame)

    def _parse_results(self, results, frame_shape) -> Tuple[bool, float, Dict[str, Any]]:
        height, width   = frame_shape[:2]
        raw_fall_detected = False
        best_confidence = 0.0
        best_bbox       = None
        all_detections  = []

        for result in results:
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                continue

            for box in boxes:
                cls_id         = int(box.cls[0])
                conf           = float(box.conf[0])
                xyxy           = box.xyxy[0].tolist()
                x1, y1, x2, y2 = [int(v) for v in xyxy]
                label          = result.names.get(cls_id, str(cls_id))

                # DEBUG — visible in Django logs / terminal
                if DEBUG_FALL_DETECTOR:
                    print(f"[FallDetector] DETECT cls={cls_id} label='{label}' conf={conf:.2f} "
                          f"bbox=[{x1},{y1},{x2},{y2}]")

                all_detections.append({
                    'class_id':   cls_id,
                    'label':      label,
                    'confidence': round(conf, 3),
                    'bbox':       [x1, y1, x2, y2],
                })

                # Critère 1 : label explicitement de type "fall/down/chute…"
                by_label = _is_fall_label(label)

                # Critère 2 : forme horizontale + position basse dans le frame
                by_shape = _is_fallen_by_shape(x1, y1, x2, y2, height)

                is_fall = by_label and conf >= CONF_THRESHOLD

                if DEBUG_FALL_DETECTOR:
                    print(f"[FallDetector]    → by_label={by_label}  by_shape={by_shape} "
                          f"conf_ok={conf >= CONF_THRESHOLD}  → is_fall={is_fall}")

                if is_fall and conf > best_confidence:
                    best_confidence = conf
                    raw_fall_detected = True
                    best_bbox       = [x1, y1, x2, y2]

        if raw_fall_detected:
            same_target = _bbox_iou(best_bbox, self._last_fall_bbox) > 0.12
            self._fall_streak = self._fall_streak + 1 if same_target else 1
            self._last_fall_bbox = best_bbox
        else:
            self._fall_streak = 0
            self._last_fall_bbox = None

        fall_detected = raw_fall_detected and (self._fall_streak >= 2 or best_confidence >= 0.75)

        details = {
            'model':             'YOLO_fall_detection.pt',
            'processing_method': 'yolo_inference',
            'detections':        all_detections,
            'bbox':              best_bbox,
            'fall_class_found':  fall_detected,
            'raw_fall_found':    raw_fall_detected,
            'fall_streak':       self._fall_streak,
        }

        return fall_detected, min(best_confidence, 1.0) if fall_detected else 0.0, details

    def _fallback_detection(self, frame: np.ndarray) -> Tuple[bool, float, Dict[str, Any]]:
        return False, 0.0, {
            'model':             'YOLO_fall_detection.pt',
            'processing_method': 'model_unavailable',
            'bbox':              None,
            'model_available':   False,
            'note':              f'Modele YOLO introuvable: {FALL_MODEL_PATH}',
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
fall_detector = FallDetector()


def detect_fall_in_frame(frame: np.ndarray) -> Tuple[bool, float, Dict[str, Any]]:
    return fall_detector.detect_fall(frame)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
if __name__ == '__main__':
    """
    Lance ce script directement sur votre vidéo pour voir
    EXACTEMENT ce que le modèle détecte frame par frame.

    Usage :
        python fall_detector.py
        python fall_detector.py chemin/vers/worker_falling3.mp4
    """
    import sys

    video_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(BASE_DIR, '..', 'media', 'worker_falling3.mp4')

    if not os.path.exists(video_path):
        print(f"❌ Vidéo introuvable : {video_path}")
        sys.exit(1)

    print(f"\n🎬 Analyse de : {video_path}")
    print("=" * 60)

    cap        = cv2.VideoCapture(video_path)
    frame_idx  = 0
    fall_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        if frame_idx % 10 != 0:   # analyser 1 frame sur 10
            continue

        detected, conf, details = detect_fall_in_frame(frame)

        state = "🔴 FALL" if detected else "🟢 NORMAL"
        print(f"\nFrame {frame_idx:04d} → {state}  conf={conf:.2f}")
        for d in details.get('detections', []):
            print(f"   cls={d['class_id']} '{d['label']}' conf={d['confidence']} bbox={d['bbox']}")

        if detected:
            fall_count += 1

        # Affichage avec annotation YOLO
        if YOLO_AVAILABLE:
            results   = yolo_model(frame, conf=CONF_THRESHOLD, verbose=False)
            annotated = results[0].plot()
            color     = (0, 0, 255) if detected else (0, 255, 0)
            cv2.putText(annotated, "FALL" if detected else "NORMAL",
                        (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)
            cv2.imshow("Fall Detection Debug", annotated)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()
    print(f"\n{'='*60}")
    print(f"✅ Terminé — {fall_count} chutes sur {frame_idx // 10} frames analysées")
