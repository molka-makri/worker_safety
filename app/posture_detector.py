"""
Posture detector — two-stage pipeline matching the training notebook:
  1. Extract pose keypoints (yolov8n-pose.pt)
  2. Render skeleton on 640×640 black background (same distribution as training data)
  3. Classify safe / unsafe with posture.pt (99.2% mAP50)
"""
import cv2
import numpy as np
import os
from pathlib import Path

BASE_DIR = Path(os.path.dirname(__file__)).parent
POSTURE_MODEL_PATH = BASE_DIR / 'models' / 'posture.pt'
POSE_MODEL_PATH    = BASE_DIR / 'models' / 'yolov8n-pose.pt'

_posture_model = None
_pose_model    = None

IMG_SIZE = 640

# COCO-17 skeleton — identical to notebook CONNECTIONS / CONN_COLORS / JOINT_COLORS
CONNECTIONS = [
    (0, 1), (0, 2), (1, 3), (2, 4), (5, 6),
    (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
]
CONN_COLORS = [
    (100, 255, 255), (100, 255, 255), (80, 200, 255), (80, 200, 255),
    (0, 255, 80),
    (255, 180, 0), (255, 180, 0), (0, 160, 255), (0, 160, 255),
    (0, 255, 120), (0, 255, 120), (0, 255, 80),
    (255, 80, 80), (255, 80, 80), (80, 80, 255), (80, 80, 255),
]
JOINT_COLORS = [
    (100, 255, 255), (100, 230, 255), (100, 230, 255), (80, 200, 255), (80, 200, 255),
    (255, 200, 0), (0, 200, 255), (255, 150, 0), (0, 150, 255), (255, 100, 0), (0, 100, 255),
    (0, 255, 120), (0, 255, 120), (255, 80, 80), (80, 80, 255), (220, 40, 40), (40, 40, 220),
]


def _load_models():
    global _posture_model, _pose_model
    if _posture_model is None and POSTURE_MODEL_PATH.exists():
        try:
            from ultralytics import YOLO
            _posture_model = YOLO(str(POSTURE_MODEL_PATH))
        except Exception:
            pass
    if _pose_model is None and POSE_MODEL_PATH.exists():
        try:
            from ultralytics import YOLO
            _pose_model = YOLO(str(POSE_MODEL_PATH))
        except Exception:
            pass


def _skeleton_on_black(kpts_xy, kpts_conf, src_w, src_h):
    """Render COCO-17 keypoints on a black 640x640 image (matches training format)."""
    img = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
    scale_x = IMG_SIZE / max(src_w, 1)
    scale_y = IMG_SIZE / max(src_h, 1)

    pts = []
    for i in range(min(17, len(kpts_xy))):
        cx = int(kpts_xy[i][0] * scale_x)
        cy = int(kpts_xy[i][1] * scale_y)
        conf = float(kpts_conf[i]) if kpts_conf is not None and i < len(kpts_conf) else 1.0
        visible = conf >= 0.3 and 0 < cx < IMG_SIZE and 0 < cy < IMG_SIZE
        pts.append((cx, cy, visible))

    for idx, (a, b) in enumerate(CONNECTIONS):
        if a < len(pts) and b < len(pts) and pts[a][2] and pts[b][2]:
            cv2.line(img, (pts[a][0], pts[a][1]), (pts[b][0], pts[b][1]),
                     CONN_COLORS[idx % len(CONN_COLORS)], 3, cv2.LINE_AA)
    for i, (cx, cy, visible) in enumerate(pts):
        if visible:
            cv2.circle(img, (cx, cy), 6, JOINT_COLORS[i % len(JOINT_COLORS)], -1, cv2.LINE_AA)

    return img


def detect_posture_in_frame(frame, camera='cam9'):
    """
    Returns (unsafe_detected: bool, confidence: float, details: dict).
    details['detections'] — list of per-person results with bbox + skeleton_keypoints.
    details['skeleton_keypoints'] — COCO-17 keypoints [[x,y], ...] of the worst person.
    """
    _load_models()
    h, w = frame.shape[:2]
    details = {
        'camera': camera,
        'model': 'posture.pt + yolov8n-pose.pt',
        'model_available': _posture_model is not None,
        'pose_available': _pose_model is not None,
        'detections': [],
        'skeleton_keypoints': [],
        'keypoints_conf': [],
        'processing_method': 'two_stage_pose_posture',
    }

    if _pose_model is None:
        details['processing_method'] = 'no_pose_model'
        return False, 0.0, details

    try:
        pose_results = _pose_model(frame, conf=0.30, verbose=False)
        persons = []

        for res in pose_results:
            if res.boxes is None or res.keypoints is None:
                continue
            for i in range(len(res.boxes)):
                bbox     = list(map(int, res.boxes[i].xyxy[0].tolist()))
                kps_xy   = res.keypoints.xy[i].cpu().numpy()
                kps_conf = (res.keypoints.conf[i].cpu().numpy()
                            if res.keypoints.conf is not None else None)

                skel_img = _skeleton_on_black(kps_xy, kps_conf, w, h)

                cls_id, cls_conf = 0, 0.50
                if _posture_model is not None:
                    pres = _posture_model(skel_img, conf=0.20, verbose=False)
                    for pr in pres:
                        if pr.boxes and len(pr.boxes) > 0:
                            best     = max(pr.boxes, key=lambda b: float(b.conf))
                            cls_id   = int(best.cls)
                            cls_conf = float(best.conf)
                            break

                persons.append({
                    'bbox':               bbox,
                    'class':              'unsafe' if cls_id == 1 else 'safe',
                    'class_id':           cls_id,
                    'confidence':         cls_conf,
                    'skeleton_keypoints': kps_xy.tolist(),
                    'keypoints_conf':     kps_conf.tolist() if kps_conf is not None else [],
                })

        details['detections'] = persons

        if not persons:
            details['processing_method'] = 'no_person_detected'
            return False, 0.0, details

        unsafe_persons = [p for p in persons if p['class_id'] == 1]
        target = (max(unsafe_persons, key=lambda p: p['confidence'])
                  if unsafe_persons
                  else max(persons, key=lambda p: p['confidence']))

        details['bbox']               = target['bbox']
        details['skeleton_keypoints'] = target['skeleton_keypoints']
        details['keypoints_conf']     = target.get('keypoints_conf', [])

        if unsafe_persons:
            return True, target['confidence'], details
        return False, target['confidence'], details

    except Exception as e:
        details['error'] = str(e)
        details['processing_method'] = 'error'
        return False, 0.0, details
