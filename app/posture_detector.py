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


def _geometric_posture_check(kpts_xy, kpts_conf):
    """Simple geometric rules for unsafe posture detection."""
    reasons = []
    confidence = 0.0

    # Keypoints indices (COCO-17):
    # 0:nose, 1:eye_l, 2:eye_r, 3:ear_l, 4:ear_r, 5:shoulder_l, 6:shoulder_r,
    # 7:elbow_l, 8:elbow_r, 9:wrist_l, 10:wrist_r, 11:hip_l, 12:hip_r,
    # 13:knee_l, 14:knee_r, 15:ankle_l, 16:ankle_r

    def get_point(idx):
        if idx >= len(kpts_xy) or (kpts_conf is not None and kpts_conf[idx] < 0.3):
            return None
        return kpts_xy[idx]

    # Check if person is visible
    shoulder_l = get_point(5)
    shoulder_r = get_point(6)
    hip_l = get_point(11)
    hip_r = get_point(12)
    knee_l = get_point(13)
    knee_r = get_point(14)
    ankle_l = get_point(15)
    ankle_r = get_point(16)

    if not all([shoulder_l, shoulder_r, hip_l, hip_r]):
        return False, 0.0, reasons  # Not enough keypoints

    # Calculate distances
    def dist(p1, p2):
        return np.linalg.norm(np.array(p1) - np.array(p2))

    # Shoulder width
    shoulder_width = dist(shoulder_l, shoulder_r)

    # Hip width
    hip_width = dist(hip_l, hip_r)

    # Check if knees are bent (distance from hip to knee vs knee to ankle)
    knee_bent = False
    if knee_l and ankle_l:
        thigh_len = dist(hip_l, knee_l)
        calf_len = dist(knee_l, ankle_l)
        if calf_len < thigh_len * 0.8:  # Bent knee
            knee_bent = True
            reasons.append("genou gauche plié")
            confidence = max(confidence, 0.7)

    if knee_r and ankle_r:
        thigh_len = dist(hip_r, knee_r)
        calf_len = dist(knee_r, ankle_r)
        if calf_len < thigh_len * 0.8:
            knee_bent = True
            reasons.append("genou droit plié")
            confidence = max(confidence, 0.7)

    # Check if back is not straight (shoulders and hips alignment)
    shoulder_center = ((shoulder_l[0] + shoulder_r[0])/2, (shoulder_l[1] + shoulder_r[1])/2)
    hip_center = ((hip_l[0] + hip_r[0])/2, (hip_l[1] + hip_r[1])/2)

    # Vertical alignment
    shoulder_hip_dist = dist(shoulder_center, hip_center)
    if shoulder_hip_dist > shoulder_width * 0.5:  # Leaning
        reasons.append("dos courbé")
        confidence = max(confidence, 0.6)

    # If any unsafe condition
    unsafe = len(reasons) > 0
    if unsafe and confidence == 0.0:
        confidence = 0.5

    return unsafe, confidence, reasons


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
                # Use geometric rules as primary detection
                geom_unsafe, geom_conf, geom_reasons = _geometric_posture_check(kps_xy, kpts_conf)
                cls_id, cls_conf = 0, 0.50
                predictions = []
                if _posture_model is not None:
                    pres = _posture_model(skel_img, conf=0.01, verbose=False)  # Lower threshold
                    for pr in pres:
                        if pr.boxes and len(pr.boxes) > 0:
                            for box in pr.boxes:
                                cls_id_box = int(box.cls)
                                conf_box = float(box.conf)
                                predictions.append({'class_id': cls_id_box, 'confidence': conf_box})
                            best = max(pr.boxes, key=lambda b: float(b.conf))
                            cls_id = int(best.cls)
                            cls_conf = float(best.conf)
                            break
                else:
                    predictions = [{'error': 'model not loaded'}]

                # Use geometric if model doesn't detect unsafe, or combine
                final_unsafe = geom_unsafe or (cls_id == 1)
                final_conf = max(geom_conf, cls_conf) if final_unsafe else cls_conf
                final_reasons = geom_reasons + (['model_prediction'] if cls_id == 1 else [])

                persons.append({
                    'bbox':               bbox,
                    'class':              'unsafe' if final_unsafe else 'safe',
                    'class_id':           1 if final_unsafe else 0,
                    'confidence':         final_conf,
                    'predictions':        predictions,
                    'geometric_unsafe':   geom_unsafe,
                    'geometric_conf':     geom_conf,
                    'geometric_reasons':  geom_reasons,
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
        details['classified_as']      = target['class']
        details['classification_conf'] = target['confidence']
        details['unsafe_count']       = len(unsafe_persons)
        details['total_persons']      = len(persons)
        details['reasons']            = target.get('geometric_reasons', [])

        if unsafe_persons:
            return True, target['confidence'], details
        return False, target['confidence'], details

    except Exception as e:
        details['error'] = str(e)
        details['processing_method'] = 'error'
        return False, 0.0, details
