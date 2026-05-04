"""
Posture detector — CAM 9.
Uses YOLOv8n-pose to extract COCO-17 keypoints then applies geometric rules
to classify each worker's posture as SAFE or UNSAFE.
"""

import os
import numpy as np

_pose_model = None
_POSE_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'yolov8n-pose.pt')

# COCO-17 keypoint indices
NOSE = 0
L_SH, R_SH = 5, 6
L_EL, R_EL = 7, 8
L_WR, R_WR = 9, 10
L_HIP, R_HIP = 11, 12
L_KN, R_KN = 13, 14
L_AN, R_AN = 15, 16

UNSAFE_THRESHOLD = 0.28  # cumulative risk score to flag UNSAFE


def _get_pose_model():
    global _pose_model
    if _pose_model is None:
        try:
            from ultralytics import YOLO
            _pose_model = YOLO(os.path.abspath(_POSE_PATH))
        except Exception:
            _pose_model = None
    return _pose_model


def _valid(kpts, *idxs):
    return all(kpts[i][0] > 0 or kpts[i][1] > 0 for i in idxs)


def _mid(kpts, i, j):
    return (kpts[i] + kpts[j]) / 2.0


def _angle_from_vertical_deg(vec):
    vx, vy = float(vec[0]), float(vec[1])
    if abs(vx) < 1e-6 and abs(vy) < 1e-6:
        return 0.0
    return float(np.degrees(np.arctan2(abs(vx), abs(vy) + 1e-9)))


def _classify_posture(kpts):
    """
    Rule-based posture classification.
    kpts: (17, 2) float array — pixel (x, y) per COCO-17 joint.
    Returns: (label, confidence, reasons)
    """
    kpts = np.asarray(kpts, dtype=float)
    reasons = []
    risk = 0.0

    have_shoulders = _valid(kpts, L_SH, R_SH)
    have_hips = _valid(kpts, L_HIP, R_HIP)
    have_ankles = _valid(kpts, L_AN, R_AN)
    have_nose = _valid(kpts, NOSE)

    if not (have_shoulders and have_hips):
        return 'safe', 0.50, ['insufficient keypoints']

    mid_sh = _mid(kpts, L_SH, R_SH)
    mid_hip = _mid(kpts, L_HIP, R_HIP)
    sh_w = max(abs(kpts[L_SH][0] - kpts[R_SH][0]), 30.0)
    torso_h = max(abs(mid_sh[1] - mid_hip[1]), 1.0)

    # 1. Spine lean
    spine_vec = mid_hip - mid_sh
    spine_lean = _angle_from_vertical_deg(spine_vec)
    if spine_lean > 50:
        r = min((spine_lean - 50) / 30.0, 1.0)
        risk += 0.40 * r
        reasons.append(f"spine lean {spine_lean:.0f}°")
    elif spine_lean > 35:
        r = (spine_lean - 35) / 15.0
        risk += 0.20 * r
        reasons.append(f"moderate spine lean {spine_lean:.0f}°")

    # 2. Forward shift
    fwd_shift = (mid_sh[0] - mid_hip[0]) / sh_w
    if abs(fwd_shift) > 0.7:
        r = min((abs(fwd_shift) - 0.7) / 0.8, 1.0)
        risk += 0.30 * r
        reasons.append(f"upper body forward shift {fwd_shift:.1f}×sw")

    # 3. Head position
    if have_nose:
        head_fwd = (kpts[NOSE][0] - mid_sh[0]) / sh_w
        if abs(head_fwd) > 0.8:
            risk += 0.15
            reasons.append(f"head forward {head_fwd:.1f}×sw")
        head_drop = (kpts[NOSE][1] - mid_sh[1]) / sh_w
        if head_drop > 1.0:
            risk += 0.20
            reasons.append(f"head dropped {head_drop:.1f}×sw")

    # 4. Lateral torso tilt
    lat_shift = abs(mid_sh[0] - mid_hip[0]) / torso_h
    if spine_lean < 35 and lat_shift > 0.40:
        r = min((lat_shift - 0.40) / 0.40, 1.0)
        risk += 0.20 * r
        reasons.append(f"lateral lean {lat_shift:.2f}")

    # 5. Low hands + lean (heavy-lift posture)
    have_wrists = _valid(kpts, L_WR, R_WR)
    if have_wrists:
        avg_wr_y = (kpts[L_WR][1] + kpts[R_WR][1]) / 2.0
        if avg_wr_y > mid_hip[1] and spine_lean > 25:
            risk += 0.25
            reasons.append("low hands + forward lean (lift risk)")

    # 6. Torso compression (deep squat / collapse)
    if have_ankles:
        mid_ank = _mid(kpts, L_AN, R_AN)
        lower_h = max(abs(mid_hip[1] - mid_ank[1]), 1.0)
        ratio = torso_h / lower_h
        if ratio < 0.40:
            risk += 0.20
            reasons.append(f"torso compressed (ratio {ratio:.2f})")

    # 7. Hip drop
    hip_tilt = abs(kpts[L_HIP][1] - kpts[R_HIP][1]) / sh_w
    if hip_tilt > 0.35:
        risk += 0.10
        reasons.append(f"hip drop {hip_tilt:.2f}×sw")

    risk = min(risk, 1.0)
    if risk >= UNSAFE_THRESHOLD:
        return 'unsafe', round(risk, 3), reasons
    return 'safe', round(1.0 - risk, 3), reasons


def detect_posture_in_frame(frame, camera='cam9'):
    """
    Detect unsafe posture in a single BGR frame (CAM 9).
    Returns: (unsafe_detected: bool, confidence: float, details: dict)
    """
    try:
        model = _get_pose_model()
        if model is None:
            return False, 0.0, {
                'error': 'Pose model unavailable',
                'model_available': False,
                'camera': camera,
            }

        results = model(frame, verbose=False, conf=0.40)

        if not results or results[0].keypoints is None or len(results[0].keypoints) == 0:
            return False, 0.5, {
                'posture': 'safe',
                'label': 'SAFE',
                'confidence': 0.5,
                'reasons': [],
                'persons_detected': 0,
                'camera': camera,
                'model_available': True,
            }

        worst_risk = -1.0
        worst_label = 'safe'
        worst_reasons = []
        worst_conf = 0.5
        worst_kpts = []
        worst_bbox = None
        persons = 0
        all_persons = []

        boxes = results[0].boxes  # bounding boxes (N, 4) xyxy

        for idx, kp in enumerate(results[0].keypoints):
            kpts_xy = kp.xy.cpu().numpy()
            if kpts_xy.ndim == 3:
                kpts_xy = kpts_xy[0]       # (17, 2)
            if kpts_xy.shape[0] < 17:
                continue

            label, conf, reasons = _classify_posture(kpts_xy)
            persons += 1

            # bounding box for this person
            bbox = None
            if boxes is not None and idx < len(boxes):
                b = boxes[idx].xyxy.cpu().numpy()[0]
                bbox = [float(b[0]), float(b[1]), float(b[2]), float(b[3])]

            risk_val = conf if label == 'unsafe' else 1.0 - conf
            all_persons.append({
                'label': label,
                'confidence': float(conf),
                'reasons': reasons,
                'keypoints': kpts_xy.tolist(),
                'bbox': bbox,
            })

            if risk_val > worst_risk:
                worst_risk = risk_val
                worst_label = label
                worst_reasons = reasons
                worst_conf = conf
                worst_kpts = kpts_xy.tolist()
                worst_bbox = bbox

        unsafe_detected = worst_label == 'unsafe'
        return unsafe_detected, float(worst_conf), {
            'posture_detected': unsafe_detected,
            'posture': worst_label,
            'label': worst_label.upper(),
            'confidence': float(worst_conf),
            'reasons': worst_reasons,
            'keypoints': worst_kpts,
            'bbox': worst_bbox,
            'persons_detected': persons,
            'all_persons': all_persons,
            'camera': camera,
            'model_available': True,
        }

    except Exception as e:
        return False, 0.0, {
            'error': str(e),
            'model_available': False,
            'camera': camera,
        }
