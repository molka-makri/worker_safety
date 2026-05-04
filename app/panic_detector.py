"""
Panic detector — CAM 10.
Uses YOLOv8n-pose to extract keypoints, computes 8 biomechanical angles per frame,
and feeds a 30-frame rolling window into a bidirectional LSTM (PanicLSTM) to classify
NORMAL vs PANIC behaviour.

A per-camera deque accumulates feature vectors across successive API calls.
Detection returns False (with low confidence) until the buffer reaches 30 frames.
"""

import os
import numpy as np
from collections import deque

import torch
import torch.nn as nn

# ── Model paths ────────────────────────────────────────────────────────────────
_POSE_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'yolov8n-pose.pt')
_LSTM_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'panic.pt')

SEQ_LEN = 30       # frames required before classification
N_FEATURES = 8     # biomechanical angle features
CONF_THRESH = 0.40 # YOLOv8 keypoint detection threshold
PANIC_THRESHOLD = 0.30  # softmax probability to declare PANIC
PANIC_WARNING_THRESHOLD = 0.18  # softmax probability to show POSSIBLE panic

_pose_model = None
_lstm_model = None
_frame_buffers: dict = {}   # {camera_id: deque(maxlen=SEQ_LEN)}


# ── PanicLSTM ─────────────────────────────────────────────────────────────────

class PanicLSTM(nn.Module):
    """
    Bidirectional LSTM with temporal attention for panic detection.
    Input : (batch, SEQ_LEN, N_FEATURES=8)
    Output: (batch, 2)  — logits for [NORMAL, PANIC]
    """

    def __init__(self, n_features=8, hidden=128, n_layers=2, dropout=0.35):
        super().__init__()
        self.input_bn = nn.BatchNorm1d(n_features)
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
            bidirectional=True,
        )
        self.attention = nn.Sequential(
            nn.Linear(hidden * 2, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 2),
        )

    def forward(self, x):
        B, T, F = x.shape
        x_flat = x.reshape(B * T, F)
        x_flat = self.input_bn(x_flat)
        x = x_flat.reshape(B, T, F)
        out, _ = self.lstm(x)
        att_w = torch.softmax(self.attention(out), dim=1)
        context = (att_w * out).sum(dim=1)
        return self.classifier(context)


# ── Angle extraction ───────────────────────────────────────────────────────────

def _angle_vec_vertical(v):
    v = np.array(v, dtype=float)
    n = np.linalg.norm(v)
    if n < 1e-6:
        return 0.0
    cos_a = -v[1] / n
    return float(np.degrees(np.arccos(np.clip(cos_a, -1.0, 1.0))))


def _angle_between(v1, v2):
    v1, v2 = np.array(v1, dtype=float), np.array(v2, dtype=float)
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return 0.0
    cos_a = np.dot(v1, v2) / (n1 * n2)
    return float(np.degrees(np.arccos(np.clip(cos_a, -1.0, 1.0))))


def _extract_features(kpts):
    """
    Compute 8 biomechanical angle features from (17, 2) COCO keypoints.
    Returns np.ndarray of shape (8,).
    """
    kpts = np.array(kpts, dtype=float)

    nose = kpts[0]
    l_sho, r_sho = kpts[5], kpts[6]
    l_elb, r_elb = kpts[7], kpts[8]
    l_wri, r_wri = kpts[9], kpts[10]
    l_hip, r_hip = kpts[11], kpts[12]
    l_kne, r_kne = kpts[13], kpts[14]
    l_ank, r_ank = kpts[15], kpts[16]

    mid_sho = (l_sho + r_sho) / 2
    mid_hip = (l_hip + r_hip) / 2

    spine_vec = mid_sho - mid_hip
    spine_lean = _angle_vec_vertical(spine_vec)

    l_thigh_ang = _angle_vec_vertical(l_kne - l_hip)
    r_thigh_ang = _angle_vec_vertical(r_kne - r_hip)
    stride_angle = abs(l_thigh_ang - r_thigh_ang)

    l_arm_ang = _angle_vec_vertical(l_elb - l_sho)
    r_arm_ang = _angle_vec_vertical(r_elb - r_sho)
    arm_asymmetry = abs(l_arm_ang - r_arm_ang)

    l_kf = _angle_between(l_kne - l_hip, l_ank - l_kne)
    r_kf = _angle_between(r_kne - r_hip, r_ank - r_kne)
    avg_knee_flex = 180 - (l_kf + r_kf) / 2

    avg_arm_elevation = (l_arm_ang + r_arm_ang) / 2

    lateral_vec = r_sho - l_sho
    lateral_lean = abs(_angle_between(spine_vec, lateral_vec) - 90)

    head_vec = nose - mid_sho
    head_forward = _angle_between(head_vec, spine_vec)

    all_x, all_y = kpts[:, 0], kpts[:, 1]
    w = max(all_x) - min(all_x) + 1e-6
    h = max(all_y) - min(all_y) + 1e-6
    body_compactness = h / w

    return np.array([
        spine_lean, stride_angle, arm_asymmetry, avg_knee_flex,
        avg_arm_elevation, lateral_lean, head_forward, body_compactness,
    ], dtype=np.float32)


# ── Model loaders ──────────────────────────────────────────────────────────────

def _get_pose_model():
    global _pose_model
    if _pose_model is None:
        try:
            from ultralytics import YOLO
            _pose_model = YOLO(os.path.abspath(_POSE_PATH))
        except Exception:
            _pose_model = None
    return _pose_model


def _get_lstm_model():
    global _lstm_model
    if _lstm_model is None:
        try:
            model = PanicLSTM()
            state = torch.load(os.path.abspath(_LSTM_PATH), map_location='cpu', weights_only=False)
            model.load_state_dict(state, strict=True)
            model.eval()
            _lstm_model = model
        except Exception as e:
            import traceback; traceback.print_exc()
            _lstm_model = None
    return _lstm_model


# ── Main detection function ────────────────────────────────────────────────────

def detect_panic_in_frame(frame, camera='cam10'):
    """
    Detect panic behaviour in a single BGR frame (CAM 10).

    Internally maintains a 30-frame rolling buffer per camera.
    Returns (panic_detected, confidence, details).
    Returns (False, 0.0, ...) while the buffer is still warming up (<30 frames).
    """
    global _frame_buffers

    if camera not in _frame_buffers:
        _frame_buffers[camera] = deque(maxlen=SEQ_LEN)

    buf = _frame_buffers[camera]

    try:
        pose = _get_pose_model()
        lstm = _get_lstm_model()

        if pose is None or lstm is None:
            return False, 0.0, {
                'error': 'Models unavailable (pose or LSTM)',
                'model_available': False,
                'camera': camera,
                'buffer_size': len(buf),
            }

        # Pose inference
        results = pose(frame, verbose=False, conf=CONF_THRESH)

        features = None
        persons_detected = 0

        if results and results[0].keypoints is not None and len(results[0].keypoints) > 0:
            # Use the first detected person (primary subject)
            kp = results[0].keypoints[0]
            kpts_xy = kp.xy.cpu().numpy()
            if kpts_xy.ndim == 3:
                kpts_xy = kpts_xy[0]
            if kpts_xy.shape[0] >= 17:
                features = _extract_features(kpts_xy)
                persons_detected = len(results[0].keypoints)

        if features is None:
            features = np.zeros(N_FEATURES, dtype=np.float32)

        buf.append(features)
        frames_collected = len(buf)

        if frames_collected < SEQ_LEN:
            return False, 0.0, {
                'panic_detected': False,
                'label': 'WARMING_UP',
                'confidence': 0.0,
                'frames_collected': frames_collected,
                'frames_needed': SEQ_LEN,
                'persons_detected': persons_detected,
                'camera': camera,
                'model_available': True,
            }

        # Build input tensor (1, SEQ_LEN, N_FEATURES)
        seq = np.stack(list(buf), axis=0)        # (30, 8)
        x = torch.tensor(seq, dtype=torch.float32).unsqueeze(0)  # (1, 30, 8)

        with torch.no_grad():
            logits = lstm(x)                     # (1, 2)
            probs = torch.softmax(logits, dim=1)[0]  # [p_normal, p_panic]
            p_panic = float(probs[1].item())

        panic_detected = p_panic >= PANIC_THRESHOLD
        possible_panic = not panic_detected and p_panic >= PANIC_WARNING_THRESHOLD

        return panic_detected, float(p_panic), {
            'panic_detected': panic_detected,
            'possible_panic': possible_panic,
            'label': 'PANIC' if panic_detected else ('POSSIBLE' if possible_panic else 'NORMAL'),
            'confidence': float(p_panic),
            'p_normal': float(probs[0].item()),
            'p_panic': p_panic,
            'frames_collected': frames_collected,
            'persons_detected': persons_detected,
            'camera': camera,
            'model_available': True,
        }

    except Exception as e:
        return False, 0.0, {
            'error': str(e),
            'model_available': False,
            'camera': camera,
            'buffer_size': len(buf),
        }
