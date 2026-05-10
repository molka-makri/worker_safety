"""
Posture detector — two-stage pipeline with multi-person tracking:
  1. YOLOv8n-pose + ByteTrack  → COCO-17 keypoints + persistent track ID per person
     (MediaPipe fallback for single-person when YOLO unavailable)
  2. Render each person's skeleton on 640×640 black background
  3. Classify safe / unsafe with posture.pt  (one inference per person)
  4. Annotate original frame: coloured BBX + skeleton + #ID label
"""
import base64
import cv2
import numpy as np
import os
from pathlib import Path
from app.hf_model_store import ensure_model_file

BASE_DIR           = Path(os.path.dirname(__file__)).parent
POSTURE_MODEL_PATH = Path(ensure_model_file('posture.pt'))
POSE_MODEL_PATH    = Path(ensure_model_file('yolov8n-pose.pt'))

_posture_model = None
_pose_model    = None
_mp_pose       = None

IMG_SIZE = 640

# COCO-17 skeleton
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

# MediaPipe → COCO-17
_MP_TO_COCO = {
    0: 0, 2: 1, 5: 2, 7: 3, 8: 4,
    11: 5, 12: 6, 13: 7, 14: 8, 15: 9, 16: 10,
    23: 11, 24: 12, 25: 13, 26: 14, 27: 15, 28: 16,
}


# ── Model loaders ──────────────────────────────────────────────────────────────

def _load_models():
    global _posture_model, _pose_model
    if _posture_model is None and POSTURE_MODEL_PATH.exists():
        try:
            from ultralytics import YOLO
            _posture_model = YOLO(str(POSTURE_MODEL_PATH))
            print(f"[PostureDetector] OK: posture model loaded: {POSTURE_MODEL_PATH}")
        except Exception as exc:
            print(f"[PostureDetector] WARNING: posture model unavailable: {exc}")
    if _pose_model is None and POSE_MODEL_PATH.exists():
        try:
            from ultralytics import YOLO
            _pose_model = YOLO(str(POSE_MODEL_PATH))
            print(f"[PostureDetector] OK: pose model loaded: {POSE_MODEL_PATH}")
        except Exception as exc:
            print(f"[PostureDetector] WARNING: pose model unavailable: {exc}")


def _load_mediapipe():
    global _mp_pose
    if _mp_pose is not None:
        return _mp_pose
    try:
        import mediapipe as mp
        _mp_pose = mp.solutions.pose.Pose(
            static_image_mode=True,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.4,
        )
    except Exception:
        _mp_pose = None
        print("[PostureDetector] WARNING: MediaPipe unavailable")
    return _mp_pose


# ── Pose extraction ────────────────────────────────────────────────────────────

def _yolo_persons_tracked(frame):
    """
    YOLOv8n-pose + ByteTrack → multi-person with persistent IDs.
    Returns list of dicts: {kps_xy, kps_conf, bbox, track_id, source}.
    """
    if _pose_model is None:
        return []

    h, w = frame.shape[:2]
    try:
        results = _pose_model.track(
            frame,
            persist=True,
            conf=0.30,
            iou=0.45,
            tracker='bytetrack.yaml',
            verbose=False,
        )
    except Exception:
        # bytetrack not available — fall back to simple inference
        results = _pose_model(frame, conf=0.30, verbose=False)

    persons = []
    for res in results:
        if res.boxes is None or res.keypoints is None:
            continue
        for i in range(len(res.boxes)):
            kps_xy   = res.keypoints.xy[i].cpu().numpy()
            kps_conf = (res.keypoints.conf[i].cpu().numpy()
                        if res.keypoints.conf is not None
                        else np.ones(17, dtype=np.float32))
            if kps_xy.shape[0] < 17:
                continue

            bbox = list(map(int, res.boxes[i].xyxy[0].tolist()))

            # Track ID from ByteTrack (None if tracking unavailable)
            track_id = None
            if res.boxes.id is not None and i < len(res.boxes.id):
                track_id = int(res.boxes.id[i].item())

            persons.append({
                'kps_xy'  : kps_xy,
                'kps_conf': kps_conf,
                'bbox'    : bbox,
                'track_id': track_id,
                'source'  : 'yolov8+bytetrack',
            })

    return persons


def _mediapipe_persons(frame):
    """MediaPipe Pose — single person, no tracking (fallback only)."""
    mp_pose = _load_mediapipe()
    if mp_pose is None:
        return []

    h, w = frame.shape[:2]
    rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = mp_pose.process(rgb)

    if not result.pose_landmarks:
        return []

    lms      = result.pose_landmarks.landmark
    kps_xy   = np.zeros((17, 2), dtype=np.float32)
    kps_conf = np.zeros(17, dtype=np.float32)
    xs, ys   = [], []

    for mp_idx, coco_idx in _MP_TO_COCO.items():
        lm = lms[mp_idx]
        kps_xy[coco_idx]   = [lm.x * w, lm.y * h]
        kps_conf[coco_idx] = lm.visibility
        if lm.visibility > 0.3:
            xs.append(lm.x * w)
            ys.append(lm.y * h)

    pad  = 20
    bbox = [max(0, int(min(xs)) - pad), max(0, int(min(ys)) - pad),
            min(w, int(max(xs)) + pad), min(h, int(max(ys)) + pad)] if xs else [0, 0, w, h]

    return [{'kps_xy': kps_xy, 'kps_conf': kps_conf, 'bbox': bbox,
             'track_id': None, 'source': 'mediapipe'}]


# ── Skeleton rendering ─────────────────────────────────────────────────────────

def _skeleton_on_black(kps_xy, kps_conf, src_w, src_h):
    img     = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
    scale_x = IMG_SIZE / max(src_w, 1)
    scale_y = IMG_SIZE / max(src_h, 1)

    pts = []
    for i in range(min(17, len(kps_xy))):
        cx   = int(kps_xy[i][0] * scale_x)
        cy   = int(kps_xy[i][1] * scale_y)
        conf = float(kps_conf[i]) if kps_conf is not None and i < len(kps_conf) else 1.0
        pts.append((cx, cy, conf >= 0.3 and 0 < cx < IMG_SIZE and 0 < cy < IMG_SIZE))

    for idx, (a, b) in enumerate(CONNECTIONS):
        if a < len(pts) and b < len(pts) and pts[a][2] and pts[b][2]:
            cv2.line(img, pts[a][:2], pts[b][:2],
                     CONN_COLORS[idx % len(CONN_COLORS)], 3, cv2.LINE_AA)
    for i, (cx, cy, vis) in enumerate(pts):
        if vis:
            cv2.circle(img, (cx, cy), 6, JOINT_COLORS[i % len(JOINT_COLORS)], -1, cv2.LINE_AA)

    return img


# ── Frame annotation ───────────────────────────────────────────────────────────

def _annotate_frame(frame, persons_data):
    """
    Draw BBX + skeleton + label (#ID SAFE/UNSAFE conf%) for every tracked person.
    Returns (annotated_bgr, base64_jpeg).
    """
    annotated = frame.copy()
    h, w = annotated.shape[:2]

    for person in persons_data:
        is_unsafe  = person.get('class_id', 0) == 1
        color_bbx  = (0, 0, 220) if is_unsafe else (0, 200, 60)
        track_id   = person.get('track_id')
        id_prefix  = f"#{track_id} " if track_id is not None else ""
        label      = f"{id_prefix}{'UNSAFE' if is_unsafe else 'SAFE'} {person['confidence']:.0%}"
        bbox       = person.get('bbox', [])

        # Bounding box
        if len(bbox) == 4:
            x1, y1, x2, y2 = [int(v) for v in bbox]
            x1, x2 = max(0, min(w - 1, x1)), max(0, min(w - 1, x2))
            y1, y2 = max(0, min(h - 1, y1)), max(0, min(h - 1, y2))
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color_bbx, 2)

            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            ly = max(y1 - 4, th + 4)
            cv2.rectangle(annotated, (x1, ly - th - 4), (x1 + tw + 6, ly + 2), color_bbx, -1)
            cv2.putText(annotated, label, (x1 + 3, ly - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)

            # Score bar under bbox
            bar_w = x2 - x1
            if bar_w > 10:
                fill = int(bar_w * person['confidence'])
                cv2.rectangle(annotated, (x1, y2),     (x2,          y2 + 5), (40, 40, 40), -1)
                cv2.rectangle(annotated, (x1, y2),     (x1 + fill,   y2 + 5), color_bbx, -1)

        # Skeleton overlay on original frame
        kps_xy   = person.get('kps_xy', [])
        kps_conf = person.get('kps_conf', [])
        if len(kps_xy) >= 17:
            pts = []
            for i in range(17):
                cx   = int(kps_xy[i][0])
                cy   = int(kps_xy[i][1])
                conf = float(kps_conf[i]) if i < len(kps_conf) else 1.0
                pts.append((cx, cy, conf >= 0.3 and 0 < cx < w and 0 < cy < h))

            skel_color = (0, 0, 200) if is_unsafe else (0, 200, 50)
            for idx, (a, b) in enumerate(CONNECTIONS):
                if a < len(pts) and b < len(pts) and pts[a][2] and pts[b][2]:
                    cv2.line(annotated, pts[a][:2], pts[b][:2], skel_color, 2, cv2.LINE_AA)
            for i, (cx, cy, vis) in enumerate(pts):
                if vis:
                    cv2.circle(annotated, (cx, cy), 4,
                               JOINT_COLORS[i % len(JOINT_COLORS)], -1, cv2.LINE_AA)

    # Header bar
    unsafe_count = sum(1 for p in persons_data if p.get('class_id', 0) == 1)
    header     = f"POSTURE | {len(persons_data)} personne(s) | {unsafe_count} UNSAFE"
    hdr_color  = (0, 80, 220) if unsafe_count else (0, 200, 60)
    cv2.rectangle(annotated, (0, 0), (w, 32), (20, 20, 20), -1)
    cv2.putText(annotated, header, (10, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.60, hdr_color, 2, cv2.LINE_AA)

    ok, buf = cv2.imencode('.jpg', annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    b64 = base64.b64encode(buf.tobytes()).decode('utf-8') if ok else ''
    return annotated, b64


# ── Geometric fallback ─────────────────────────────────────────────────────────

def _geometric_posture_check(kpts_xy, kpts_conf):
    reasons, confidence = [], 0.0

    def get_point(idx):
        if idx >= len(kpts_xy) or (kpts_conf is not None and kpts_conf[idx] < 0.3):
            return None
        return kpts_xy[idx]

    shoulder_l = get_point(5); shoulder_r = get_point(6)
    hip_l = get_point(11);     hip_r = get_point(12)
    knee_l = get_point(13);    knee_r = get_point(14)
    ankle_l = get_point(15);   ankle_r = get_point(16)

    if not all([shoulder_l, shoulder_r, hip_l, hip_r]):
        return False, 0.0, reasons

    def dist(p1, p2):
        return np.linalg.norm(np.array(p1) - np.array(p2))

    shoulder_width = dist(shoulder_l, shoulder_r)

    if knee_l is not None and ankle_l is not None:
        if dist(knee_l, ankle_l) < dist(hip_l, knee_l) * 0.8:
            reasons.append("genou gauche plié"); confidence = max(confidence, 0.7)

    if knee_r is not None and ankle_r is not None:
        if dist(knee_r, ankle_r) < dist(hip_r, knee_r) * 0.8:
            reasons.append("genou droit plié"); confidence = max(confidence, 0.7)

    sc = ((shoulder_l[0] + shoulder_r[0]) / 2, (shoulder_l[1] + shoulder_r[1]) / 2)
    hc = ((hip_l[0] + hip_r[0]) / 2,           (hip_l[1] + hip_r[1]) / 2)
    if dist(sc, hc) > shoulder_width * 0.5:
        reasons.append("dos courbé"); confidence = max(confidence, 0.6)

    unsafe = len(reasons) > 0
    if unsafe and confidence == 0.0:
        confidence = 0.5
    return unsafe, confidence, reasons


# ── Main detection function ────────────────────────────────────────────────────

def detect_posture_in_frame(frame, camera='cam9'):
    """
    Multi-person posture detection with ByteTrack tracking.

    Returns (unsafe_detected: bool, confidence: float, details: dict).
      details['detections']      — one entry per tracked person
      details['annotated_frame'] — base64 JPEG with BBX + #ID + skeleton per person
      details['pose_backend']    — 'yolov8+bytetrack' | 'mediapipe' | 'none'
    """
    _load_models()
    _load_mediapipe()

    h, w = frame.shape[:2]
    details = {
        'camera'           : camera,
        'model'            : 'posture.pt + yolov8n-pose + bytetrack',
        'model_available'  : _posture_model is not None,
        'pose_available'   : _pose_model is not None or _mp_pose is not None,
        'detections'       : [],
        'skeleton_keypoints': [],
        'keypoints_conf'   : [],
        'annotated_frame'  : '',
        'processing_method': 'two_stage_pose_posture_tracked',
    }

    # ── Step 1: extract persons ────────────────────────────────────────────────
    # YOLOv8 + ByteTrack first (multi-person + IDs)
    persons_raw  = _yolo_persons_tracked(frame)
    pose_backend = 'yolov8+bytetrack' if persons_raw else ''

    # MediaPipe fallback (single person, no tracking)
    if not persons_raw:
        persons_raw  = _mediapipe_persons(frame)
        pose_backend = 'mediapipe' if persons_raw else 'none'

    details['pose_backend'] = pose_backend

    if not persons_raw:
        details['processing_method'] = 'no_person_detected'
        _, b64 = _annotate_frame(frame, [])
        details['annotated_frame'] = b64
        details['annotated_image'] = f'data:image/jpeg;base64,{b64}' if b64 else ''
        return False, 0.0, details

    # ── Step 2 & 3: classify each person with posture.pt ──────────────────────
    persons = []
    try:
        for person_raw in persons_raw:
            kps_xy   = person_raw['kps_xy']
            kps_conf = person_raw['kps_conf']
            bbox     = person_raw['bbox']
            track_id = person_raw.get('track_id')

            skel_img = _skeleton_on_black(kps_xy, kps_conf, w, h)
            cls_id, cls_conf = 0, 0.50
            predictions = []
            geom_unsafe, geom_conf, geom_reasons = False, 0.0, []

            if _posture_model is not None:
                pres = _posture_model(skel_img, conf=0.01, verbose=False)
                for pr in pres:
                    if pr.boxes and len(pr.boxes) > 0:
                        for box in pr.boxes:
                            predictions.append({
                                'class_id'  : int(box.cls),
                                'confidence': float(box.conf),
                            })
                        best    = max(pr.boxes, key=lambda b: float(b.conf))
                        cls_id  = int(best.cls)
                        cls_conf = float(best.conf)
                        break
                final_unsafe  = cls_id == 1
                final_conf    = cls_conf
                final_reasons = ['model_prediction'] if final_unsafe else []
            else:
                predictions = [{'error': 'model not loaded'}]
                geom_unsafe, geom_conf, geom_reasons = _geometric_posture_check(kps_xy, kps_conf)
                final_unsafe  = geom_unsafe
                final_conf    = geom_conf
                final_reasons = geom_reasons

            persons.append({
                'track_id'        : track_id,
                'bbox'            : bbox,
                'class'           : 'unsafe' if final_unsafe else 'safe',
                'class_id'        : 1 if final_unsafe else 0,
                'confidence'      : final_conf,
                'predictions'     : predictions,
                'geometric_unsafe': geom_unsafe,
                'geometric_conf'  : geom_conf,
                'geometric_reasons': geom_reasons,
                'skeleton_keypoints': kps_xy.tolist(),
                'keypoints_conf'  : kps_conf.tolist() if kps_conf is not None else [],
                'kps_xy'          : kps_xy,
                'kps_conf'        : kps_conf,
                'pose_source'     : person_raw.get('source', ''),
            })

        # ── Step 4: annotate frame (BBX + #ID + skeleton per person) ──────────
        _, b64 = _annotate_frame(frame, persons)
        details['annotated_frame'] = b64
        details['annotated_image'] = f'data:image/jpeg;base64,{b64}' if b64 else ''

        details['detections'] = [
            {k: v for k, v in p.items() if k not in ('kps_xy', 'kps_conf')}
            for p in persons
        ]

        if not persons:
            details['processing_method'] = 'no_person_detected'
            return False, 0.0, details

        unsafe_persons = [p for p in persons if p['class_id'] == 1]
        target = (max(unsafe_persons, key=lambda p: p['confidence'])
                  if unsafe_persons
                  else max(persons, key=lambda p: p['confidence']))

        details['bbox']                = target['bbox']
        details['skeleton_keypoints']  = target['skeleton_keypoints']
        details['keypoints_conf']      = target.get('keypoints_conf', [])
        details['classified_as']       = target['class']
        details['classification_conf'] = target['confidence']
        details['unsafe_count']        = len(unsafe_persons)
        details['total_persons']       = len(persons)
        details['reasons']             = target.get('geometric_reasons', [])
        details['tracked_ids']         = [p['track_id'] for p in persons
                                          if p['track_id'] is not None]

        if unsafe_persons:
            return True, target['confidence'], details
        return False, target['confidence'], details

    except Exception as e:
        details['error'] = str(e)
        details['processing_method'] = 'error'
        return False, 0.0, details
