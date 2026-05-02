# exit_detector.py — corrected version
import os
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EXIT_MODEL_PATH = os.path.abspath(
    os.path.join(BASE_DIR, "..", "models", "exit_emergency.pth")
)

# ── FIXED class mapping ───────────────────────────────────────────────────────
# Your COCO annotations use:
#   category_id 1 = Obstacle       → model label index 1
#   category_id 2 = emergency_exit → model label index 2
# Faster R-CNN keeps these as-is (background=0, then your classes)
CLASS_NAMES = {
    1: "obstacle",  # was swapped — now correct
    2: "exit",  # was swapped — now correct
}

# ── Lowered threshold — your dataset is small, model needs room ───────────────
CONF_THRESHOLD = 0.54  # was 0.45 — too strict for a small dataset
EXIT_PROXIMITY_SCALE = 0.55  # obstacle within 55% of exit diagonal = blocked
MIN_PROXIMITY_PIXELS = 30.0
TEMPORAL_HOLD_FRAMES = 6
CONFIDENCE_DECAY = 0.90
BBOX_SMOOTHING_ALPHA = 0.72

TORCHVISION_AVAILABLE = False
model = None
torch = None

try:
    import torch as _torch
    from torchvision.models.detection import (
        fasterrcnn_resnet50_fpn_v2,
        fasterrcnn_resnet50_fpn,
    )

    torch = _torch
    if os.path.exists(EXIT_MODEL_PATH):
        state_dict = _torch.load(EXIT_MODEL_PATH, map_location="cpu")

        # Try v2 first, fallback to v1
        for builder in [
            lambda: fasterrcnn_resnet50_fpn_v2(
                weights=None, weights_backbone=None, num_classes=3
            ),
            lambda: fasterrcnn_resnet50_fpn(
                weights=None, weights_backbone=None, num_classes=3
            ),
        ]:
            try:
                detector = builder()
                detector.load_state_dict(state_dict)
                detector.eval()
                model = detector
                TORCHVISION_AVAILABLE = True
                print(f"[ExitDetector] OK: model loaded from {EXIT_MODEL_PATH}")
                break
            except Exception as exc:
                print(f"[ExitDetector] builder failed: {exc}")

        if not TORCHVISION_AVAILABLE:
            print("[ExitDetector] ERROR: could not load weights into any architecture")
    else:
        print(f"[ExitDetector] WARNING: model not found: {EXIT_MODEL_PATH}")

except ImportError as exc:
    print(f"[ExitDetector] WARNING: torchvision not installed: {exc}")


# ── Utilities ─────────────────────────────────────────────────────────────────


def _smooth_bbox(previous_bbox, current_bbox):
    if not previous_bbox or not current_bbox:
        return current_bbox
    return [
        int(round(p * BBOX_SMOOTHING_ALPHA + c * (1 - BBOX_SMOOTHING_ALPHA)))
        for p, c in zip(previous_bbox, current_bbox)
    ]


def _edge_distance(a, b) -> float:
    """Minimum pixel distance between edges of two boxes."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    dx = max(ax1 - bx2, bx1 - ax2, 0)
    dy = max(ay1 - by2, by1 - ay2, 0)
    return float(np.hypot(dx, dy))


def _expanded_box(box, margin: float) -> List[int]:
    x1, y1, x2, y2 = box
    return [int(x1 - margin), int(y1 - margin), int(x2 + margin), int(y2 + margin)]


def _boxes_intersect(a, b) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


# ── Main detector class ───────────────────────────────────────────────────────


class ExitBlockedDetector:

    def __init__(self):
        self.model = model
        self.available = TORCHVISION_AVAILABLE
        self._states: Dict[str, Dict[str, Any]] = {}

    def detect_exit_blocked(
        self,
        frame: np.ndarray,
        camera: str = "cam6",
    ) -> Tuple[bool, float, Dict[str, Any]]:

        if not self.available or self.model is None or torch is None:
            return self._fallback_detection()

        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            tensor = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
            with torch.no_grad():
                outputs = self.model([tensor])[0]
            detected, confidence, details = self._parse_outputs(outputs, frame.shape)
            return self._apply_temporal_hold(camera, detected, confidence, details)

        except Exception as exc:
            print(f"[ExitDetector] Inference error: {exc}")
            return self._fallback_detection()

    def _parse_outputs(
        self,
        outputs,
        frame_shape,
    ) -> Tuple[bool, float, Dict[str, Any]]:

        boxes = outputs.get("boxes")
        labels = outputs.get("labels")
        scores = outputs.get("scores")

        exits = []
        obstacles = []
        detections = []

        if boxes is None or labels is None or scores is None:
            return False, 0.0, self._empty_details()

        for box_t, label_t, score_t in zip(boxes, labels, scores):
            score = float(score_t.item())
            cls_id = int(label_t.item())
            label = CLASS_NAMES.get(cls_id, f"class_{cls_id}")

            # Debug line — visible in Django terminal
            print(
                f"[ExitDetector] raw: cls={cls_id} label={label} "
                f"score={score:.3f}  (threshold={CONF_THRESHOLD})"
            )

            if score < CONF_THRESHOLD:
                continue

            bbox = [int(round(v)) for v in box_t.tolist()]
            payload = {
                "class_id": cls_id,
                "label": label,
                "confidence": round(score, 3),
                "bbox": bbox,
            }
            detections.append(payload)

            if label == "exit":
                exits.append(payload)
            elif label == "obstacle":
                obstacles.append(payload)

        print(
            f"[ExitDetector] exits={len(exits)}  obstacles={len(obstacles)}  "
            f"total_raw={len(list(zip(boxes, labels, scores)))}"
        )

        if not exits:
            d = self._empty_details()
            d["detections"] = detections
            return False, 0.0, d

        # ── Distance-based proximity check ────────────────────────────────────
        best_event = None
        best_score = 0.0

        for exit_det in exits:
            exit_box = exit_det["bbox"]
            exit_w = max(1.0, exit_box[2] - exit_box[0])
            exit_h = max(1.0, exit_box[3] - exit_box[1])
            exit_diag = float(np.hypot(exit_w, exit_h))
            threshold = max(MIN_PROXIMITY_PIXELS, EXIT_PROXIMITY_SCALE * exit_diag)
            search_box = _expanded_box(exit_box, threshold)

            nearest_obs = None
            nearest_dist = None

            for obs_det in obstacles:
                obs_box = obs_det["bbox"]
                distance = _edge_distance(exit_box, obs_box)
                in_zone = _boxes_intersect(search_box, obs_box)

                print(
                    f"[ExitDetector]   dist exit→obstacle: {distance:.1f}px  "
                    f"threshold: {threshold:.1f}px  in_zone={in_zone}"
                )

                if distance <= threshold or in_zone:
                    if nearest_obs is None or distance < nearest_dist:
                        nearest_obs = obs_det
                        nearest_dist = distance

            if nearest_obs is None:
                # Exit found but no nearby obstacle
                event_score = float(exit_det["confidence"]) * 0.20
                if event_score > best_score:
                    best_score = event_score
                    best_event = {
                        "blocked": False,
                        "confidence": event_score,
                        "exit": exit_det,
                        "obstacle": None,
                        "distance_pixels": None,
                        "distance_ratio": None,
                        "threshold_pixels": round(threshold, 2),
                    }
                continue

            # Obstacle is near — compute blocked confidence
            distance_ratio = nearest_dist / threshold if threshold > 0 else 1.0
            blocked_confidence = min(
                1.0,
                float(exit_det["confidence"]) * 0.55
                + float(nearest_obs["confidence"]) * 0.45
                + max(0.0, 1.0 - distance_ratio) * 0.35,
            )

            if blocked_confidence > best_score:
                best_score = blocked_confidence
                best_event = {
                    "blocked": True,
                    "confidence": blocked_confidence,
                    "exit": exit_det,
                    "obstacle": nearest_obs,
                    "distance_pixels": round(float(nearest_dist), 2),
                    "distance_ratio": round(float(distance_ratio), 4),
                    "threshold_pixels": round(threshold, 2),
                }

        if best_event is None:
            return False, 0.0, self._empty_details()

        details = {
            "model": "FRCNN_exit_emergency.pth",
            "processing_method": "fasterrcnn_distance_rule",
            "model_available": True,
            "detections": detections,
            "exit_bbox": best_event["exit"]["bbox"],
            "obstacle_bbox": (
                best_event["obstacle"]["bbox"] if best_event["obstacle"] else None
            ),
            "bbox": best_event["exit"]["bbox"],
            "blocked_exit": bool(best_event["blocked"]),
            "distance_pixels": best_event["distance_pixels"],
            "distance_ratio": best_event["distance_ratio"],
            "threshold_pixels": best_event["threshold_pixels"],
            "event_type": "blocked_exit" if best_event["blocked"] else "exit_clear",
        }

        is_blocked = bool(best_event["blocked"])
        return (
            is_blocked,
            float(best_event["confidence"]) if is_blocked else 0.0,
            details,
        )

    def _apply_temporal_hold(self, camera, detected, confidence, details):
        state = self._states.get(
            camera, {"misses": 0, "confidence": 0.0, "details": None}
        )

        if detected:
            prev = state.get("details") or {}
            details["exit_bbox"] = _smooth_bbox(
                prev.get("exit_bbox"), details.get("exit_bbox")
            )
            if details.get("obstacle_bbox"):
                details["obstacle_bbox"] = _smooth_bbox(
                    prev.get("obstacle_bbox"), details.get("obstacle_bbox")
                )
            details["bbox"] = details.get("exit_bbox")
            details["temporal_hold"] = False
            details["missed_frames"] = 0
            self._states[camera] = {
                "misses": 0,
                "confidence": confidence,
                "details": details,
            }
            return True, confidence, details

        last = state.get("details")
        misses = int(state.get("misses", 0)) + 1
        if last and misses <= TEMPORAL_HOLD_FRAMES:
            held = dict(last)
            held_conf = max(
                CONF_THRESHOLD,
                float(state.get("confidence", 0.0)) * (CONFIDENCE_DECAY**misses),
            )
            held.update(
                {
                    "temporal_hold": True,
                    "missed_frames": misses,
                    "processing_method": "temporal_hold",
                }
            )
            self._states[camera] = {
                "misses": misses,
                "confidence": state.get("confidence", held_conf),
                "details": last,
            }
            return True, min(held_conf, 1.0), held

        self._states[camera] = {"misses": 0, "confidence": 0.0, "details": None}
        details["temporal_hold"] = False
        details["missed_frames"] = misses
        return False, 0.0, details

    def _empty_details(self):
        return {
            "model": "FRCNN_exit_emergency.pth",
            "processing_method": "fasterrcnn_distance_rule",
            "model_available": True,
            "detections": [],
            "exit_bbox": None,
            "obstacle_bbox": None,
            "bbox": None,
            "blocked_exit": False,
            "distance_pixels": None,
            "distance_ratio": None,
            "threshold_pixels": None,
            "event_type": "exit_clear",
        }

    def _fallback_detection(self):
        return (
            False,
            0.0,
            {
                "model": "FRCNN_exit_emergency.pth",
                "processing_method": "model_unavailable",
                "model_available": False,
                "detections": [],
                "exit_bbox": None,
                "obstacle_bbox": None,
                "bbox": None,
                "blocked_exit": False,
                "distance_pixels": None,
                "distance_ratio": None,
                "threshold_pixels": None,
                "event_type": "exit_clear",
                "note": f"Model not found: {EXIT_MODEL_PATH}",
            },
        )


# ── Singleton ─────────────────────────────────────────────────────────────────
exit_blocked_detector = ExitBlockedDetector()


def detect_blocked_exit_in_frame(
    frame: np.ndarray,
    camera: str = "cam6",
) -> Tuple[bool, float, Dict[str, Any]]:
    return exit_blocked_detector.detect_exit_blocked(frame, camera=camera)


# ═══════════════════════════════════════════════════════════════════════════════
# STANDALONE TEST — run this directly to verify model before Django integration
#
# Usage:
#   python exit_detector.py
#   python exit_detector.py path/to/your/image.jpg
#   python exit_detector.py path/to/your/video.mp4
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys

    # Default test image
    test_path = (
        sys.argv[1]
        if len(sys.argv) > 1
        else os.path.join(
            BASE_DIR,
            "..",
            "media",
            "Blocked_Emergency_Exit-10-_jpg.rf.293524ab5a62900e646850a42a1e583f.jpg",
        )
    )

    if not os.path.exists(test_path):
        print(f"File not found: {test_path}")
        sys.exit(1)

    ext = os.path.splitext(test_path)[1].lower()
    is_video = ext in (".mp4", ".avi", ".mov", ".mkv", ".webm")

    print(f"\nTest path : {test_path}")
    print(f'Mode      : {"video" if is_video else "image"}')
    print(f"Model     : {EXIT_MODEL_PATH}")
    print(f"Loaded    : {TORCHVISION_AVAILABLE}")
    print("=" * 60)

    if is_video:
        cap = cv2.VideoCapture(test_path)
        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1
            if frame_idx % 5 != 0:
                continue

            blocked, conf, details = detect_blocked_exit_in_frame(frame, camera="test")
            status = "BLOCKED" if blocked else "clear"
            print(
                f"Frame {frame_idx:04d} → {status}  conf={conf:.3f}  "
                f'exits={len([d for d in details["detections"] if d["label"]=="exit"])}  '
                f'obstacles={len([d for d in details["detections"] if d["label"]=="obstacle"])}'
            )

            # Draw result
            out = frame.copy()
            for det in details["detections"]:
                x1, y1, x2, y2 = det["bbox"]
                color = (56, 142, 56) if det["label"] == "exit" else (0, 124, 245)
                cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    out,
                    f"{det['label']} {det['confidence']:.0%}",
                    (x1 + 3, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.52,
                    color,
                    1,
                )
            if blocked and details.get("exit_bbox"):
                x1, y1, x2, y2 = details["exit_bbox"]
                cv2.rectangle(out, (x1, y1), (x2, y2), (47, 47, 211), 4)
                cv2.putText(
                    out,
                    f"BLOCKED {details.get('distance_pixels', 0):.0f}px",
                    (x1 + 3, y2 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (47, 47, 211),
                    2,
                )

            cv2.imshow("Exit detector test", out)
            if cv2.waitKey(30) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()

    else:
        # Single image test
        frame = cv2.imread(test_path)
        if frame is None:
            print(f"Cannot read image: {test_path}")
            sys.exit(1)

        blocked, conf, details = detect_blocked_exit_in_frame(frame, camera="test")

        print(f'\nResult    : {"EXIT BLOCKED" if blocked else "exit clear"}')
        print(f"Confidence: {conf:.3f}")
        print(
            f'Exits     : {len([d for d in details["detections"] if d["label"] == "exit"])}'
        )
        print(
            f'Obstacles : {len([d for d in details["detections"] if d["label"] == "obstacle"])}'
        )
        if details.get("distance_pixels") is not None:
            print(
                f'Distance  : {details["distance_pixels"]:.1f}px  '
                f'(threshold={details["threshold_pixels"]:.1f}px)'
            )
        print(f'Event     : {details.get("event_type")}')
        print(f"\nAll detections:")
        for d in details["detections"]:
            print(
                f'  cls={d["class_id"]} label={d["label"]} '
                f'conf={d["confidence"]:.3f} bbox={d["bbox"]}'
            )

        # Draw and save result
        out = frame.copy()
        for det in details["detections"]:
            x1, y1, x2, y2 = det["bbox"]
            color = (56, 142, 56) if det["label"] == "exit" else (0, 124, 245)
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
            label_txt = f"{det['label']} {det['confidence']:.0%}"
            (tw, th), _ = cv2.getTextSize(label_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(out, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
            cv2.putText(
                out,
                label_txt,
                (x1 + 3, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                1,
            )

        if blocked and details.get("exit_bbox"):
            x1, y1, x2, y2 = details["exit_bbox"]
            cv2.rectangle(out, (x1, y1), (x2, y2), (47, 47, 211), 4)
            dist_txt = f"EXIT BLOCKED  {details.get('distance_pixels', 0):.0f}px"
            cv2.putText(
                out,
                dist_txt,
                (x1 + 3, y2 + 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (47, 47, 211),
                2,
            )

        status_txt = "EXIT BLOCKED" if blocked else "EXIT ACCESSIBLE"
        status_color = (47, 47, 211) if blocked else (56, 142, 56)
        cv2.putText(
            out, status_txt, (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 1.0, status_color, 2
        )

        # Save result next to input image
        out_path = os.path.splitext(test_path)[0] + "_exit_result.jpg"
        cv2.imwrite(out_path, out)
        print(f"\nResult image saved: {out_path}")

        cv2.imshow("Exit detector test", out)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
