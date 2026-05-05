import math
import os
from typing import Any, Dict, List, Set, Tuple

import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PEOPLE_MODEL_PATH = os.path.abspath(os.path.join(BASE_DIR, "..", "models", "peopleNet.pt"))

CONF_THRESHOLD = 0.2
IOU_THRESHOLD = 0.45
STALE_TRACK_FRAMES = 60
LINE_START = (195, 21)
LINE_END = (426, 256)
TRACKING_IMAGE_SIZE = 640
TRACKING_MAX_DET = 48
TRACKER_CONFIG_PATH = os.path.abspath(os.path.join(BASE_DIR, "worker_tracking_bytetrack.yaml"))
ID_STITCH_MAX_GAP_FRAMES = 18
ID_STITCH_MIN_IOU = 0.05
ID_STITCH_DISTANCE_SCALE = 0.75
ID_STITCH_MIN_PIXEL_DISTANCE = 42.0

YOLO_AVAILABLE = False
yolo_model = None
PERSON_CLASS_IDS: List[int] = []

try:
    from ultralytics import YOLO

    if os.path.exists(PEOPLE_MODEL_PATH):
        yolo_model = YOLO(PEOPLE_MODEL_PATH)
        YOLO_AVAILABLE = True
        print(f"[WorkerTrackingDetector] OK: Model loaded: {PEOPLE_MODEL_PATH}")
        print("[WorkerTrackingDetector] Model classes:")
        for cls_id, cls_name in yolo_model.names.items():
            print(f"   class {cls_id} -> '{cls_name}'")

        person_ids = []
        for cls_id, cls_name in yolo_model.names.items():
            cls = str(cls_name).lower()
            if "person" in cls or "worker" in cls or "human" in cls:
                person_ids.append(int(cls_id))
        if person_ids:
            PERSON_CLASS_IDS = person_ids
            print(f"[WorkerTrackingDetector] Person-like class IDs: {PERSON_CLASS_IDS}")
        else:
            print("[WorkerTrackingDetector] WARNING: No person-like classes found, class filter disabled")
    else:
        print(f"[WorkerTrackingDetector] WARNING: Model not found: {PEOPLE_MODEL_PATH}")
except ImportError as exc:
    print(f"[WorkerTrackingDetector] WARNING: ultralytics not installed: {exc}")


def get_centroid(bbox: List[float]) -> Tuple[int, int]:
    """
    Get the centroid of a bbox.
    bbox = [x1, y1, x2, y2]
    """
    return int((bbox[0] + bbox[2]) / 2), int((bbox[1] + bbox[3]) / 2)


def side_of_line(
    point: Tuple[int, int],
    line_start: Tuple[int, int],
    line_end: Tuple[int, int],
) -> int:
    """
    Returns +1 or -1 depending on which side of the line the point is on.
    Uses the cross product of the line vector and the point vector.
    Works for any line orientation, not just horizontal.
    """
    dx = line_end[0] - line_start[0]
    dy = line_end[1] - line_start[1]
    px = point[0] - line_start[0]
    py = point[1] - line_start[1]
    cross = dx * py - dy * px
    return 1 if cross >= 0 else -1


class WorkerTrackingDetector:
    def __init__(self) -> None:
        self.model = yolo_model
        self.available = YOLO_AVAILABLE
        self._states: Dict[str, Dict[str, Any]] = {}

    @staticmethod
    def _new_state() -> Dict[str, Any]:
        return {
            "track_history": {},   # track_id -> last known side of line
            "counted_ids": set(),  # track_ids that have already crossed
            "count_in": 0,
            "count_out": 0,
            "frame_idx": 0,
            "last_seen": {},       # track_id -> frame_idx (for stale cleanup)
            "raw_to_stable": {},   # tracker raw id -> stable id (ID stitching)
            "raw_last_seen": {},   # raw tracker id -> frame_idx
            "stable_tracks": {},   # stable id -> bbox/centroid/last_seen
            "next_stable_id": 1,
        }

    def _get_state(self, camera: str) -> Dict[str, Any]:
        if camera not in self._states:
            self._states[camera] = self._new_state()
        return self._states[camera]

    def reset_camera(self, camera: str) -> None:
        """Reset per-camera crossing state AND YOLO's internal ByteTrack state."""
        self._states[camera] = self._new_state()
        self._reset_yolo_tracker()

    def _reset_yolo_tracker(self) -> None:
        """
        Properly reset YOLO's internal tracker so IDs restart from 1.
        Ultralytics stores per-predictor trackers; resetting them avoids
        ghost tracks carrying over after a camera switch or reset.
        """
        if self.model is None:
            return
        predictor = getattr(self.model, "predictor", None)
        if predictor is None:
            return
        trackers = getattr(predictor, "trackers", None)
        if trackers:
            for tracker in trackers:
                if hasattr(tracker, "reset"):
                    tracker.reset()
        # Also clear the results cache so persist works cleanly
        if hasattr(predictor, "results"):
            predictor.results = None

    @staticmethod
    def _bbox_iou(box_a: List[float], box_b: List[float]) -> float:
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b
        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)
        inter_w = max(0.0, inter_x2 - inter_x1)
        inter_h = max(0.0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h
        if inter_area <= 0.0:
            return 0.0
        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        denom = area_a + area_b - inter_area
        if denom <= 0.0:
            return 0.0
        return inter_area / denom

    def _resolve_stable_track_id(
        self,
        state: Dict[str, Any],
        raw_track_id: int,
        bbox: List[float],
        frame_idx: int,
        used_stable_ids: Set[int],
    ) -> int:
        raw_to_stable: Dict[int, int] = state["raw_to_stable"]
        stable_tracks: Dict[int, Dict[str, Any]] = state["stable_tracks"]

        mapped_stable_id = raw_to_stable.get(raw_track_id)
        if (
            mapped_stable_id is not None
            and mapped_stable_id in stable_tracks
            and mapped_stable_id not in used_stable_ids
        ):
            return mapped_stable_id

        cx, cy = get_centroid(bbox)
        width = max(1.0, bbox[2] - bbox[0])
        height = max(1.0, bbox[3] - bbox[1])
        dynamic_max_distance = max(
            ID_STITCH_MIN_PIXEL_DISTANCE,
            math.hypot(width, height) * ID_STITCH_DISTANCE_SCALE,
        )

        best_stable_id = None
        best_score = float("inf")
        for stable_id, stable_info in stable_tracks.items():
            if stable_id in used_stable_ids:
                continue
            if frame_idx - int(stable_info.get("last_seen", 0)) > ID_STITCH_MAX_GAP_FRAMES:
                continue

            prev_cx, prev_cy = stable_info.get("centroid", (cx, cy))
            distance = math.hypot(float(cx - prev_cx), float(cy - prev_cy))
            if distance > dynamic_max_distance:
                continue

            prev_bbox = stable_info.get("bbox")
            iou = self._bbox_iou(bbox, prev_bbox) if prev_bbox else 0.0
            if iou < ID_STITCH_MIN_IOU and distance > dynamic_max_distance * 0.5:
                continue

            # Favor nearby detections while using IoU as a tie-breaker.
            score = distance - (iou * 60.0)
            if score < best_score:
                best_score = score
                best_stable_id = stable_id

        if best_stable_id is None:
            best_stable_id = int(state["next_stable_id"])
            state["next_stable_id"] = best_stable_id + 1

        raw_to_stable[raw_track_id] = best_stable_id
        return best_stable_id

    def detect_workers(
        self,
        frame: np.ndarray,
        camera: str = "cam8",
        reset: bool = False,
    ) -> Tuple[bool, float, Dict[str, Any]]:
        if not self.available or self.model is None:
            note = f"Model not found: {PEOPLE_MODEL_PATH}" if not YOLO_AVAILABLE else "Tracking unavailable"
            return False, 0.0, {
                "model": "YOLO_peopleNet.pt",
                "model_available": False,
                "tracker_available": False,
                "note": note,
            }

        if reset:
            self.reset_camera(camera)

        state = self._get_state(camera)
        state["frame_idx"] += 1
        frame_idx = int(state["frame_idx"])

        line_start = LINE_START
        line_end = LINE_END

        # ------------------------------------------------------------------
        # KEY FIX: use model.track(persist=True) instead of model.predict().
        # persist=True tells Ultralytics to keep ByteTrack state between calls
        # so the same physical person keeps the same track ID across frames.
        # ------------------------------------------------------------------
        try:
            result = self.model.track(
                frame,
                conf=CONF_THRESHOLD,
                iou=IOU_THRESHOLD,
                classes=PERSON_CLASS_IDS if PERSON_CLASS_IDS else None,
                verbose=False,
                imgsz=TRACKING_IMAGE_SIZE,
                max_det=TRACKING_MAX_DET,
                persist=True,          # <-- maintains tracker state between frames
                tracker=TRACKER_CONFIG_PATH if os.path.exists(TRACKER_CONFIG_PATH) else "bytetrack.yaml",
            )[0]
        except Exception as exc:
            print(f"[WorkerTrackingDetector] YOLO tracking error: {exc}")
            return False, 0.0, {"model_available": True, "tracker_available": False, "error": str(exc)}

        tracks_output: List[Dict[str, Any]] = []
        crossed_ids: List[int] = []
        best_confidence = 0.0
        assigned_stable_ids: Set[int] = set()

        boxes = getattr(result, "boxes", None)
        if boxes is not None and len(boxes) > 0:
            for box in boxes:
                # ByteTrack assigns .id; it may be None for unconfirmed tracks
                if box.id is None:
                    continue

                raw_track_id = int(box.id[0])
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                confidence = float(box.conf[0]) if getattr(box, "conf", None) is not None else 0.0

                ltrb = [float(x1), float(y1), float(x2), float(y2)]
                width = max(0.0, x2 - x1)
                height = max(0.0, y2 - y1)
                if width <= 0.0 or height <= 0.0:
                    continue

                stable_track_id = self._resolve_stable_track_id(
                    state=state,
                    raw_track_id=raw_track_id,
                    bbox=ltrb,
                    frame_idx=frame_idx,
                    used_stable_ids=assigned_stable_ids,
                )
                assigned_stable_ids.add(stable_track_id)

                if confidence > best_confidence:
                    best_confidence = confidence

                cx, cy = get_centroid(ltrb)
                current_side = side_of_line((cx, cy), line_start, line_end)

                state["last_seen"][stable_track_id] = frame_idx
                state["raw_last_seen"][raw_track_id] = frame_idx
                state["stable_tracks"][stable_track_id] = {
                    "bbox": ltrb,
                    "centroid": (cx, cy),
                    "last_seen": frame_idx,
                }

                # ── Line-crossing logic ────────────────────────────────────
                was_counted = stable_track_id in state["counted_ids"]
                crossed = False
                direction = None

                if stable_track_id in state["track_history"]:
                    prev_side = state["track_history"][stable_track_id]
                    if prev_side != current_side and stable_track_id not in state["counted_ids"]:
                        state["counted_ids"].add(stable_track_id)
                        crossed = True
                        crossed_ids.append(stable_track_id)
                        if prev_side == -1 and current_side == 1:
                            state["count_in"] += 1
                            direction = "in"
                        elif prev_side == 1 and current_side == -1:
                            state["count_out"] += 1
                            direction = "out"

                state["track_history"][stable_track_id] = current_side

                tracks_output.append({
                    "track_id": stable_track_id,
                    "raw_track_id": raw_track_id,
                    "bbox": [round(v, 2) for v in ltrb],
                    "confidence": round(confidence, 4),
                    "centroid": [cx, cy],
                    "side": current_side,
                    "counted": was_counted or crossed,
                    "crossed": crossed,
                    "direction": direction,
                })

        # ── Prune stale tracks so memory doesn't grow unboundedly ──────────
        stale_ids = [
            tid
            for tid, last_f in list(state["last_seen"].items())
            if frame_idx - last_f > STALE_TRACK_FRAMES
        ]
        for tid in stale_ids:
            state["last_seen"].pop(tid, None)
            state["track_history"].pop(tid, None)
            state["stable_tracks"].pop(tid, None)
            # NOTE: intentionally do NOT remove from counted_ids so a person
            # who re-enters after being pruned is counted fresh.

        stale_raw_ids = [
            raw_id
            for raw_id, last_f in list(state["raw_last_seen"].items())
            if frame_idx - last_f > ID_STITCH_MAX_GAP_FRAMES
        ]
        for raw_id in stale_raw_ids:
            state["raw_last_seen"].pop(raw_id, None)
            state["raw_to_stable"].pop(raw_id, None)

        invalid_links = [
            raw_id
            for raw_id, stable_id in list(state["raw_to_stable"].items())
            if stable_id not in state["last_seen"]
        ]
        for raw_id in invalid_links:
            state["raw_to_stable"].pop(raw_id, None)

        total_crossings = int(state["count_in"] + state["count_out"])
        crossing_detected = len(crossed_ids) > 0

        details = {
            "model": "peopleNet.pt",
            "processing_method": "yolo_bytetrack_line_crossing",
            "model_available": True,
            "tracker_available": True,
            "camera": camera,
            "line_start": list(line_start),
            "line_end": list(line_end),
            "track_count": len(tracks_output),
            "person_class_ids": PERSON_CLASS_IDS,
            "class_filter_enabled": bool(PERSON_CLASS_IDS),
            "tracking_backend": "bytetrack",
            "tracks": tracks_output,
            "crossed_ids": crossed_ids,
            "crossed_count": len(crossed_ids),
            "count_in": int(state["count_in"]),
            "count_out": int(state["count_out"]),
            "total_crossings": total_crossings,
            "frame_idx": frame_idx,
        }

        return crossing_detected, min(best_confidence, 1.0), details


worker_tracking_detector = WorkerTrackingDetector()


def detect_worker_tracking_in_frame(
    frame: np.ndarray,
    camera: str = "cam8",
    reset: bool = False,
) -> Tuple[bool, float, Dict[str, Any]]:
    return worker_tracking_detector.detect_workers(frame, camera=camera, reset=reset)
