import os
import threading

from .hf_model_store import ensure_model_file


_warmup_started = False
_warmup_lock = threading.Lock()


def _warmup_models() -> None:
    print("[StartupWarmup] Starting safe model prefetch...")

    tasks = [
        ("fall_detection.pt", lambda: ensure_model_file("fall_detection.pt")),
        ("fatigue_detection.pt", lambda: ensure_model_file("fatigue_detection.pt")),
        ("spill_detection_model.pt", lambda: ensure_model_file("spill_detection_model.pt")),
        ("manhole_seg.pt", lambda: ensure_model_file("manhole_seg.pt")),
        ("exit_emergency.pth", lambda: ensure_model_file("exit_emergency.pth")),
        ("proximity.pt", lambda: ensure_model_file("proximity.pt")),
        ("ppe.pt", lambda: ensure_model_file("ppe.pt")),
        ("resnet50_router.pt", lambda: ensure_model_file("resnet50_router.pt")),
        ("resnet50_E.pt", lambda: ensure_model_file("resnet50_E.pt")),
        ("resnet50_F.pt", lambda: ensure_model_file("resnet50_F.pt")),
        ("resnet50_P.pt", lambda: ensure_model_file("resnet50_P.pt")),
        ("resnet50_M.pt", lambda: ensure_model_file("resnet50_M.pt")),
        ("resnet50_W.pt", lambda: ensure_model_file("resnet50_W.pt")),
        ("posture.pt", lambda: ensure_model_file("posture.pt")),
        ("yolov8n-pose.pt", lambda: ensure_model_file("yolov8n-pose.pt")),
        ("panic.pt", lambda: ensure_model_file("panic.pt")),
        ("peopleNet.pt", lambda: ensure_model_file("peopleNet.pt")),
        ("fire_smoke_detection.pt", lambda: ensure_model_file("fire_smoke_detection.pt")),
    ]

    for name, task in tasks:
        try:
            task()
            print(f"[StartupWarmup] OK: {name}")
        except Exception as exc:
            print(f"[StartupWarmup] WARNING: {name} warmup failed: {exc}")

    print("[StartupWarmup] Safe prefetch finished.")


def start_background_warmup() -> None:
    global _warmup_started
    if os.getenv("PRELOAD_DETECTORS_ON_STARTUP", "1").strip().lower() not in {"1", "true", "yes", "on"}:
        return

    with _warmup_lock:
        if _warmup_started:
            return
        _warmup_started = True

    thread = threading.Thread(target=_warmup_models, name="startup-warmup", daemon=True)
    thread.start()
