from ultralytics import YOLO
import os
import numpy as np

MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "fire_smoke_detection.pt")
model = YOLO(MODEL_PATH)

def run_fire_smoke_detection(frame: np.ndarray):
    """Accepts an OpenCV BGR frame and returns fire/smoke detection summary."""
    results = model(frame)
    detections = []

    for r in results:
        for box in r.boxes:
            label = model.names[int(box.cls)]
            confidence = float(box.conf)
            detections.append({
                "label": label,
                "confidence": round(confidence * 100, 2)
            })

    summary = {
        "fire_detected": any(d["label"] == "fire" for d in detections),
        "smoke_detected": any(d["label"] == "smoke" for d in detections),
        "total_detections": len(detections),
        "details": detections
    }
    return summary