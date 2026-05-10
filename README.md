# 🦺 Worker Safety Dashboard

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)
![Django](https://img.shields.io/badge/Django-4.x-092E20?style=flat&logo=django&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)
![YOLO](https://img.shields.io/badge/Ultralytics-YOLO-00FFFF?style=flat)
![Docker](https://img.shields.io/badge/Docker-2496ED?style=flat&logo=docker&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Spaces-FFD21E?style=flat&logo=huggingface&logoColor=black)
![License](https://img.shields.io/badge/License-MIT-green?style=flat)

A real-time **industrial worker safety monitoring platform** built with Django, PyTorch, Ultralytics YOLO, OpenCV, MediaPipe, and Docker. Centralizes multiple AI-powered safety modules in a single live dashboard, with support for both local execution and cloud deployment on Hugging Face Spaces.

> 🎓 Developed as part of the **AI Project** module at **Esprit School of Engineering** — *Private Higher School of Engineering and Technology*.

🔗 **Live Demo**: [View on Hugging Face Spaces](https://huggingface.co/spaces/molka8/worker-safety-dashboard) 

---

## 📌 Overview

This project was developed as part of the coursework for **AI Project**
at **Esprit School of Engineering**.

The platform aggregates over **13 AI safety detection modules** into a unified dashboard, enabling real-time monitoring of worker safety on industrial sites. It supports local execution as well as seamless deployment on Hugging Face Spaces via Docker.

**Keywords**: worker safety, AI surveillance, fall detection, PPE compliance, fire detection, industrial monitoring, deep learning, computer vision, YOLO, Django, Esprit School of Engineering

---

## ✨ Features

| Module | Description |
|---|---|
| 🤸 Fall Detection | Detects worker falls from surveillance video streams |
| 😴 Fatigue Detection | Monitors signs of worker fatigue in real time |
| 💧 Spill Segmentation | Identifies liquid spills and triggers hazard alerts |
| 🕳️ Manhole Detection | Detects open/closed manhole covers |
| 🚪 Emergency Exit Monitoring | Flags blocked emergency exit paths |
| 🦺 PPE Compliance | Verifies proper use of personal protective equipment |
| ⚠️ Safety Sign Inspection | Detects defects or absence of safety signage |
| 👷 Worker Tracking | Tracks and counts workers across camera zones |
| ⚙️ Proximity Detection | Alerts when workers are too close to machinery |
| 🧍 Posture Analysis | Detects unsafe postures with skeleton overlay |
| 😱 Panic Behavior | Identifies panic or distress behavior patterns |
| 🔥 Fire & Smoke Detection | Real-time fire and smoke detection |
| 📊 Live Dashboard | Centralized alerts, reports, overlays, and camera management |

---

## 🛠️ Tech Stack

### Frontend
- HTML5, CSS3, JavaScript

### Backend
- Python · Django · OpenCV · NumPy
- PyTorch · Torchvision · Ultralytics YOLO · MediaPipe

### Infrastructure & Tools
- Docker · Hugging Face Spaces · Hugging Face Hub
- SQLite · Gunicorn · WhiteNoise

---

## 📁 Directory Structure

```text
worker_safety/
├── app/
│   ├── fall_detector.py
│   ├── fatiguedetector.py
│   ├── spill_detector.py
│   ├── manhole_detector.py
│   ├── exit_detector.py
│   ├── proximity_detector.py
│   ├── posture_detector.py
│   ├── panic_detector.py
│   ├── fire_smoke_detector.py
│   ├── worker_tracking_detector.py
│   ├── views.py
│   ├── urls.py
│   ├── models.py
│   └── hf_model_store.py
├── config/
│   ├── settings.py
│   ├── urls.py
│   └── wsgi.py
├── static/
│   ├── css/
│   └── js/
├── templates/
│   └── safety_vision/
├── models/
├── media/
├── requirements.txt
├── Dockerfile
├── start.sh
├── push_to_hf_space.py
└── manage.py
```

---

## 🚀 Getting Started

### 1. Clone the project

```bash
git clone https://github.com/molka-makri/worker_safety.git
cd worker_safety
```

### 2. Create a virtual environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file at the root of the project:

```env
SECRET_KEY=change-me
DEBUG=1
ALLOWED_HOSTS=127.0.0.1,localhost
CSRF_TRUSTED_ORIGINS=http://127.0.0.1:8000,http://localhost:8000
HF_MODEL_REPO_ID=molka8/worker_safety_models
SERVE_MEDIA=1
```

### 5. AI Model Weights

The project expects model weights available either:
- **locally** inside `models/`, or
- **remotely** via Hugging Face Hub (`HF_MODEL_REPO_ID`)

Missing weights are auto-downloaded from `molka8/worker_safety_models`.

<details>
<summary>📦 Expected model files</summary>

```
fall_detection.pt        fatigue_detection.pt
spill_detection_model.pt manhole_seg.pt
exit_emergency.pth       proximity.pt
ppe.pt                   resnet50_router.pt
resnet50_E.pt            resnet50_F.pt
resnet50_P.pt            resnet50_M.pt
resnet50_W.pt            posture.pt
yolov8n-pose.pt          panic.pt
peopleNet.pt             fire_smoke_detection.pt
```

</details>

### 6. Apply migrations

```bash
python manage.py migrate
```

### 7. Run locally

```bash
python manage.py runserver
```

Open: [http://127.0.0.1:8000/](http://127.0.0.1:8000/)

### 8. (Optional) Create admin user

```bash
python manage.py createsuperuser
```

---

## 📝 Local Setup Notes

### Media & demo videos

The dashboard can serve media from:
- local files in `media/`, or
- proxied files from the Hugging Face dataset repo `molka8/worker_safety_media`

### After `git pull`

```bash
pip install -r requirements.txt   # 1. Install dependencies
# Set .env variables               # 2. Configure environment
python manage.py migrate           # 3. Apply migrations
python manage.py runserver         # 4. Start the server
```

> If Hugging Face weights are missing or corrupted, affected modules may fall back to a lighter heuristic mode when available.

---

## ☁️ Hugging Face Deployment

This project is packaged as a **Docker Space** on Hugging Face.

### Key files

| File | Role |
|---|---|
| `Dockerfile` | Container definition |
| `start.sh` | Startup script |
| `push_to_hf_space.py` | Deployment automation |

### Deploy steps

```bash
hf auth login
```

```bash
# PowerShell
$env:HF_SPACE_ID="your-username/worker-safety-dashboard"
python push_to_hf_space.py
```


---

## 🙏 Acknowledgments

This project was completed under the guidance of the teaching staff of **Esprit School of Engineering** as part of the **AI Project** module at **Esprit School of Engineering**.
---
We would like to thank our professors for their guidance, support, and valuable feedback throughout the sessions that made this project possible.
Special thanks to the open-source communities behind PyTorch, Ultralytics YOLO, OpenCV, MediaPipe, and Hugging Face for making this work possible.
