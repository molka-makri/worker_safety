---
title: Worker Safety Dashboard
emoji: 🦺
colorFrom: red
colorTo: yellow
sdk: docker
app_port: 7860
pinned: false
---

# Nom du Projet

Worker Safety Dashboard

A real-time industrial safety monitoring platform built with Django, JavaScript, PyTorch, Torchvision, Ultralytics YOLO, OpenCV, NumPy, MediaPipe, Docker, and Hugging Face Spaces.

## Overview

This project was developed as part of the coursework for **AI Project**  
at **Private Higher School of Engineering and Technology - Esprit School of Engineering**.

The application centralizes multiple AI safety modules in one dashboard and supports local execution as well as deployment on Hugging Face Spaces.

## Features

- Fall detection from surveillance video streams
- Fatigue detection for workers
- Spill segmentation and hazard alerting
- Manhole open/closed detection
- Emergency exit blockage detection
- PPE compliance detection
- Safety sign defect detection
- Worker tracking and counting
- Worker-machine proximity detection
- Unsafe posture detection with skeleton overlay
- Panic behavior detection
- Fire and smoke detection
- Live dashboard with alerts, reports, overlays, and camera management

## Tech Stack

### Frontend
- HTML5
- CSS3
- JavaScript

### Backend
- Python
- Django
- OpenCV
- NumPy
- PyTorch
- Torchvision
- Ultralytics YOLO
- MediaPipe

### Other Tools
- Docker
- Hugging Face Spaces
- Hugging Face Hub
- SQLite
- Gunicorn
- WhiteNoise

## Directory Structure

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

## Getting Started

### 1. Clone the project

```bash
git clone <your-repository-url>
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

Create a `.env` file or define environment variables manually.

Example:

```env
SECRET_KEY=change-me
DEBUG=1
ALLOWED_HOSTS=127.0.0.1,localhost
CSRF_TRUSTED_ORIGINS=http://127.0.0.1:8000,http://localhost:8000
HF_MODEL_REPO_ID=molka8/worker_safety_models
SERVE_MEDIA=1
```

### 5. Provide the models

This project expects AI weights to be available either:

- locally inside `models/`, or
- remotely from Hugging Face through `HF_MODEL_REPO_ID`

The current deployment logic auto-downloads missing weights from:

```text
molka8/worker_safety_models
```

Important expected files include:

- `fall_detection.pt`
- `fatigue_detection.pt`
- `spill_detection_model.pt`
- `manhole_seg.pt`
- `exit_emergency.pth`
- `proximity.pt`
- `ppe.pt`
- `resnet50_router.pt`
- `resnet50_E.pt`
- `resnet50_F.pt`
- `resnet50_P.pt`
- `resnet50_M.pt`
- `resnet50_W.pt`
- `posture.pt`
- `yolov8n-pose.pt`
- `panic.pt`
- `peopleNet.pt`
- `fire_smoke_detection.pt`

### 6. Apply migrations

```bash
python manage.py migrate
```

### 7. Run the project locally

```bash
python manage.py runserver
```

Open:

- `http://127.0.0.1:8000/`

### 8. Optional admin user

```bash
python manage.py createsuperuser
```

## Local Setup Notes

### Media and demo videos

The live dashboard can use:

- local files from `media/`, or
- proxied remote files from the Hugging Face dataset repo

Current hosted demo media is expected from:

```text
molka8/worker_safety_media
```

### If someone pulls the project later

To run the project successfully after `git pull`, they should:

1. install dependencies from `requirements.txt`
2. set the `.env` variables
3. make sure the Hugging Face model repo is accessible, or copy the model files into `models/`
4. run `python manage.py migrate`
5. start the server with `python manage.py runserver`

If Hugging Face weights are missing or corrupted, the affected module may fall back to a lighter heuristic mode when available.

## Hugging Face Deployment

### Space

This project is prepared for a **Docker Space**.

Main files:

- `Dockerfile`
- `start.sh`
- `push_to_hf_space.py`

### Deploy steps

```bash
hf auth login
```

```bash
# PowerShell
$env:HF_SPACE_ID="your-username/worker-safety-dashboard"
python push_to_hf_space.py
```

Recommended Space variables/secrets:

- `SECRET_KEY`
- `DEBUG=0`
- `ALLOWED_HOSTS=*`
- `CSRF_TRUSTED_ORIGINS=https://*.hf.space,https://huggingface.co`
- `SERVE_MEDIA=1`
- `HF_MODEL_REPO_ID=molka8/worker_safety_models`

## Detector Notes

### Posture

- Uses `posture.pt` for safe/unsafe classification
- Uses `yolov8n-pose.pt` first for pose extraction
- Falls back to MediaPipe when YOLO pose is unavailable

### Panic

- Uses `panic.pt` as the primary BiLSTM classifier
- Uses pose extraction from YOLO pose first
- Falls back to MediaPipe-based heuristic behavior scoring if the main classifier or pose backend is unavailable

### Proximity

- Uses `proximity.pt`
- Now resolves worker and machine classes from model labels, not only from hard-coded class IDs

## Acknowledgments

This project was completed under the guidance of the teaching staff of **Esprit School of Engineering** as part of the **AI Project** module at **Private Higher School of Engineering and Technology - Esprit School of Engineering**.
