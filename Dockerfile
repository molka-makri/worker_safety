FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=7860 \
    DJANGO_SETTINGS_MODULE=config.settings \
    DEBUG=0 \
    SERVE_MEDIA=1 \
    HF_MODEL_REPO_ID=molka8/worker_safety_models \
    YOLO_CONFIG_DIR=/tmp/Ultralytics \
    PRELOAD_DETECTORS_ON_STARTUP=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . .

RUN chmod +x /app/start.sh

EXPOSE 7860

CMD ["sh", "/app/start.sh"]
