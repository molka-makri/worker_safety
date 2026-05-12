#!/bin/sh
set -e

PRELOAD_DETECTORS_ON_STARTUP=0 python manage.py migrate --noinput
PRELOAD_DETECTORS_ON_STARTUP=0 python manage.py collectstatic --noinput

export PRELOAD_DETECTORS_ON_STARTUP="${PRELOAD_DETECTORS_ON_STARTUP:-1}"
exec gunicorn config.wsgi:application \
  --bind 0.0.0.0:${PORT:-7860} \
  --workers ${GUNICORN_WORKERS:-1} \
  --threads ${GUNICORN_THREADS:-4} \
  --timeout ${GUNICORN_TIMEOUT:-300}
