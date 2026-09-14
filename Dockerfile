# Qorğan demo — FastAPI site (landing + live call + analyst dashboard) in one container.
# This server never accepts audio (PLAN_2026-09 §2); ASR belongs on the device.
# Model, corpus, and demo seeds are baked at BUILD time by
# scripts/deploy_bootstrap.py, so cold starts are instant.

FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    QORGAN_CLASSIFIER_BACKEND=linear

# libgomp1: required by ctranslate2 (faster-whisper) and hdbscan at import time.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# CPU-only torch FIRST — the default PyPI build would pull multi-GB CUDA wheels.
RUN pip install "torch>=2.2" --index-url https://download.pytorch.org/whl/cpu

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install -e .

COPY . .

# Bake corpus + model + Level-2 seeds + Vosk models into the image.
RUN python scripts/deploy_bootstrap.py

EXPOSE 8000
# Bootstrap re-run is a fast no-op when baked; single worker on purpose — live-call
# sessions are in-process state (do not scale replicas past 1).
CMD ["/bin/sh", "-c", "python scripts/deploy_bootstrap.py && exec uvicorn qorgan.api:app --host 0.0.0.0 --port ${PORT:-8000}"]
