# Qorğan demo — FastAPI site (landing + live call + analyst dashboard) in one container.
# This server never accepts audio (PLAN_2026-09 §2); ASR belongs on the device.
# Model, corpus, and demo seeds are baked at BUILD time by
# scripts/deploy_bootstrap.py, so cold starts are instant.

FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    QORGAN_CLASSIFIER_BACKEND=linear

# libgomp1: OpenMP runtime, a tiny safety net for scikit-learn / onnxruntime wheels.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Runtime dependencies only: int8 ONNX embedder + sklearn heads + FastAPI. torch,
# transformers, Streamlit and Gemini are optional extras (pyproject.toml) the served product
# never imports -- tests/test_runtime_deps.py keeps it that way.
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install -e .

COPY . .

# Bake corpus + model into the image. Level-2 seeds are created on first start, because
# their caller numbers are HMAC-hashed with QORGAN_NUMBER_HMAC_KEY (a runtime secret that
# must never be baked into the image) -- pass it with `docker run -e QORGAN_NUMBER_HMAC_KEY=...`.
# Runtime secrets, never baked (see .env.example): QORGAN_NUMBER_HMAC_KEY (number linking),
# QORGAN_AUDIT_CHAIN_KEY (tamper-evident audit log; without it /api/admin and /api/v1 are
# closed), QORGAN_ANALYST_KEYS (analyst console; without it /api/admin answers 503) and,
# for partners, QORGAN_PARTNER_API_KEYS.
RUN python scripts/deploy_bootstrap.py

EXPOSE 8000
# Bootstrap re-run is a fast no-op when baked; single worker on purpose — live-call
# sessions are in-process state (do not scale replicas past 1).
CMD ["/bin/sh", "-c", "python scripts/deploy_bootstrap.py && exec uvicorn qorgan.api:app --host 0.0.0.0 --port ${PORT:-8000}"]
