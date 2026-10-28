# Qorğan -- the FastAPI site (landing, live call, analyst console) in one container.
# Run it with docker compose (compose.yaml); operations are in docs/DEPLOY.md.
#
# - This server never accepts audio (PLAN_2026-09 §2); speech recognition runs on the device.
# - Runtime dependencies only (ADR D48): int8 ONNX embedder + sklearn heads + FastAPI, no torch.
# - The server scores with exactly the head weights the browser ships (QORGAN_LINEAR_WEIGHTS=web,
#   ADR D55): nothing is downloaded from the Hub model repo or retrained, at build or at start.
# - No secret is needed to build and none is baked in. Runtime secrets (.env.example), passed
#   as an env file at run time: QORGAN_NUMBER_HMAC_KEY (number linking; the Level-2 demo seeds
#   are created on the FIRST START because their numbers are hashed with it),
#   QORGAN_AUDIT_CHAIN_KEY + QORGAN_ANALYST_KEYS (analyst console; without them /api/admin
#   answers 503) and, for partners, QORGAN_PARTNER_API_KEYS.
# - ONE worker, one replica, on purpose: rate limits, the investigator's open budget and the
#   retention purge schedule are in-process state. Never scale past 1 on the same data volume.
# - Portable to managed builders (Railway, docs/DEPLOY.md §8): no BuildKit cache mounts
#   (Railway refuses a cache mount whose id lacks its service prefix), no `COPY --exclude`
#   (unknown to older Dockerfile frontends), no `HEALTHCHECK --start-interval`.

ARG PYTHON_IMAGE=python:3.11-slim

# --- requirements: the dependency list alone, so the install layer is keyed on nothing else --
FROM ${PYTHON_IMAGE} AS requirements
COPY pyproject.toml /tmp/pyproject.toml
RUN python -c "import tomllib; deps = tomllib.load(open('/tmp/pyproject.toml', 'rb'))['project']['dependencies']; open('/tmp/requirements.txt', 'w').write(chr(10).join(deps) + chr(10))"

# --- deps: runtime packages at the tested versions (scripts/deploy_constraints.txt) ----------
FROM ${PYTHON_IMAGE} AS deps
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_ROOT_USER_ACTION=ignore
# libgomp1: OpenMP runtime for the scikit-learn / onnxruntime wheels.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 \
 && rm -rf /var/lib/apt/lists/*
COPY --from=requirements /tmp/requirements.txt /tmp/requirements.txt
COPY scripts/deploy_constraints.txt /tmp/constraints.txt
RUN pip install --requirement /tmp/requirements.txt --constraint /tmp/constraints.txt \
 && pip check \
 && rm /tmp/requirements.txt /tmp/constraints.txt

# --- corpus: the published, scrubbed splits (HF dataset sanzh-ts/govtech_ds, ~5 MB) ---------
# Cached independently of the code. To re-fetch after the dataset is re-published:
#   docker compose build --build-arg QORGAN_CORPUS_REFRESH=$(date +%s)
FROM deps AS corpus
ARG QORGAN_CORPUS_REFRESH=
WORKDIR /app
COPY scripts/deploy_bootstrap.py scripts/deploy_bootstrap.py
RUN python -c "import importlib.util as u; s = u.spec_from_file_location('bootstrap', 'scripts/deploy_bootstrap.py'); m = u.module_from_spec(s); s.loader.exec_module(m); m.ensure_corpus()"

# --- runtime ---------------------------------------------------------------------------------
FROM deps AS runtime
ENV QORGAN_CLASSIFIER_BACKEND=linear \
    QORGAN_LINEAR_WEIGHTS=web \
    QORGAN_EMBED_BACKEND=onnx \
    QORGAN_DATA_DIR=/app/data \
    QORGAN_MODEL_DIR=/app/models
RUN groupadd --system --gid 10001 qorgan \
 && useradd --system --uid 10001 --gid qorgan --home-dir /home/qorgan --create-home --shell /usr/sbin/nologin qorgan
WORKDIR /app

# The published corpus: in data/processed (a new data volume starts from this copy) and a
# pristine copy the entrypoint refreshes the volume from on every start.
COPY --chown=10001:10001 --from=corpus /app/data/processed ./data/processed
COPY --from=corpus /app/data/processed /opt/qorgan/corpus
# Code, configuration and the site (pages, core JS, committed head weights, and the embedder /
# speech models when the build context has them -- otherwise the bootstrap below fetches them).
COPY pyproject.toml README.md ./
COPY configs ./configs
COPY data ./data
COPY scripts ./scripts
COPY src ./src
COPY site ./site
RUN pip install --no-deps --no-build-isolation --editable . \
 && python -m compileall -q src scripts

# Build-time provisioning, no secret involved: the dialogue pool from the corpus, the embedder
# and speech models checked (downloaded only when the clone lacks them; the
# download caches are dropped in the same layer so the image holds each model once), the linear
# backend probed on the committed browser weights.
# The GigaAM browser runtime (ADR D77/D79): fetched here, at build time, from our commit-pinned HF
# mirror and sha256-checked, so it must be a build ARG (a platform's runtime variable never
# reaches this step). Override with --build-arg; empty = no runtime (the page uses the Vosk vote).
ARG QORGAN_SHERPA_WASM_URL=https://huggingface.co/sanzh-ts/govtech/resolve/f3297be1a042b7c6ed86f6c4d88ad563eadfea42/sherpa-onnx/1.13.8
RUN python scripts/deploy_bootstrap.py \
 && rm -rf /root/.cache/huggingface /root/.cache/vosk

# The running container downloads nothing (the entrypoint keeps the corpus in the volume).
# Access logs are off: request lines carry client addresses and report receipts. Turn them on
# for debugging with UVICORN_ACCESS_LOG=true.
ENV HF_HUB_OFFLINE=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    UVICORN_ACCESS_LOG=false \
    PORT=8000
USER 10001:10001
EXPOSE 8000
# Liveness: the API answers. The first start also seeds Level 2 (embeds ~500 demo
# transcripts), hence the long start period.
HEALTHCHECK --interval=30s --timeout=5s --start-period=300s --retries=3 \
    CMD ["python", "-c", "import json, os, sys, urllib.request as u; r = u.urlopen('http://127.0.0.1:%s/api/health' % os.environ.get('PORT', '8000'), timeout=4); sys.exit(0 if json.load(r).get('status') == 'ok' else 1)"]
ENTRYPOINT ["/app/scripts/deploy_entrypoint.sh"]
