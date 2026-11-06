#!/usr/bin/env bash
# Deploy Qorğan to a team server with Docker (e.g. the GovTech team server: its HTTPS front end
# forwards https://<team>.govtech-kz.com to the app port, 8020). Run it yourself:
#
#   scripts/deploy_remote.sh user@host [app_port]        # you type the SSH password once
#
# What it does:
#   1. ships the image build context (the .dockerignore allowlist) -- never .env, keys.md or
#      runtime data (data/processed, data/synthetic, ...);
#   2. on the first deploy, writes ~/qorgan/.env on the server: the analyst keys from your local
#      .env (the ones documented in keys.md) plus a NEW number-HMAC key and audit-chain key
#      generated on the server (they never leave it); later deploys keep the server's .env;
#   3. builds and starts the container on 0.0.0.0:<app_port> with the qorgan-data volume, and
#      waits for /api/health.
set -euo pipefail

TARGET="${1:?usage: scripts/deploy_remote.sh user@host [app_port]}"
PORT="${2:-8020}"
REMOTE_DIR="qorgan"
cd "$(dirname "$0")/.."

SOCKET="$(mktemp -u "${TMPDIR:-/tmp}/qorgan-ssh.XXXXXX")"
SSH=(ssh -o ControlMaster=auto -o ControlPath="$SOCKET" -o ControlPersist=15m -o StrictHostKeyChecking=accept-new)
cleanup() { "${SSH[@]}" -O exit "$TARGET" 2>/dev/null || true; }
trap cleanup EXIT

echo "==> connecting to $TARGET (one password prompt)"
"${SSH[@]}" "$TARGET" 'command -v docker >/dev/null || { echo "docker is not installed on the server" >&2; exit 1; }; docker compose version >/dev/null'

echo "==> shipping the build context (up to ~650 MB with the on-device models)"
tar -czf - \
  --exclude='__pycache__' --exclude='*.pyc' \
  --exclude='data/processed' --exclude='data/synthetic' --exclude='data/raw' --exclude='data/real' \
  --exclude='data/cache' --exclude='data/asr_capture/_wav' \
  src site data configs scripts pyproject.toml README.md Dockerfile .dockerignore compose.yaml \
  | "${SSH[@]}" "$TARGET" "mkdir -p $REMOTE_DIR && tar -xzf - -C $REMOTE_DIR"

if "${SSH[@]}" "$TARGET" "test -f $REMOTE_DIR/.env"; then
  echo "==> keeping the server's existing .env"
else
  echo "==> writing the server's .env (analyst keys from local .env; new HMAC + audit keys generated there)"
  grep -E '^QORGAN_ANALYST_KEYS=' .env | "${SSH[@]}" "$TARGET" "umask 077 && cd $REMOTE_DIR && cat > .env && python3 -c 'import secrets; print(\"QORGAN_NUMBER_HMAC_KEY=\" + secrets.token_hex(32)); print(\"QORGAN_AUDIT_CHAIN_KEY=\" + secrets.token_hex(32))' >> .env && printf 'QORGAN_SUPPORTED_LOCALES=ru,kk,en\nQORGAN_DEFAULT_LOCALE=ru\n' >> .env"
fi

# Compose reads this file as the deploy user and the container as gid 10001; passwordless sudo only.
"${SSH[@]}" "$TARGET" "cd $REMOTE_DIR && (sudo -n chgrp 10001 .env && chmod 640 .env) || echo 'WARNING: run: sudo chgrp 10001 ~/$REMOTE_DIR/.env && chmod 640 ~/$REMOTE_DIR/.env'"

echo "==> building and starting on port $PORT"
"${SSH[@]}" "$TARGET" "cd $REMOTE_DIR && QORGAN_BIND_ADDR=0.0.0.0 QORGAN_HOST_PORT=$PORT docker compose up -d --build"

echo "==> waiting for health (the analyst demo data then seeds in the background: minutes on a small server)"
"${SSH[@]}" "$TARGET" "for i in \$(seq 1 60); do curl -sf http://127.0.0.1:$PORT/api/health && exit 0; sleep 5; done; echo 'not healthy yet: cd $REMOTE_DIR && docker compose logs' >&2; exit 1"
echo
echo "==> deployed. Check the public URL your organisers gave you (it forwards to port $PORT)."
