#!/bin/sh
# Container entrypoint (Dockerfile; operations in docs/DEPLOY.md).
#
#   (no arguments, or uvicorn options)  serve: refresh the published corpus in the data volume,
#                                       run the idempotent bootstrap, seed Level 2 in the
#                                       background on the FIRST start (ADR D60, with the runtime
#                                       number key), then exec ONE uvicorn worker; options are
#                                       passed on to uvicorn.
#   <command> [args]                    run that command instead, e.g. an operations tool:
#                                       docker compose run --rm qorgan python -m qorgan.audit verify
set -eu

# Reports, the audit log and the Level-2 seeds are readable by the service account only.
umask 027

if [ "$#" -gt 0 ] && [ "${1#-}" = "$1" ]; then
    exec "$@"
fi

state_dir="${QORGAN_DATA_DIR:-/app/data}/processed"
mkdir -p "$state_dir"

# A platform volume is mounted root-owned (Railway: run with RAILWAY_RUN_UID=0, docs/DEPLOY.md
# §8), which the service account cannot write. Started as root, take the state directory for
# the service account and re-run this script as it; the server itself never runs as root.
service_uid=10001
if [ "$(id -u)" = 0 ]; then
    chown -R "$service_uid:$service_uid" "$state_dir"
    export HOME=/home/qorgan
    exec setpriv --reuid="$service_uid" --regid="$service_uid" --init-groups --inh-caps=-all -- "$0" "$@"
fi
if [ ! -w "$state_dir" ]; then
    echo "qorgan: $state_dir is not writable by uid $(id -u) (a root-owned volume?). Start the" \
         "container as root so it can take the directory -- on Railway set RAILWAY_RUN_UID=0" \
         "(docs/DEPLOY.md §8) -- or chown the volume to $service_uid." >&2
    exit 78
fi
# The volume holds runtime state next to the published corpus splits the bootstrap expects
# there. The image's copy wins, so a volume created by an older image gets the current,
# scrubbed splits and nothing is fetched from the network. Runtime state is never in this copy.
cp /opt/qorgan/corpus/* "$state_dir"/

# Railway's edge fronts every request and the container is not reachable any other way: without this
# every per-client limit counts the proxy, and one visitor could use up the whole room's. Elsewhere
# the operator decides (docs/DEPLOY.md).
if [ -n "${RAILWAY_ENVIRONMENT:-}${RAILWAY_ENVIRONMENT_NAME:-}" ]; then
    export QORGAN_TRUSTED_PROXY_HOPS="${QORGAN_TRUSTED_PROXY_HOPS:-1}"
fi

python /app/scripts/deploy_bootstrap.py --phase serve
# Seeding Level 2 embeds ~500 transcripts: minutes on a shared vCPU, longer than a platform's
# health-check window (Railway: 5 min). It runs in the background while the server answers; the
# console shows "no Level-2 analysis yet" until the files land (they are written atomically).
python /app/scripts/deploy_bootstrap.py --phase seeds &

exec uvicorn qorgan.api:app --host 0.0.0.0 --port "${PORT:-8000}" --workers 1 --no-server-header "$@"
