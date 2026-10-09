# DEPLOY.md — running Qorğan in production-like mode (Docker)

This runbook covers the **local, production-like deployment**: one container, secrets supplied
at run time, a persistent data volume and a health check. The last section lists what a
**public** deployment still needs, and why this setup does not provide it yet.

| File | Role |
|---|---|
| `Dockerfile` | multi-stage build: dependency list → pinned deps → published corpus → runtime (non-root, health check) |
| `.dockerignore` | allowlist: only `src/ site/ data/ configs/ scripts/ pyproject.toml README.md` reach the builder |
| `compose.yaml` | the service: loopback port, env file, data volume, limits, hardening |
| `scripts/deploy_entrypoint.sh` | at start: refresh the corpus in the volume → bootstrap → one uvicorn worker |
| `scripts/deploy_constraints.txt` | exact runtime versions the test suite passed on |

## 1. What runs where

```
host 127.0.0.1:8000 ──► container qorgan-qorgan-1 (uid 10001, read-only root filesystem)
                          uvicorn qorgan.api:app, ONE worker
                          /app/.env              ◄── bind mount of ./.env, read-only (secrets)
                          /app/data/processed    ◄── named volume `qorgan-data` (all state)
                          /tmp                   ◄── tmpfs
```

- **The image holds no secrets and no personal data.** It holds the code, the pages, the
  int8 embedder and the Vosk models (from the build context), the committed head weights
  `site/models/weights.json`, and the published corpus from `sanzh-ts/govtech_ds`. The build
  needs no secret. `.env`, `data/processed`, `data/synthetic`, `models/`, `node_modules`,
  the tests and the Streamlit harness never enter the build context.
- **Model:** the server scores with exactly the weights the browser ships
  (`QORGAN_LINEAR_WEIGHTS=web`, ADR D55). Nothing is downloaded from the Hub model repo or
  retrained, either at build time or at start. `/api/health` reports `"linear_model_source": "web"`.
- **State** lives only in the `qorgan-data` volume: `citizen_reports.jsonl`, `audit_log.jsonl`,
  the Level-2 demo seeds (`incidents.jsonl`, `organizations.jsonl`, `incident_embeddings.npz`),
  `org_feedback.jsonl`, and a copy of the published corpus splits, which the entrypoint
  refreshes from the image on every start. Files are created with mode 0640.
- **Secrets** stay in the env file. It is mounted read-only at `/app/.env`, where
  `src/qorgan/config.py` loads it. They are **not** container environment variables, so
  neither `docker inspect`, nor Docker Desktop's *Inspect* tab, nor `docker compose config`
  shows them. The `environment:` block in `compose.yaml` pins the deployment invariants
  (backend, weights, paths), and it takes precedence over the env file.
- **The running container downloads nothing** (`HF_HUB_OFFLINE=1`).
- **One worker, one replica.** Rate limits, the investigator's hourly open budget and the
  purge schedule are in-process state. Never scale past one container on the same volume.
- **The access log is off** (`UVICORN_ACCESS_LOG=false`), because request lines carry client
  addresses and report receipts (`DELETE /api/reports/<receipt>`). Turn it on only for
  debugging: set `UVICORN_ACCESS_LOG: "true"` under `environment:`.

## 2. Deploy

Prerequisites:
- Docker Engine with Compose v2.24 or later (Docker Desktop on macOS works).
- About 4 GB of free disk.
- `site/models/` populated: the int8 embedder under `Xenova/` and the Vosk tarballs under
  `vosk/`. If they are missing, the build downloads them once, from Hugging Face (the
  embedder) and alphacephei.com (the Vosk models).

```bash
cd govtech
# 1. Secrets: the env file (never committed; see .env.example for every variable).
cp .env.example .env && chmod 600 .env      # skip the copy if .env already exists
#    Required: QORGAN_NUMBER_HMAC_KEY, QORGAN_AUDIT_CHAIN_KEY, QORGAN_ANALYST_KEYS (§3).
# 2. Build and start.
docker compose up -d --build
docker compose ps                     # wait for "(healthy)"
curl -s http://127.0.0.1:8000/api/health
#    {"status":"ok","configured_backend":"linear","threshold":0.59,"linear_model_source":"web"}
```

Open the pages:
- Landing page: `http://127.0.0.1:8000/`
- Live call: `http://127.0.0.1:8000/live.html`. The first visit downloads the ~280 MB
  embedder, plus ~106 MB of speech models if you use the microphone.
- Analyst console: `http://127.0.0.1:8000/admin.html`
- API docs: `http://127.0.0.1:8000/docs`

**Timings measured on an Apple M4 (arm64, Docker Desktop, 2026-09-29).**

| Event | Time | Notes |
|---|---|---|
| First start on an empty volume | ~91 s to healthy | seeds Level 2: embeds 500 demo transcripts |
| Restart or recreate | ~7 s | |
| Image build | ~20 s | dependency layer cached |
| Rebuild after a code or page edit | ~6 s | |

**Resources:**
- Image: ~1.7 GB.
- Memory peaks at ~0.8 GB during the first-start seeding and idles at ~140 MB.
- Limits: 4 CPUs, 2 GB RAM, 512 PIDs.

To use another host port, run `QORGAN_HOST_PORT=8080 docker compose up -d`. To use an env
file kept outside the repository, run `QORGAN_ENV_FILE=/etc/qorgan/qorgan.env docker compose up -d`.

## 3. Keys: format, issue, rotate

The three runtime keys, each at least 32 bytes and each generated by a CSPRNG:

| Variable | Purpose | Generate |
|---|---|---|
| `QORGAN_NUMBER_HMAC_KEY` | salts the caller-number digests (raw numbers are never stored) | `python -c "import secrets; print(secrets.token_hex(32))"` |
| `QORGAN_AUDIT_CHAIN_KEY` | keys the audit-log hash chain. Must differ from the number key: whoever verifies the log must not be able to reverse phone digests. | same |
| `QORGAN_ANALYST_KEYS` | analyst console identities | see below |

**Analyst keys.**
- Format: comma-separated entries of `account_id:secret:role`, one entry per **person** (the
  audit log names the account).
- Roles:
  - `analyst`: overview, statistics, excerpts, confirm/dismiss/merge, ingest.
  - `investigator`: everything `analyst` can do, plus opening a whole call with a stated
    purpose, at most 30 per hour.
- Secrets must be at least 16 characters, unique, and never also used as a partner key.
  Generate one with `python -c "import secrets; print(secrets.token_urlsafe(32))"`.
- If the variable is unset, the console answers 503. It is never open.
- Hand each key to its person out of band. The owner's list of issued keys is kept outside
  the repository.
- Partner keys (`QORGAN_PARTNER_API_KEYS`, `partner_id:secret[:daily_quota]`) work the same way.

**Apply a change.** After editing the env file, recreate the container. A plain `restart` can
keep serving the old file content: many editors replace the file, and a single-file bind mount
keeps pointing at the old one.

```bash
docker compose up -d --force-recreate
```

**Rotate.**
- **Analyst or partner key:** replace the secret in the env file, then recreate the container.
  The old key stops working at once. The audit log records account ids, never keys.
- **Audit chain key:** rotating it breaks verification of the lines written under the old key.
  Close the old chain first:
  ```bash
  docker compose exec qorgan python -m qorgan.audit verify      # must say "OK: chain intact"
  docker compose exec qorgan python -m qorgan.audit head        # record SEQ:MAC off-host
  docker compose stop
  docker run --rm -v qorgan-data:/data python:3.11-slim \
      mv /data/audit_log.jsonl /data/audit_log.until-$(date +%Y%m%d).jsonl
  # set the new QORGAN_AUDIT_CHAIN_KEY in the env file; keep the old key in the secret store
  docker compose up -d --force-recreate                          # a new chain starts
  ```
  To check the archived chain later, export the **old** key in your shell from the secret
  store (never type it on a command line), then run
  `docker compose run --rm -e QORGAN_AUDIT_CHAIN_KEY qorgan python -m qorgan.audit verify --path /app/data/processed/audit_log.until-YYYYMMDD.jsonl`.
- **Number HMAC key:** reports hashed under the old key no longer link to new ones (by design:
  digests cannot be converted). Rotate only for a suspected leak. The Level-2 demo seeds are
  hashed with the key, so reseed them after a rotation:
  ```bash
  docker compose stop
  docker run --rm -v qorgan-data:/data python:3.11-slim \
      rm -f /data/incidents.jsonl /data/organizations.jsonl /data/incident_embeddings.npz
  docker compose up -d --force-recreate          # the bootstrap reseeds (~1.5 min)
  ```

## 4. Operations

**Health.** `docker compose ps` shows the state from the image's `HEALTHCHECK`
(`GET /api/health` every 30 s). The start period is 300 s, because the first start seeds
Level 2. Check the model source as well:

```bash
curl -s http://127.0.0.1:8000/api/health        # linear_model_source must be "web"
```

**Logs.** `docker compose logs -f qorgan`. Logs rotate at 10 MB × 3 files. They contain the
bootstrap and server messages, never call content.

**Retention.**
- Consented reports expire after `QORGAN_REPORT_RETENTION_DAYS` (default 180), on the
  server's clock.
- The server purges them itself, at startup and every `QORGAN_REPORT_PURGE_INTERVAL_HOURS`
  (default 24; 0 means you must schedule the purge yourself).
- Preview a purge: `docker compose exec qorgan python -m qorgan.reports.purge --dry-run`.
- The audit log is kept **at least three years** (`QORGAN_AUDIT_RETENTION_DAYS`, default and
  floor 1096; a lower value is refused at startup) and is **never** purged with the reports:
  no deletion path touches `audit_log.jsonl` (ADR D66, `tests/test_audit_retention.py`).
  Deleting lines would break `verify`. After the retention period, archive the whole file to
  WORM storage together with its last `head` anchor, then start a new log; never edit it.
- Not covered: the org-feedback digests have no separate retention period (they follow their
  reports, ADR D49). Whether the audit log must be kept longer than 3 years is a legal decision
  (`docs/LEGAL_ASSESSMENT.md` §5.2(5)).

**Audit log.** The log is a keyed hash chain (ADR D46).

```bash
docker compose exec qorgan python -m qorgan.audit verify    # exit 0 + "OK: chain intact"
docker compose exec qorgan python -m qorgan.audit head      # SEQ:MAC of the newest entry
```

- Record `head` daily somewhere the server cannot rewrite, such as a ticket or another host.
- Later, `verify --anchor SEQ:MAC` also detects tail truncation, which the chain alone
  cannot detect.
- `verify` names the first edited, deleted or reordered entry and exits 1.

**One-off commands.** The image runs any command you give it instead of the server:

```bash
docker compose run --rm qorgan python -m qorgan.audit verify
```

## 5. Backup and restore

The volume holds personal data: scrubbed transcripts, number digests and the audit log.
Handle backups under these rules:
- Encrypt them.
- Store them in Kazakhstan.
- Keep them no longer than the report retention period.
- **A restore brings back reports that citizens deleted after the backup was taken**, so keep
  backup retention short. The startup purge removes only *expired* reports, not deleted ones.
- Back up the secrets separately, in the secret store. The backup is useless without them:
  the audit chain cannot be verified, and new numbers do not link to the old digests.

```bash
# Backup (a consistent copy: stop for a few seconds)
mkdir -p backups && docker compose stop
docker run --rm -v qorgan-data:/data:ro -v "$PWD/backups:/backup" python:3.11-slim \
    tar czf /backup/qorgan-data-$(date +%Y%m%d-%H%M%S).tgz -C /data .
docker compose start

# Restore (replaces ALL current state)
docker compose stop
docker run --rm -v qorgan-data:/data -v "$PWD/backups:/backup" python:3.11-slim \
    sh -c 'find /data -mindepth 1 -delete && tar xzf /backup/qorgan-data-YYYYMMDD-HHMMSS.tgz -C /data'
docker compose start
docker compose exec qorgan python -m qorgan.audit verify
```

`backups/` is not ignored by git. Keep backups outside the repository, or add the directory to
`.gitignore` before you use it.

## 6. Upgrade and roll back

```bash
# 1. Back up (§5) and keep the running image under a rollback tag
docker tag qorgan:local qorgan:rollback-$(date +%Y%m%d-%H%M)
# 2. Build from the new tree and recreate; the volume is kept
docker compose up -d --build
docker compose ps && curl -s http://127.0.0.1:8000/api/health
docker compose exec qorgan python -m qorgan.audit verify
# 3. Roll back if the checks fail
QORGAN_IMAGE_TAG=rollback-YYYYMMDD-HHMM docker compose up -d --no-build
#    (restore the §5 backup too if the new version had already written to the volume)
```

**Rebuild triggers.**
- A code or page edit rebuilds only the small layers (~6 s).
- The models and the dependency layer are cached separately.
- The corpus is fetched once and cached. After the dataset is re-published, fetch it again:
  ```bash
  docker compose build --build-arg QORGAN_CORPUS_REFRESH=$(date +%s)
  ```

**Upgrade dependencies.**
- `pyproject.toml` stays the dependency list. `scripts/deploy_constraints.txt` pins the exact
  versions of its runtime closure.
- To move a pin:
  1. Edit the pin.
  2. Prove the new version in a venv:
     `pip install -e ".[dev]" -c scripts/deploy_constraints.txt && pytest -q && npm test`
  3. Rebuild the image.
- A new dependency installs unpinned until you add it. See what was installed with
  `docker run --rm qorgan:local pip freeze`.
- `onnxruntime` moves only together with transformers.js (ADR D32).

## 7. Before a public deployment (not done here, on purpose)

This setup binds to **127.0.0.1 without TLS** and keeps state in files on one host. That is
right for a laptop demo and wrong for citizens' data. A public deployment needs:

1. **Hosting in Kazakhstan.**
   - Reports, the audit log and the analyst store are personal data of KZ residents. They
     must live in a database located in Kazakhstan (PD Law 94-V Art. 12(2);
     `docs/LEGAL_ASSESSMENT.md` §2.4).
   - That covers backups and logs too.
   - Hugging Face and GitHub may hold only code and synthetic data.
   - Cloud storage of special-category data needs keys controlled by a KZ entity (Law 231-VIII).
2. **TLS** at a reverse proxy (Caddy or nginx) in front of the container, with HSTS.
   - Set `FORWARDED_ALLOW_IPS` to the proxy's address, so rate limits see real client
     addresses and not the proxy.
   - The microphone mode needs a secure context (HTTPS, or `localhost`).
   - The app already sets COOP/COEP on `live.html`.
3. **Analyst SSO with MFA and RBAC.**
   - The per-person keys stand in for it (ADR D46, not done: expiry, MFA).
   - The 2026 orders reportedly require MFA and role-based access even for medium operators
     (LEGAL_ASSESSMENT §2.4).
4. **A real database** (Postgres, in KZ) instead of JSONL plus `flock` on one host, with
   point-in-time recovery. Until then: one worker, one host.
5. **A secret store** (Vault or a KMS, with keys under KZ control) instead of a file.
   - Rotation procedures as in §3.
   - A key id for the audit chain key.
6. **Tamper evidence off the host.** Ship audit lines or daily `head` anchors to WORM storage;
   the key holder can still rewrite history on the host.
7. **Observability:** an external uptime check on `/api/health`, alerting on "unhealthy",
   volume disk-usage alerts, and log shipping with a retention period. Logs carry no content.
8. **Supply chain:**
   - Pin the base image by digest.
   - Scan the image (Trivy or Docker Scout) and use hash-checked requirements.
   - SBOMs (CycloneDX) and the dependency / static-analysis reports are in
     `docs/egov/security/` (ADR D66); regenerate with `python scripts/generate_sbom.py`.
   - Decide the publication posture before pushing the image to any registry. The image
     contains the head weights and the lexicons (council advice, ADR D15).
9. **Legal preconditions.**
   - A legal owner for the `docs/DATA_INTAKE.md` checklist.
   - Notification under Art. 10-1 above 10,000 subjects.
   - A privacy notice, and the AI-law classification (LEGAL_ASSESSMENT §4).
   - The cloud tier stays `off`: it sends text abroad.

## 8. Railway (the hosted demo)

The hosted demo (`govtech-production.up.railway.app`) builds this same `Dockerfile` from the
GitHub repository. Railway's builder is stricter than a local BuildKit, and ADR D57 records the
deploy it failed. `tests/test_dockerfile_portability.py` keeps these rules:

- **No BuildKit cache mounts.** Railway refuses `RUN --mount=type=cache` unless the id carries
  the hard-coded service id (`id=s/<service-id>-<path>`). The build fails with *"Cache mount ID
  is not prefixed with cache key"*. Variables do not work in the id, and a hard-coded id would
  tie the Dockerfile to one service. So downloads go into a normal layer, and the bootstrap
  `RUN` deletes its download caches in the same layer.
- **No `COPY --exclude` and no `HEALTHCHECK --start-interval`.** Older Dockerfile frontends
  reject both (*"unknown flag: exclude"*). Railway ignores `HEALTHCHECK` anyway. Configure the
  health check in the service settings instead.
- **The clone has no models.** `site/models/Xenova`, `site/models/vosk` and `site/vendor` are
  gitignored. On Railway, the bootstrap step of the build downloads them: the embedder from
  the Hub, and the Vosk models and runtime pinned and hash-checked. It also downloads the
  browser embedder's runtime (transformers.js plus the ONNX-runtime WASM, also pinned and
  hash-checked, ADR D61). That download is **required**: a CDN import is blocked by COEP in
  Safari, so the build fails without it. This adds about 3 minutes to the build.

Service settings:

| Setting | Value |
|---|---|
| Variables | `QORGAN_NUMBER_HMAC_KEY`, `QORGAN_AUDIT_CHAIN_KEY`, `QORGAN_ANALYST_KEYS` (§3). Railway passes them as environment variables, and there is no `.env` file in the container. `PORT` is set by Railway. |
| Health check | path `/api/health`. The first start seeds Level 2 before the server listens (about 25 s on a laptop CPU). |
| Open demo access (optional) | `QORGAN_ADMIN_OPEN_ACCESS=investigator` (or `analyst`) lets visitors without a key, such as a jury, into the console as `public-demo`. It is still audited. Use it only with fabricated data, and remove it to require keys again (ADR D59). |
| Replicas | **1** (§1: in-process state) |
| Volume (optional) | mount it at **`/app/data/processed`**, never at `/app/data`, which holds the lexicons and the taxonomy. Set **`RAILWAY_RUN_UID=0`**. |

**Volumes and the non-root user.** Railway mounts a volume owned by root, and the image runs as
uid 10001. Started as root (`RAILWAY_RUN_UID=0`), the entrypoint gives the state directory to
uid 10001 and re-executes itself as that uid through `setpriv`. The server never runs as
root. Without that variable, the container stops at once with exit code 78 and a message that
names the fix.

**Without a volume, every redeploy starts empty.** Citizen reports, receipts and the audit
chain are lost, and the Level-2 seeds are re-created with the current key. That is acceptable
for a demo with fabricated data, and it is not acceptable for real reports (§7).
