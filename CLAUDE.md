# CLAUDE.md — Qorğan

> Brief for agents and new contributors. It replaces the July 2026 selection-sprint brief
> (still readable with `git show e26f91e:CLAUDE.md`), whose XLM-R / Streamlit-first plan is
> history. **Where the truth is, in order:** `docs/STATUS.md` (current state) →
> `docs/PLAN_2026-09.md` (post-review plan, open items) → `docs/DECISIONS.md` (ADRs; read the
> newest first) → `docs/eval_report.md` (every number, with intervals). The external review the
> September plan answers is `qorgan-council-verdict.md` (RU) — read it before proposing scope,
> Level 2 or privacy changes. `DOCUMENTATION.md` is the long-range vision, not the current state.

## What it is
Decision support for **phone-scam (social-engineering) detection in Kazakh, Russian and
code-switched calls**. A human always decides: nothing auto-blocks, auto-reports or hangs up.
- **Level 1 (citizen), on the device:** `site/` is a static PWA. It transcribes a
  speakerphone call in the browser (Vosklet KK + RU, voted per utterance) or replays/pastes a
  transcript, embeds it with `multilingual-e5-base` int8 ONNX in a web worker, and shows a
  calibrated 0–100 suspicion meter, verbatim trigger spans with tactic tags, templated RU/KK
  reasons, advice, a post-call summary and a **reviewed, consent-gated, editable report**.
- **Level 2 (analyst / partner):** fed only by consented reports. Numbers are HMAC-hashed,
  transcripts scrubbed. Linking into scam "organizations" is mainly via the phone-number
  co-occurrence graph. Opening a full transcript is an explicit, audited action.

## Architecture invariants — enforced by `tests/test_architecture.py`, never break them
1. No route accepts audio. 2. `/api/analyze` persists nothing. 3. Level 2 has one citizen
ingress (`POST /api/reports`) plus the consented partner ingress (`POST /api/v1/reports`).
4. Raw phone numbers are never persisted (HMAC digest + `+7 700 ***` prefix only).
5. Stored transcripts are PII-scrubbed. 6. Every report has a receipt, can be deleted, and
expires on the server's clock (`QORGAN_REPORT_RETENTION_DAYS`, default 180; the server purges at
startup and daily). Every write route is named in `tests/test_architecture.py`.
7. Every `/api/admin` route needs an authenticated analyst (`X-Analyst-Key`, `QORGAN_ANALYST_KEYS`)
and fails closed; a full transcript needs the investigator role and a stated purpose; the
audit log is an HMAC chain (`QORGAN_AUDIT_CHAIN_KEY`; `python -m qorgan.audit verify`).

`app/` (Streamlit) is a **local dev harness only** (ADR D4): it scores on the server and its
mic mode uploads audio. Never present it as the product.

## The model that ships (`linear` backend)
- Head+tail rolling window, `query: ` prefix → 768-d e5-base embedding ⊕ 5 hard-signal cue
  flags ⊕ 1 reassurance flag → calibrated LR (risk) + 15 per-tactic LRs. Alert at risk ≥ 0.59
  (hysteresis 0.59 / 0.49); meter arms from the 3rd utterance unless a hard signal fires.
- Heads are trained on the **browser's own embeddings** (`QORGAN_EMBED_BACKEND=device` via
  `npm run device:serve`); the server's ONNX is a documented proxy (ADR D32/D33). Server
  `onnxruntime` is pinned to match transformers.js — do not bump it casually.
- The bundle hash-validates the lexicons: **any lexicon edit forces a retrain and must come
  with training data.** Upload model + lexicons together (`scripts/hf_upload.py`).
- **Parity:** `site/core/*.js` is a 1:1 port of the Python classifier, meter, explain, cue
  matcher and scrubber, pinned by golden fixtures. Change both sides, regenerate fixtures
  (`scripts/export_parity_fixtures.py`, `scripts/export_scrub_fixtures.py`), keep `npm test` green.
- Explanations are verbatim spans + templated RU/KK text (`explain/templates_*.yaml`,
  `advice_{ru,kk}.yaml`). **Never show LLM prose to the user as an explanation.**
- Other backends: `mock` (keyword fallback), `llm` (Gemini cloud second opinion — it sends text
  abroad, so it is off unless `QORGAN_CLOUD_TIER=on`, needs per-request `cloud_consent`, is never
  cached and never used by analyst routes; ADR D49), `xlmr` (abandoned).

## Evaluation rules
- **FPR first**, with Clopper–Pearson intervals, per split and per language. No change ships
  if it moves FPR on test / authored / ood / ASR-styled data or the browser gate without an ADR.
- **The honest number is `shift`** (66 calls from a second generator): recall 0.364. Every other
  split shares its generator with train. Real calls (`docs/DATA_INTAKE.md`) do not exist yet.
- **Never tune on held-out data**; never extend the lexicon from `shift` or `authored_heldout`
  (ADRs D35/D43). Any held-out row you read goes into `data/anchors/inspection_ledger.yaml`.
  Paste numbers from harness output, never by hand.
- Never publish `data/synthetic/` or `data/processed/{incidents,organizations,citizen_reports,audit_log}.jsonl`.
  `scripts/hf_upload.py` refuses to publish a split that is not a scrub fixed point (ADR D45).

## Commands
```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]" && cp .env.example .env  # runtime + tests; ".[all]" for harness/cloud/research
python scripts/deploy_bootstrap.py               # corpus + heads from HF, int8 ONNX, Vosk tarballs, L2 seeds
python -m qorgan.api                             # http://localhost:8000 — index · live.html · admin.html · /docs
pytest -q && npm install && npm test             # Python suite (offline) + JS parity / core tests
QORGAN_CLASSIFIER_BACKEND=linear python -m qorgan.eval.run --split test --split authored_heldout --split ood --split shift --by-language
QORGAN_CLASSIFIER_BACKEND=linear python -m qorgan.eval.stream --split test --split authored_heldout --backend linear
python -m qorgan.data.build_corpus && python -m qorgan.classifier.linear_train   # after ANY data/lexicon change
python -m qorgan.web.client_config && python scripts/export_parity_fixtures.py  # refresh JS config + fixtures
```
Browser checks use Playwright (`tests_js/tools/`); set `QORGAN_E2E_CHANNEL=chrome` to drive an
installed Chrome when Playwright's own Chromium is unavailable.

## Working rules
- Record every decision as the next ADR in `docs/DECISIONS.md` and update `docs/STATUS.md`.
  Rollbacks are fine; document them.
- `src/qorgan/config.py` holds every constant; small files; deterministic seeds.
- Tests first for deterministic code. Each change ends with evidence (a number, a test, a
  screenshot), never "should work".
- Secrets live only in `.env` / the deployment's secret store (see `.env.example`): the server
  refuses to link numbers without `QORGAN_NUMBER_HMAC_KEY`, and closes `/api/admin` and `/api/v1`
  without `QORGAN_ANALYST_KEYS` / `QORGAN_AUDIT_CHAIN_KEY`.
- The citizen page's text lives in `site/i18n.js` (kk / ru / en); model content (advice, tactic
  names, reasons) stays in the reviewed YAML and is never copied there.
- Ask before anything outward-facing: pushing, opening PRs, publishing to Hugging Face
  (`sanzh-ts/govtech`, `sanzh-ts/govtech_ds`), deploying.
