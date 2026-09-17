# Qorğan

**AI decision-support that flags social-engineering (scam) patterns in Kazakh/Russian
phone conversations — live, as the call happens — explains *why*, and clusters citizen
reports into scam "organizations" for a government analyst. A human always decides.**

Built for the GovTech Camp selection stage. This repo is the **web prototype**; the full
mobile / on-device vision lives in [`DOCUMENTATION.md`](DOCUMENTATION.md) and the
[`docs/`](docs/) roadmap.

> **For contributors & agents:** current state / handoff is
> [`docs/STATUS.md`](docs/STATUS.md); then [`CLAUDE.md`](CLAUDE.md) →
> [`docs/SCOPE.md`](docs/SCOPE.md) → [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).
> Model numbers live in [`docs/eval_report.md`](docs/eval_report.md).

## What it does — three tabs, one pipeline

| Tab | Persona | What happens |
|---|---|---|
| **Level 1 — Call check** | citizen | Paste/pick a transcript → calibrated **risk score** → **explained** alert: highlighted trigger phrases, tactic tags, plain RU/KK reason, honest confidence. |
| **Live call** | citizen | A call is analyzed **turn by turn**: streaming utterances → rolling window → **0–100 suspicion meter** (hysteresis + hard-signal floors) → grounded evidence cards → tactic-specific advice (RU/KK) → post-call summary → **consent-gated, editable report**. Input: replay a script (zero setup), or a real **microphone** (browser or local) with dual Vosk KK+RU streaming ASR. |
| **Level 2 — Analyst view** | gov analyst | KPI row, priority queue of scam **organizations** (named by dominant tactics), new-scheme flags, drill-down with tactic/activity charts — and an **Ingest** button that pulls submitted citizen reports into the analysis (a report whose number matches a known org joins it; unknown numbers become novelty candidates). |

## Quick start

```bash
# 1. Environment (Python 3.11+)
python -m venv .venv && source .venv/bin/activate
pip install -e .                 # src-layout: puts `qorgan` on the path
cp .env.example .env             # set QORGAN_NUMBER_HMAC_KEY (see the file); GEMINI_API_KEY only for data-gen

# 2. Models + demo data in one go (idempotent; ~300 MB download on first run):
#    corpus splits + trained head weights from Hugging Face, the int8 ONNX embedder the
#    browser ships (self-hosted under site/models/), and the Level-2 demo seeds.
python scripts/deploy_bootstrap.py
#    ...or retrain the heads from the committed corpus (seconds, CPU):
python -m qorgan.data.build_corpus && python -m qorgan.classifier.linear_train

# 3. Run the site (landing + live call + analyst dashboard) -- analysis runs ON THE DEVICE
python -m qorgan.api             # http://localhost:8000
#    or the Streamlit prototype:
streamlit run app/streamlit_app.py
```

**On-device by design.** The pages under `site/` load the same `multilingual-e5-base`
int8 ONNX graph and the exported head weights (`site/models/`) and score transcripts in
the browser (`site/core/`, a 1:1 port of the Python classifier — `npm test` proves parity
on golden fixtures). The server embeds with the *same* int8 graph
(`QORGAN_EMBED_BACKEND=onnx`), so a verdict is identical wherever it is computed. No
route accepts audio; call content leaves the device only on an explicit report.

**No model, no key?** The app still runs — it degrades to a deterministic `mock` backend
so the demo scripts work out of the box.

**Live microphone (optional):** `pip install -e ".[live]"` (vosk, streamlit-webrtc,
sounddevice). First use downloads two small Vosk models (~100 MB) to `~/.cache/vosk`.
Put the call on speakerphone near the device. Without the extra, the Live tab's replay
mode still works and the mic modes show an install hint.

## The demo storyline (3 scenes)
1. **Live scam call** (`live_scam_bank_ru` scenario) — the meter climbs to Critical,
   evidence and advice appear mid-call, post-call summary offers a report.
2. **Hard negative** (`live_hard_negative_bank_ru`) — a *real* bank call does **not**
   trigger. False-positive discipline is the product's core metric.
3. **Analyst view** — submit the report from scene 1 (use a number from a seeded org,
   e.g. `+7 700 101 20 30`), then click **Ingest into analysis**: watch it land inside
   that organization.

## Evaluate

```bash
# FPR-first tables (test + authored_heldout + ASR-stress), per language
QORGAN_CLASSIFIER_BACKEND=linear python -m qorgan.eval.run \
    --split test --split authored_heldout --split ood --by-language

# Streaming eval: false-latch rate (live FPR analog), time-to-alert
QORGAN_CLASSIFIER_BACKEND=linear python -m qorgan.eval.stream \
    --split test --split authored_heldout --backend linear

pytest -q        # ~700 tests, all offline
```

Shipped numbers (threshold 0.55, July): **test FPR 0.000 / recall 0.953 · authored_heldout
FPR 0.000 / recall 1.000 · ood FPR 0.000 / recall 0.889**. Read them with their intervals:
`authored_heldout` is **hand-written, not real calls** (18 scam / 24 legit), so its FPR of
0.000 has a 95 % Clopper–Pearson interval of **[0.000, 0.142]** and its recall of 1.000 a
lower bound of 0.815; five of its negatives were read during feature engineering and are
reported separately (`data/anchors/inspection_ledger.yaml`). The harness prints intervals
on every run. Methodology + caveats: [`docs/eval_report.md`](docs/eval_report.md), data
provenance: [`data/README.md`](data/README.md).

## Real calls (when they arrive)

`docs/DATA_INTAKE.md` is the policy and protocol: encrypted/on-prem delivery, a
`batch.yaml` + `calls.csv` batch format, `python scripts/ingest_partner_calls.py <batch>`
(scrub → hashed number linkage → first-come allocation into a hash-locked
`real_heldout_v2` of 60 legit / 40 scam, the rest to `real_train`). The locked set is
scored, never read, and only at release points. `data/real/` never enters git.

## Reports API (consented ingress) & privacy

`POST /api/reports` is the only way call content enters the analyst layer, and only on an
explicit user action. The server stores the transcript PII-scrubbed, the caller number as
a salted HMAC digest + prefix (`+7 700 ***`), returns exactly what it kept plus a receipt,
and `DELETE /api/reports/{receipt}` forgets it everywhere (`python -m qorgan.reports.purge`
applies the retention window). Set `QORGAN_NUMBER_HMAC_KEY` (see `.env.example`); without
it the server refuses reports that carry a number. No route accepts audio; these
invariants are enforced by `tests/test_architecture.py` (ADRs D12–D14).

## Partner API (`/api/v1`) — consented reports in, aggregates out

A bank fraud desk, telecom or hotline can feed confirmed cases into the analyst layer and
read back the organization-level picture — without ever sending call content it does not
have to. It is a second *consented* ingress, not a bulk feed (ADR D19):

- **Auth**: `X-API-Key` per partner; keys live in the server env
  `QORGAN_PARTNER_API_KEYS="bank_a:<secret ≥16 chars>[:daily_quota],telecom_b:<secret>"`
  (unset = the API is closed; `QORGAN_PARTNER_DAILY_QUOTA` is the default budget).
- **One report per request**, preferably **structured tactic hits**; a transcript is accepted
  only if the partner already PII-scrubbed it (the server checks and refuses otherwise,
  without echoing it). `consent_basis` (a code from the data-sharing agreement) is required.
  The caller number is hashed on receipt like a citizen report. `partner_reference` makes
  retries idempotent.
- **Limits**: 60 requests/min and a rolling 24 h quota per partner (`X-Quota-Limit` /
  `X-Quota-Remaining` on every response); a content-free audit line for every action
  (`data/processed/audit_log.jsonl`).
- **Export**: `GET /api/v1/organizations` returns aggregates only — no numbers, no digests,
  no transcripts. Partners can `DELETE` only their own receipts.

```bash
export QORGAN_PARTNER_API_KEYS="bank_a:replace-with-a-32-char-secret-00000000"
python -m qorgan.api    # OpenAPI at http://localhost:8000/docs (scheme: PartnerApiKey)

# 1. a confirmed case as structured signals (preferred shape)
curl -s -X POST http://localhost:8000/api/v1/reports \
  -H "X-API-Key: replace-with-a-32-char-secret-00000000" -H "Content-Type: application/json" \
  -d '{"consent_basis":"customer_consent","tactic_ids":["otp_request","safe_account"],
       "phone_number":"+7 700 555 66 77","partner_reference":"CASE-2026-0912"}'
# -> 201 {"receipt_id": "...", "number_prefix": "+7 700 ***", "quota": {"limit":200,"used":1,...}}
# 2. the same case again -> 200 "duplicate", nothing stored, quota untouched
# 3. the organization-level picture (aggregates only)
curl -s http://localhost:8000/api/v1/organizations?locale=ru -H "X-API-Key: replace-with-a-32-char-secret-00000000"
# 4. withdraw a report
curl -s -X DELETE http://localhost:8000/api/v1/reports/<receipt_id> -H "X-API-Key: replace-with-a-32-char-secret-00000000"
```

Signals-only reports (no transcript) are stored, receipted, deletable and counted; placing
them into organizations through the number graph alone is the next Level-2 item (PLAN C9).

## Publish model/data updates (maintainers)

After a retrain: `hf auth login` (write token) then `python scripts/hf_upload.py` — it
uploads `models/linear` **together with** `data/lexicon` (the bundle hash-validates the
lexicons) and the scrubbed dataset splits. Never widen the dataset patterns: the other
`data/processed/` files (incidents, citizen reports) and raw `data/synthetic/` must stay
off the Hub.

## Layout
`src/qorgan/` (config · taxonomy · data · classifier · explain · **live** · analytics ·
asr · eval) · `app/` (Streamlit: `streamlit_app.py`, `live_view.py`, `mic_live.py`,
`analyst_view.py`) · `data/` (taxonomy · lexicons · corpus + provenance) · `tests/` ·
`docs/` (scope, architecture, decisions, status, eval report).

## Status & limits
Web prototype. The live-mic path is real (Vosk streaming, KK/RU voting) but
speakerphone-quality ASR — especially Kazakh — is the accuracy bottleneck; the meter's
confidence weighting absorbs some of it. ~1 in 6 legit calls still latches the live meter
*transiently* mid-call (documented, measured by `eval.stream`; fix candidates in
`docs/STATUS.md`). Mobile, on-device, and carrier integration are the roadmap
([`DOCUMENTATION.md`](DOCUMENTATION.md)), not this repo.
