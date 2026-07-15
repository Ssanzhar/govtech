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
cp .env.example .env             # optional: GEMINI_API_KEY only for data-gen / llm backend

# 2. Get the trained model (pick ONE)
#    a) pull from Hugging Face (recommended):
python -c "from huggingface_hub import snapshot_download; snapshot_download('sanzh-ts/govtech', local_dir='models/linear_hf')"
export QORGAN_LINEAR_MODEL_DIR=models/linear_hf     # lexicons are already committed in data/lexicon/
#    b) or retrain from the committed corpus (seconds, CPU):
python -m qorgan.data.build_corpus && python -m qorgan.classifier.linear_train

# 3. Seed the Level-2 demo data (~500 synthetic incidents -> organizations + embedding cache)
python scripts/demo_seed.py && python -m qorgan.analytics.pipeline

# 4. Run
streamlit run app/streamlit_app.py
```

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
# FPR-first tables (test + real_heldout + ASR-stress), per language
QORGAN_CLASSIFIER_BACKEND=linear python -m qorgan.eval.run \
    --split test --split real_heldout --split ood --by-language

# Streaming eval: false-latch rate (live FPR analog), time-to-alert
QORGAN_CLASSIFIER_BACKEND=linear python -m qorgan.eval.stream \
    --split test --split real_heldout --backend linear

pytest -q        # ~700 tests, all offline
```

Shipped numbers (threshold 0.55): **test FPR 0.000 / recall 0.953 · real_heldout FPR
0.000 / recall 1.000 · ood FPR 0.000 / recall 0.889**. Methodology + honest caveats:
[`docs/eval_report.md`](docs/eval_report.md), data provenance:
[`data/README.md`](data/README.md).

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
