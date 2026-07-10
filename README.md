# Qorğan

**AI decision-support that flags social-engineering (scam) patterns in Kazakh/Russian
phone conversations, explains *why*, and clusters confirmed reports into scam
"organizations" for a government analyst. A human always decides.**

Built for the GovTech Camp selection stage. This repo is the **1-week web prototype**;
the full mobile / on-device vision lives in [`DOCUMENTATION.md`](DOCUMENTATION.md) and the
[`docs/`](docs/) roadmap.

> **For contributors & agents:** start at [`CLAUDE.md`](CLAUDE.md) →
> [`docs/SCOPE.md`](docs/SCOPE.md) → [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

## What it does
- **Level 1 (centerpiece):** transcript → scam **risk score** → **explained** alert
  (highlighted trigger phrases, tactic tags, calibrated confidence, plain RU/KK reason).
- **Level 2 (analytics):** clusters ~500 synthetic incidents into scam "organizations",
  flags new schemes, and ranks them for an analyst.

## Quick start
> Placeholder — the engineer fills exact commands as modules land. Target shape:

```bash
# 1. Environment
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .                # src-layout: puts `qorgan` on the path
cp .env.example .env            # add GEMINI_API_KEY (data-gen + LLM backend; not needed for the mock demo)

# 2. Build the demo corpus + seed Level-2 incidents
python -m qorgan.data.build_corpus --config configs/corpus.yaml
python scripts/demo_seed.py

# 3. Run the demo (both levels)
streamlit run app/streamlit_app.py
#   ...or:  docker compose up
```

## Evaluate
```bash
python -m qorgan.eval.run --split test --split real_heldout   # FPR-first metric tables
```

## Layout
`src/qorgan/` (data · classifier · explain · analytics · asr · eval) · `app/` (Streamlit) ·
`data/` (taxonomy + corpus + provenance) · `tests/` · `notebooks/` (Colab fine-tune) ·
`docs/` (scope, architecture, decisions).

## Status & limits
1-week prototype. ASR is offline/black-box; mobile, on-device, streaming, speaker-linking
are Phase-1 roadmap. FPR is reported on a separate `real_heldout` set — see
[`data/README.md`](data/README.md).
