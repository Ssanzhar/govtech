# Project Status & Handoff — Qorğan

_Last updated: 2026-07-13. This is the **current-state** doc for anyone picking the project up.
Read this first, then `CLAUDE.md` (the brief + locked decisions) and `docs/eval_report.md`
(the model numbers)._

## TL;DR
- Working **web prototype** (Streamlit): **L1** = transcript → scam risk → grounded, localized
  (RU/KK) explanation; **L2** = cluster confirmed incidents into scam "organizations" + novelty.
- Shipped classifier: the **`linear` hybrid** model — frozen `multilingual-e5-base` embedding
  **⊕ 6 interpretable features** (5 hard-signal request cues + 1 reassurance) → calibrated
  Logistic Regression. Offline, CPU, retrains in seconds, ~64 KB export.
- **518 tests green.** All work is **committed on `sanzhs-branch`** (5 commits ahead of `main`,
  **not yet merged**). Only `hf_upload.py` is untracked.
- **Trained weights and corpus splits are gitignored** → they live on **Hugging Face** (below)
  or are **regenerated** from the committed code + lexicons + augmentation.

## Repository map (what to read)
| Path | What |
|---|---|
| `CLAUDE.md` | Master brief, locked decisions, conventions. Auto-loaded by Claude agents. |
| `docs/SCOPE.md` · `ARCHITECTURE.md` · `DECISIONS.md` | Scope cut, system design, decision log. |
| `docs/eval_report.md` | **Model eval + the FPR ablation** (baseline vs hybrid). Source of truth for numbers. |
| `data/README.md` | Data provenance, PII scrubbing, augmentation (graded, ТЗ §9). |
| `src/qorgan/classifier/` | The classifier (see below). |
| `src/qorgan/analytics/` | Level-2: clustering, novelty, ranking. |
| `src/qorgan/data/` | Corpus generation, labeling, scrub, build, L2 incident synthesis. |
| `app/streamlit_app.py` | The demo (both levels). |
| `scripts/` | `augment_reassurance_negatives.py` (Gemini), `demo_seed.py` (L2 incidents). |
| `hf_upload.py` | Publishes models + dataset to Hugging Face (untracked; run manually). |

## Classifier backends — `classifier/predict.py :: score(transcript, backend=...)`
One interface, backend chosen by `QORGAN_CLASSIFIER_BACKEND` (default in `.env`: `linear`):
- **`linear`** — the SHIPPED offline model. Hybrid risk head; embedding-only per-tactic head.
  Weights in `models/linear/` (gitignored). Loads the cue lexicon + reassurance patterns from
  `data/lexicon/*.yaml` (committed) and **hash-validates** them (`metadata.json`).
- **`mock`** — deterministic keyword safety net, zero deps/keys. The app auto-degrades to it if
  no model/key is present.
- **`llm`** — Gemini structured classifier; needs `GEMINI_API_KEY`.
- **`xlmr`** — **ABANDONED** (fine-tune collapsed to the class prior). Code path is tested but
  the real weights (`models/xlmr/`, ~1.1 GB) are unusable — safe to delete.

### The hybrid feature (why the model looks the way it does)
The embedding-only model false-positived on realistic legit RU bank/telecom calls
(`real_heldout` FPR 0.143). Root cause = a **train/reality gap**: synthetic legit calls never
_reassure_ ("we'll never ask for your code"), a pattern in 0/307 training negatives. Fix =
**data + a reassurance feature together** (neither works alone):
- `data/augment/reassurance_negatives.jsonl` (27, committed) → added to **train only**.
- `classifier/reassurance.py` + `data/lexicon/reassurance_patterns.yaml` → the anti-scam feature.
- `classifier/cue_lexicon.py` + `data/lexicon/hard_signal_cues.yaml` → the 5 request-cue features
  (also drive grounded highlights).

**Shipped eval (real harness, alert threshold 0.55):** test FPR 0 / recall 0.953 ·
**real_heldout FPR 0 / recall 1.000** (was 0.143 / 0.923) · ood FPR 0 / recall 0.867. Full
ablation + honest caveats in `docs/eval_report.md`. Rollback bundle: `models/linear_embed_only/`
(flag off; `metadata.json` has no `hard_signal_enabled`).

## Data pipeline (`src/qorgan/data/`)
`generate.py` (Gemini synthetic corpus) → `label.py` (independent re-label, optional) →
`build_corpus.py` (**scrub PII** → dedup → deterministic seeded split into train/val/test;
folds `data/augment/*.jsonl` into **train only**; `real_heldout` = curated anchors, kept
separate) → `data/processed/{split}.jsonl` + `manifest.json`.
- **PII:** `data/processed/*` splits and `data/augment/*` are scrubbed. `data/synthetic/*` (raw)
  is **NOT** — never publish it. `data/processed/{incidents,organizations}.jsonl` (L2) contain
  fabricated numbers and predate the scrub fix — don't publish them either.
- `real_heldout` (hand-written, held-out source) is the **honest** generalization signal.

## Hugging Face repos
- **Models:** `sanzh-ts/govtech` — root = shipped hybrid bundle; `embed_only/` = baseline;
  `lexicon/` = the cue/reassurance yamls the bundle validates.
- **Dataset:** `sanzh-ts/govtech_ds` — scrubbed splits (`train/val/test/real_heldout/ood.jsonl`
  + `manifest.json`) and `augment/`.

**Publish** (write token): `hf auth login` then `python hf_upload.py`.

**Pull the trained model** (instead of retraining):
```python
from huggingface_hub import snapshot_download
snapshot_download("sanzh-ts/govtech", local_dir="models/linear_hf")
# then: export QORGAN_LINEAR_MODEL_DIR=models/linear_hf/  (lexicons already committed in data/lexicon/)
```

## How to run
```bash
pip install -e .                      # src-layout install -> `qorgan` importable (needs Python 3.11+)
streamlit run app/streamlit_app.py    # the demo (mock backend if no model/key)

# eval (FPR-first tables on the honest splits)
QORGAN_CLASSIFIER_BACKEND=linear python -m qorgan.eval.run --split test --split real_heldout --split ood --by-language

# regenerate everything from source (needs GEMINI_API_KEY for generation steps)
python -m qorgan.data.generate                         # raw synthetic corpus (or pull from HF dataset)
python scripts/augment_reassurance_negatives.py --per-cell 8   # reassurance negatives (already committed)
python -m qorgan.data.build_corpus                     # scrub + dedup + split (+augment) + manifest
python -m qorgan.classifier.linear_train               # train + export the hybrid model (default)

# Level 2
python scripts/demo_seed.py && python -m qorgan.analytics.pipeline

pytest -q                             # 518 tests
```
Alert threshold is `QORGAN_RISK_THRESHOLD` (`.env` + `config.py` default = **0.55**, tuned for the
hybrid model by `eval/threshold.py`).

## Known issues / open threads (for your own planning)
- **`ood` recall cost:** the FPR fix trades ~4 pts of recall on the disfluent/ASR stress set
  (0.911→0.867). Fine per the FPR-first mandate, but a lever to revisit.
- **Mild test-set circularity:** the reassurance matcher was first motivated by the 2 `real_heldout`
  FPs. Mitigated (generalizes on independent + fresh un-inspected sets) but noted honestly.
- **No `legit_telecom` negative category** in the taxonomy, though a telecom call was an FP; the
  fix generalized, but a dedicated category would be cleaner.
- **`models/xlmr/` (~1.1 GB) is dead weight** — abandoned fine-tune, safe to `rm -rf`.
- **`error_log.txt`** at repo root is a stray debug dump (gitignored) — can delete.
- **Not merged to `main`** — the work is on `sanzhs-branch`.
- Persistent working memory for agents lives in
  `~/.claude/projects/-Users-sanzhars-Documents-proga--govtech/memory/` (indexed by `MEMORY.md`).
