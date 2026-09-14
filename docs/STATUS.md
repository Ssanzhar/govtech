# Project Status & Handoff — Qorğan

_Last updated: 2026-07-15. This is the **current-state** doc for anyone picking the project up.
Read this first, then `CLAUDE.md` (the brief + locked decisions) and `docs/eval_report.md`
(the model numbers). Run/usage instructions for humans are in the top-level `README.md`._

## TL;DR
- Working **web prototype** (Streamlit), three tabs: **L1** transcript → risk → grounded RU/KK
  explanation · **Live call** real-time meter over streaming utterances (script replay or real
  microphone) · **L2 analyst** dashboard of scam "organizations" fed by citizen reports.
- Shipped classifier: the **`linear` hybrid** — frozen `multilingual-e5-base` embedding
  **⊕ 6 interpretable features** (5 hard-signal request cues + 1 reassurance) → calibrated
  Logistic Regression. Offline, CPU, retrains in seconds, ~64 KB export.
- **Shipped eval (threshold 0.55):** test FPR 0.000/rec 0.953 · **authored_heldout FPR 0.000/rec
  1.000** (42 hand-written anchors) · ood FPR 0.000/rec 0.889. Streaming: false-latch 0.167
  heldout / 0.115 test (`python -m qorgan.eval.stream`).
- **697 tests green**, all offline. Work is on **`sanzhs-branch`**.
- **Trained weights and corpus splits are gitignored** → pull from **Hugging Face** (below) or
  regenerate from committed code + lexicons + augmentation.

## What landed 2026-07-14 → 15 (the sprint log)

**1 · Live citizen pipeline** (from the real-time design spec / `task.md`): committed
utterances → head+tail rolling window → `score()` → **0–100 suspicion meter** (asymmetric EMA,
hard-signal floors 61/81, 55/45 hysteresis latch; `live/meter.py`, `live/session.py`) →
tactic→advice engine (`explain/recommend.py` + `advice_{ru,kk}.yaml`) → post-call summary +
**consent-gated editable report** (`live/summary.py` → `data/processed/citizen_reports.jsonl`).
Meter knobs: `QORGAN_METER_ALPHA_UP/_DOWN`. UI: `app/live_view.py`.

**2 · Live microphone**: Input radio in the Live tab — *Replay script* (default, zero-setup) ·
*Microphone — browser* (streamlit-webrtc) · *Microphone — local* (sounddevice). ASR = **dual
Vosk KK+RU streaming recognizers, per-utterance word-confidence voting** (`asr/vosk_stream.py`;
small models auto-download to `~/.cache/vosk`, config `QORGAN_VOSK_MODEL_KK/_RU`). Plumbing:
`asr/capture.py` (drop-oldest queue); wiring: `app/mic_live.py` (worker threads → event queue →
`st.fragment` drain; threads never touch Streamlit). Optional extra `pip install -e ".[live]"`,
graceful degrade without it. Verified end-to-end with synthesized speech (macOS `say`, RU+KK)
through real Vosk + the linear model: scam scene → 84/Critical/latched; hard negative stays Low.
That e2e also fixed `predict._merge_cue_evidence`: a verbatim cue hit now upgrades the tactic
tag to weight 1.0 (was suppressed by the head's lower weight → hard-signal floor never fired).

**3 · Model-improvement sprint (all FPR-gated, one attempt rolled back)**:
- `authored_heldout` widened 27→**42** anchors → immediately exposed a real KK/mixed FP (legit
  tariff notice @ 0.809).
- New **streaming eval harness** `python -m qorgan.eval.stream`: false-latch rate (live FPR
  analog), alert-hit rate, time-to-alert.
- **KK boundary stabilized**: 15 hand-written KK legit hard negatives
  (`data/augment/kk_legit_negatives.jsonl`, train-only, leakage-tested vs anchors) + KK
  reassurance terms (`қажеті жоқ` …) + payment sensitive-terms. A lexicon-only attempt FAILED
  gates (the joint-LR refit moved unrelated KK negatives; test FPR 0→0.019) and was rolled
  back — **pair lexicon edits with training data**, that combination passed.
- **Cue-lexicon ASR variants** ("безопасной счёт", "код из сообщения", KK bare imperatives) —
  the live hard-signal floor now fires on real Vosk output.
- Full ablation + honest caveats: `docs/eval_report.md` **2026-07-15 addendum**.

**4 · Analyst dashboard overhaul + citizen-report loop**: L2 extracted to `app/analyst_view.py`
— KPI row, novel-scheme callout, priority queue with **tactic-derived display names**
(`analytics/presentation.py`; never raw cluster ids), drill-down with tactic/activity charts
and span-highlighted representative script. **Report loop closed**: Live-tab reports →
`analytics/intake.py` (idempotent content-addressed ids; embeds ONLY new transcripts via the
`incident_embeddings.npz` cache `analytics/pipeline.py` now writes) → **"Ingest into
analysis"** button → number-graph placement (known number joins that org; unknown = novelty
candidate). Verified live: a real report joined the seeded `bank_security` org in ~9 s;
re-ingest is a no-op. Intake normalizes tz-aware report timestamps to the store's naive
convention (mixing crashed priority ranking).

## Repository map (what to read)
| Path | What |
|---|---|
| `CLAUDE.md` | Master brief, locked decisions, conventions. Auto-loaded by Claude agents. |
| `docs/SCOPE.md` · `ARCHITECTURE.md` · `DECISIONS.md` | Scope cut, system design, decision log. |
| `docs/eval_report.md` | **Model eval + FPR ablations** (incl. 2026-07-15 addendum). Source of truth for numbers. |
| `data/README.md` | Data + ASR-model provenance, PII scrubbing, augmentation (graded, ТЗ §9). |
| `src/qorgan/classifier/` | The classifier: predict interface, features, lexicon matchers, training. |
| `src/qorgan/live/` | Suspicion meter, live session, post-call summary/report. |
| `src/qorgan/asr/` | Batch whisper wrapper · replay/pseudo-stream · Vosk dual-stream · capture queues. |
| `src/qorgan/explain/` | Explainer + templates + the tactic→advice recommendation engine. |
| `src/qorgan/analytics/` | Level-2: clustering, novelty, ranking, presentation helpers, report intake. |
| `src/qorgan/eval/` | FPR-first tables (`run`), threshold tuner, **streaming eval** (`stream`). |
| `app/` | `streamlit_app.py` (shell + L1) · `live_view.py` · `mic_live.py` · `analyst_view.py` · `ui_shared.py`. |
| `scripts/` | `demo_seed.py` (L2 incidents) · `augment_reassurance_negatives.py` (Gemini) · `hf_upload.py`. |

## Classifier backends — `classifier/predict.py :: score(transcript, backend=...)`
One interface, backend chosen by `QORGAN_CLASSIFIER_BACKEND` (default `linear`):
- **`linear`** — the SHIPPED offline model. Hybrid risk head; embedding-only per-tactic head
  (verbatim cue hits upgrade tags to weight 1.0). Weights in `models/linear/` (gitignored);
  **hash-validates** `data/lexicon/*.yaml` — any lexicon edit forces a retrain, and per the
  sprint lesson should be **paired with training data** (see sprint log §3).
- **`mock`** — deterministic keyword safety net, zero deps/keys; the app auto-degrades to it.
- **`llm`** — Gemini structured classifier; needs `GEMINI_API_KEY`.
- **`xlmr`** — **ABANDONED** (fine-tune collapsed). `models/xlmr/` (~1.1 GB) is deletable.

### The hybrid feature (why the model looks the way it does)
The embedding-only model false-positived on realistic legit RU calls (`authored_heldout` FPR
0.143). Root cause = a **train/reality gap**: synthetic legit calls never _reassure_ ("we'll
never ask for your code"). Fix = **data + a reassurance feature together** (neither works
alone): `data/augment/reassurance_negatives.jsonl` (train-only) + `classifier/reassurance.py`
+ the request-cue features (`classifier/cue_lexicon.py`). The 2026-07-15 sprint repeated the
same pattern for **Kazakh** (KK legit negatives + KK reassurance terms). Rollback bundle:
`models/linear_embed_only/`.

## Data pipeline (`src/qorgan/data/`)
`generate.py` (Gemini synthetic corpus) → `build_corpus.py` (**scrub PII** → dedup →
deterministic split; folds `data/augment/*.jsonl` — reassurance + KK-legit negatives — into
**train only**; `authored_heldout` = 42 curated anchors, kept separate) →
`data/processed/{split}.jsonl` + `manifest.json`.
- **PII / publishing:** `data/processed/` splits and `data/augment/` are scrubbed →
  publishable. `data/synthetic/` (raw) is **NOT**. `data/processed/{incidents,organizations,
  citizen_reports}.jsonl` contain fabricated numbers / live report intake — **never publish**
  (the `hf_upload.py` allow-patterns already exclude them; don't widen them).

## Hugging Face repos
- **Models:** `sanzh-ts/govtech` — root = shipped hybrid bundle; `embed_only/` = baseline;
  `lexicon/` = the cue/reassurance yamls the bundle validates. **Model and lexicons must be
  uploaded together** (hash guard).
- **Dataset:** `sanzh-ts/govtech_ds` — scrubbed splits + `augment/`.
- **Publish** (write token): `hf auth login` then `python scripts/hf_upload.py`.
- **Pull instead of retraining:** `snapshot_download("sanzh-ts/govtech", local_dir="models/linear_hf")`
  then `export QORGAN_LINEAR_MODEL_DIR=models/linear_hf` (lexicons are committed in-repo).

## How to run
```bash
pip install -e .                      # Python 3.11+; add ".[live]" for microphone modes
streamlit run app/streamlit_app.py    # 3 tabs; mock backend if no model/key

# eval — full-transcript FPR tables + streaming false-latch/time-to-alert
QORGAN_CLASSIFIER_BACKEND=linear python -m qorgan.eval.run --split test --split authored_heldout --split ood --by-language
QORGAN_CLASSIFIER_BACKEND=linear python -m qorgan.eval.stream --split test --split authored_heldout --backend linear

# retrain from committed corpus (seconds, CPU; needed after ANY data/lexicon change)
python -m qorgan.data.build_corpus && python -m qorgan.classifier.linear_train

# Level 2: seed demo incidents -> organizations + embedding cache (needed for report ingest)
python scripts/demo_seed.py && python -m qorgan.analytics.pipeline

pytest -q                             # 697 tests, all offline
```
Alert threshold `QORGAN_RISK_THRESHOLD` = **0.55** (tuner now recommends 0.59–0.62 with equal
recall — headroom, deliberately not taken). Demo storyline (3 scenes) is in `README.md`.

## Known issues / open threads (for your own planning)
- **Transient streaming false-latch ~0.167**: ~1 in 6 legit calls latches the live meter
  mid-call, then recovers (full-transcript FPR is 0). Structural: early short windows are
  noisy. Candidates: minimum-turns before the latch arms, short-window damping. Now
  measurable via `eval.stream` — tune against it.
- **Plan Phase 3 not done**: Gemini tail-tactic data top-up (`mule_recruitment` has only ~12
  train examples). Needs `GEMINI_API_KEY`; was the designated drop candidate.
- **Mild eval circularity, twice**: both the reassurance fix (2026-07-13) and the KK fix
  (2026-07-15) were motivated by inspected heldout FPs. Mitigated (fresh un-inspected sets,
  leakage tests) but keep honest in the pitch.
- **No `legit_telecom` negative category in the taxonomy** — covered at the data level
  (anchors + KK augment), a dedicated category would still be cleaner.
- **Vosk KK small-model accuracy** on far-field/synthetic speech is the live-mode bottleneck;
  meter confidence-weighting absorbs some of it.
- **Browser-mic mode** cannot run under AppTest (webrtc needs the real runtime) — one manual
  click-through before demoing.
- `models/xlmr/` (~1.1 GB) and `error_log.txt` are dead weight — safe to delete.
- Persistent working memory for agents lives in
  `~/.claude/projects/-Users-sanzhars-Documents-proga--govtech/memory/` (indexed by `MEMORY.md`).
