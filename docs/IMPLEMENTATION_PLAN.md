# IMPLEMENTATION_PLAN.md — Qorğan (GovTech Camp selection sprint)

> **Superseded (2026-09-11) by [`PLAN_2026-09.md`](PLAN_2026-09.md).** Kept as the July sprint record;
> `real_heldout` below is today's `authored_heldout`.

> Execution plan for the engineer agent. Pairs with `CLAUDE.md`, `docs/SCOPE.md`,
> `docs/ARCHITECTURE.md`, `docs/DECISIONS.md`. Deadline **2026-07-17 23:59 GMT+5**.

## Overview

Build a Python-only, web-first prototype: a Kazakh/Russian/code-switched phone transcript →
explained scam-risk alert (Level 1 centerpiece), plus a lighter analyst panel that clusters
~500 synthetic incidents into scam "organizations" with a novelty flag (Level 2). The LLM
structured classifier ships Day 1 as baseline **and** fallback; a fine-tuned XLM-R lands
mid-sprint behind the same interface. FPR is the primary metric; explanations are grounded
(real attributed spans + tactic tags), templated, localized RU/KK.

## Current repo state

Docs + config only (`CLAUDE.md`, `docs/*`, `TECHNICAL_TASK.md`, `DOCUMENTATION.md`,
`data/taxonomy/tactics.yaml`, `data/README.md`, `README.md`, `requirements.txt`,
`.env.example`, `.gitignore`). **No `src/` code yet** — everything under `src/qorgan/`,
`app/`, `tests/`, `notebooks/`, `configs/`, `scripts/` must be created.

## Doc gaps caught during planning (G1–G10) — fixes folded into tasks below

- **G1 Package installability.** src-layout has no `pyproject.toml`; `python -m qorgan...`
  won't resolve. → add `pyproject.toml` (setuptools, `package-dir = {"" = "src"}`), document
  `pip install -e .` (D1-1).
- **G2 Missing dirs.** `configs/`, `scripts/` referenced but absent. → add them.
- **G3 `models/` dir** in `.env` but not in layout/gitignore. → create + gitignore weights.
- **G4 Docker artifacts missing.** `docker compose up` promised, no files. → add `Dockerfile`
  + `docker-compose.yml` (D6-4).
- **G5 L2 incidents under-specified.** L2 needs ~500 *incidents* with phone numbers,
  timestamps, script-family ids — not the same as dialogues. → explicit incident synthesis
  (`scripts/demo_seed.py` + `data/incidents.py`), seed from known families with overlapping
  numbers + inject one novel script (D5-1, D5-5).
- **G6 No storage module** for "SQLite+FAISS". → light `analytics/store.py` (FAISS + optional
  SQLite; JSONL+FAISS is enough for the demo) (D5-2).
- **G7 LLM confidence undefined.** Calibration only fits XLM-R. → `ScoreResult` carries raw
  risk; explainer surfaces calibrated confidence for `xlmr`, labeled "uncalibrated (LLM
  baseline)" for `llm` (D1-4, D4-2).
- **G8 "Risk meter climbs"** implies incremental scoring. → windowing util for cumulative
  scoring; hysteresis governs enter/exit (D1-8, D4-4).
- **G9 Eval cost + live-demo fragility.** → transcript-hash prediction cache in
  `llm_classifier.py`; ship demo clips with cached `ScoreResult`s (D1-5, D6-6).
- **G10 `real_heldout` depends on ASR by Day 2.** → pull `asr/transcribe.py` forward to
  Day 1–2 with a manual-transcript fallback (D1-7, D2-3).

## Interface-first contracts (freeze Day 1)

Defined in `src/qorgan/data/schema.py`, consumed everywhere:

- `classifier/predict.py` → `score(transcript: str) -> ScoreResult`
  `ScoreResult(risk: float, tags: list[TacticTag], attributions: list[Span], backend: str, raw_confidence: float | None)`; backend via `QORGAN_CLASSIFIER_BACKEND` (default `llm`).
- `explain/explainer.py` → `explain(result, transcript, locale) -> Explanation`
  `Explanation(reason, highlights, tags, confidence, caveat, human_note)`.
- `analytics/cluster.py` → `cluster(incidents) -> list[Organization]`
  `Organization(members, numbers[], representative_script, priority, is_novel)`.

Shared models: `Dialogue, Utterance, Label, Span, TacticTag, Incident, ScoreResult,
Explanation, Organization`. Building these first lets the Day-1 explainer + app work against
a stubbed `predict.score()` while real backends land.

## Critical path & fallback

Must-ship spine (also the fallback):
`config.py → schema.py (+contracts) → taxonomy.py → llm_classifier.py → predict.py (llm) →
explainer.py → app/streamlit_app.py (L1)`.

- **Fine-tune (Day 3) is off the critical path** — if Colab stalls, `QORGAN_CLASSIFIER_BACKEND=llm`
  keeps the demo fully functional. Biggest timeline protection.
- **L2 (Day 5) shrinks first** to read-only/precomputed if time is short.
- **Live API is off the demo path** — demo clips ship with cached `ScoreResult`s + a
  pre-recorded fallback video.
- Floor: spine + Day-2 corpus/eval = viable submission (L1 explained alert + hard-negative +
  FPR tables).

---

## Day-by-day tasks

Columns: **ID | Task | Files | Deps | Parallel | Test** (TDD = test-first deterministic; SMOKE = mock/stochastic).

### Day 1 — Skeleton + taxonomy + schema + LLM classifier + minimal Streamlit (working demo today)
| ID | Task | Files | Deps | Parallel | Test |
|---|---|---|---|---|---|
| D1-1 | Package scaffold: `pyproject.toml` (src-layout), dirs (`src/qorgan/`, `app/`, `tests/`, `configs/`, `scripts/`, `notebooks/`, `models/`), gitignore weights (G1,G3) | `pyproject.toml`, `.gitignore`, `__init__.py`s | — | — | — |
| D1-2 | `config.py`: `.env` load, constants — model routing, backend, risk threshold, hysteresis, paths, seeds, locales | `src/qorgan/config.py` | D1-1 | D1-3 | TDD |
| D1-3 | `taxonomy.py`: load+validate `tactics.yaml`, tactics/hard-signal/RU-KK names/negatives | `src/qorgan/taxonomy.py` | D1-1 | D1-2 | TDD |
| D1-4 | Contracts + schema: pydantic models + validators (risk∈[0,1]; spans verbatim substrings); confidence field (G7) | `src/qorgan/data/schema.py` | D1-3 | — | TDD |
| D1-5 | `llm_classifier.py`: Gemini JSON-mode output `{risk,tactic_tags,trigger_spans,confidence}`; schema-validated; transcript-hash cache (G9) | `src/qorgan/classifier/llm_classifier.py` | D1-4 | — | SMOKE |
| D1-6 | `predict.py`: unified `score()`, backend routing; `llm` wired, `xlmr` stub | `src/qorgan/classifier/predict.py` | D1-5 | — | SMOKE |
| D1-7 | `asr/transcribe.py`: faster-whisper offline wrapper + manual fallback (G10) | `src/qorgan/asr/transcribe.py` | D1-1 | D1-5/6 | SMOKE |
| D1-8 | Minimal explainer + windowing: templated reason from tags+spans; `windows()` cumulative scoring (G8) | `src/qorgan/explain/explainer.py`, `explain/windowing.py` | D1-4 | — | TDD |
| D1-9 | Minimal Streamlit L1: transcript → `score()` → meter + highlights + tags + reason; 2–3 demo transcripts | `app/streamlit_app.py` | D1-6, D1-8 | — | SMOKE |
| D1-10 | Synthetic-gen prompt design + first small batch | `src/qorgan/data/generate.py`, `configs/corpus.yaml` | D1-4 | D1-9 | SMOKE |

After D1-4, three parallel tracks: (a) D1-5→D1-6, (b) D1-8, (c) D1-10; D1-7 independent.
**DoD:** `streamlit run app/streamlit_app.py` → transcript → risk → highlighted phrases + tags + reason via LLM backend. Fallback spine complete.

### Day 2 — Corpus + labeling + splits + provenance + eval (FPR baseline)
| ID | Task | Files | Deps | Parallel | Test |
|---|---|---|---|---|---|
| D2-1 | `generate.py` full: seeded Gemini gen across tactics × {KK,RU,mixed} + hard negatives | `data/generate.py`, `configs/corpus.yaml` | D1-10 | D2-2 | SMOKE |
| D2-2 | `label.py`: Gemini tactic tags + verbatim trigger spans; substring validation | `data/label.py` | D1-4 | D2-1 | SMOKE |
| D2-3 | `real_heldout` anchors: transcribe real public scam-baiting clips (+manual fallback), PII scrub | `data/raw/`, `data/processed/` | D1-7, D2-2 | — | SMOKE |
| D2-4 | `build_corpus.py`: assemble, dedupe, PII-scrub, deterministic split + separate `real_heldout`; manifest+hash | `data/build_corpus.py`, `data/processed/manifest.json` | D2-1,2,3 | — | TDD |
| D2-5 | `metrics.py`: FPR (primary), precision/recall/F1, PR-AUC, per-tactic F1; purity/ARI stubs | `eval/metrics.py` | D1-4 | D2-1/2 | TDD |
| D2-6 | `eval/run.py`: tables for `test` **and** `real_heldout` separately; seeds; cached LLM preds | `eval/run.py` | D2-4,5,D1-6 | — | SMOKE |
| D2-7 | Fill `data/README.md` (sources, counts, cleaning, features, limits — graded ТЗ §9) | `data/README.md` | D2-4 | D2-6 | — |

**DoD:** reproducible corpus + manifest; FPR-first tables for LLM classifier on both splits; provenance accurate.

### Day 3 — Fine-tune XLM-R + calibration + attribution (behind `predict.py`)
| ID | Task | Files | Deps | Parallel | Test |
|---|---|---|---|---|---|
| D3-1 | Colab: fine-tune XLM-R base, multi-label head, class-weighted (low FPR); export to `models/` | `notebooks/finetune_xlmr.ipynb`, `classifier/train.py` | D2-4 | — | SMOKE |
| D3-2 | `calibrate.py`: temperature/isotonic on `val` → calibrated confidence (xlmr) | `classifier/calibrate.py` | D3-1 | — | SMOKE |
| D3-3 | `attribution.py`: Captum IG / attention rollout → token→char spans aligned to tags | `classifier/attribution.py` | D3-1 | D3-2 | SMOKE |
| D3-4 | Wire `xlmr` backend into `predict.py`; `llm` stays default; callers untouched | `classifier/predict.py` | D3-1,2,3 | — | SMOKE |

**DoD:** trained model scores via `predict.py` with `xlmr`; LLM fallback intact. If D3-1 stalls, skip D3 entirely.

### Day 4 — Explainability polish + full eval + FPR tuning
| ID | Task | Files | Deps | Parallel | Test |
|---|---|---|---|---|---|
| D4-1 | Localized RU/KK templates: reasons, tag names, "where this can be wrong" caveat, human-decides note | `explain/explainer.py`, `explain/templates_ru.yaml`, `templates_kk.yaml` | D1-8 | D4-2 | TDD |
| D4-2 | Confidence surfacing: calibrated for `xlmr`, labeled "uncalibrated (LLM baseline)" for `llm` (G7) | `explain/explainer.py` | D3-2,D4-1 | D4-1 | TDD |
| D4-3 | Full eval tables: LLM **and** trained, on `test` **and** `real_heldout`; per-tactic + PR-AUC | `eval/run.py`, `docs/eval_report.md` | D3-4,D2-6 | — | SMOKE |
| D4-4 | FPR tuning + hysteresis: threshold min-FPR on `real_heldout`; enter/exit in `config.py`; meter animation | `config.py`, `app/streamlit_app.py` | D4-3,D1-8 | — | TDD |
| D4-5 | Polish L1 app: meter climb, span highlights + tag chips + confidence + caveat + human note; RU/KK toggle | `app/streamlit_app.py` | D4-1,D4-4 | — | SMOKE |

**DoD:** grounded, localized, calibrated/labeled explanation w/ caveat + human note; scam fires, hard-negative bank call does not (verified on `real_heldout`); FPR honest.

### Day 5 — Level 2 (light): incidents → clusters → number graph → novelty → panel
| ID | Task | Files | Deps | Parallel | Test |
|---|---|---|---|---|---|
| D5-1 | Incident synthesis: ~500 incidents from a few script families w/ overlapping numbers + timestamps; inject 1 novel script (G5) | `scripts/demo_seed.py`, `data/incidents.py` | D2-4 | — | TDD |
| D5-2 | `store.py` + `embed.py`: BGE-M3 (or e5) embeddings → FAISS; optional SQLite metadata (G6) | `analytics/store.py`, `analytics/embed.py` | D5-1 | D5-3 | SMOKE |
| D5-3 | `cluster.py`: HDBSCAN + phone-number co-occurrence overlay (networkx) → `Organization`s | `analytics/cluster.py` | D5-2 | — | TDD |
| D5-4 | `rank.py`: priority = f(size, recency, growth); prioritized queue | `analytics/rank.py` | D5-3 | D5-5 | TDD |
| D5-5 | `novelty.py`: distance-to-cluster / IsolationForest → `is_novel`; verify flags injected novel script | `analytics/novelty.py` | D5-3 | D5-4 | TDD |
| D5-6 | Streamlit L2 tab: cluster map, priority queue, drill-down, novelty flag; reads precomputed (degrade-ready) | `app/streamlit_app.py` | D5-3,4,5 | — | SMOKE |
| D5-7 | Cluster metrics: purity/ARI on labeled script families | `eval/metrics.py`, `docs/eval_report.md` | D5-3 | D5-6 | TDD |

**DoD:** panel shows ≥1 meaningful "organization" (linked numbers), priority queue, drill-down, novelty-flagged new scheme; purity/ARI reported.

### Day 6 — Integration, tests, one-command run, Docker, README, deploy, dry-run
| ID | Task | Files | Deps | Parallel | Test |
|---|---|---|---|---|---|
| D6-1 | Fill test gaps green on deterministic modules; smoke suite for ML/LLM/ASR | `tests/**` | all | — | run |
| D6-2 | E2E wiring: config-only backend swap; no hardcoded values; immutability review | `src/qorgan/**` | D6-1 | — | run |
| D6-3 | One-command run + finalize `configs/corpus.yaml`; verify build_corpus + demo_seed + streamlit chain | `scripts/`, `README.md` | D6-2 | D6-4 | SMOKE |
| D6-4 | `Dockerfile` + `docker-compose.yml`; verify `docker compose up` (G4) | `Dockerfile`, `docker-compose.yml` | D6-3 | D6-3 | manual |
| D6-5 | README: exact run/deploy, both launch paths, eval commands, limits | `README.md` | D6-3,4 | — | — |
| D6-6 | Demo prep: transcribe/cache 3 scenes; cache `ScoreResult`s (no live API, G9); dry-run | `data/`, `scripts/`, `app/` | D6-2 | — | manual |

**DoD:** clean checkout deploys via `pip install -e . && streamlit run` **and** `docker compose up`; tests pass; 3-scene dry-run from cache.

### Day 7 — Demo video + slides + submit (buffer)
| ID | Task | Deps |
|---|---|---|
| D7-1 | Record demo video (3 scenes) + pre-recorded fallback video | D6-6 |
| D7-2 | 7–10 slides: problem, users (analyst + citizen), solution, data, AI/ML, explainability, limits, next steps, inDrive framing | D6-5 |
| D7-3 | Final polish + checklist + submit before 23:59 GMT+5 | D7-1,2 |

---

## TDD vs smoke policy

**TDD (test-first):** schema validators · build_corpus (dedup/split/manifest) · metrics · taxonomy load · explainer templating+locale+caveat · windowing · hysteresis · cluster number-graph merge · rank ordering · novelty threshold · incident seeding.
**SMOKE (mock LLM/model):** llm_classifier · generate · label · train/notebook · attribution · embed · predict routing · transcribe · Streamlit render.
Do not chase 80% coverage on stochastic components (CLAUDE §7); the eval harness is the quality gate there.

## Risks & mitigations
- Colab stalls → default `llm` backend; drop D3, demo unaffected. *(primary protection)*
- High FPR on real anchors → hard negatives Day 2; threshold on `real_heldout`; hysteresis; honest reporting.
- Weak L2 clusters → seed from known families w/ shared numbers; inject novel script; report purity/ARI.
- Live-demo failure → cached `ScoreResult`s + fallback video.
- Time overrun → L2 shrinks to read-only first; L1 + explainability are the floor.
- Eval API cost → transcript-hash cache.
- Won't import on judges' machines → `pyproject.toml` + `pip install -e .` verified on clean checkout (D6-4).

## Final submission checklist (100-pt rubric)
- [ ] **Problem (15):** concrete scam problem, both personas, inDrive framing slide.
- [ ] **Value (15):** L2 organizations + priority queue + novelty → "why the state cares"; quantified analyst benefit.
- [ ] **Data (15):** `data/README.md` complete; deterministic build + manifest; hard negatives + `real_heldout`.
- [ ] **AI/ML (20):** hybrid classifier behind one interface; FPR-first tables on both splits; L2 embeddings+HDBSCAN+number graph+novelty+ranking w/ purity/ARI.
- [ ] **Explainability (10):** grounded spans + tags + calibrated/labeled confidence + RU/KK reason + caveat + human note; no free-form prose.
- [ ] **Prototype/deploy (15):** both launch paths verified on clean checkout.
- [ ] **UX/demo (5):** 3-scene demo; animated meter; clear alert UI.
- [ ] **Docs (5):** README, clean structure, 7–10 slides.
- [ ] **Artifacts:** repo · README · demo video (+fallback) · slides · submitted before 2026-07-17 23:59 GMT+5.
