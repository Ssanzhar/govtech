# CLAUDE.md — Qorğan (GovTech Camp selection sprint)

> Master brief for planner/engineer agents. Read this first, then `docs/SCOPE.md`,
> `docs/ARCHITECTURE.md`, `docs/DECISIONS.md`. `TECHNICAL_TASK.md` is the organizers'
> ask; `DOCUMENTATION.md` is the **full 10-week vision — do NOT try to build all of it.**

## 1. What this is

Qorğan detects **social-engineering (scam) patterns in phone conversations** for
Kazakhstan (Kazakh / Russian / code-switched speech), explains *why* a call looks like
a scam, and — as a secondary view — clusters confirmed reports into scam "organizations"
for a government analyst. It is a **decision-support tool; a human always decides.**

## 2. The hard constraint (read this twice)

- **Deadline: 2026-07-17 23:59 GMT+5.** Today is 2026-07-10 → **~7 calendar days**,
  and ~1 of those goes to the demo video + slide deck + README polish.
- **≈5–6 real build days.** Every decision below exists to fit that. `DOCUMENTATION.md`
  describes a 10-week product; `docs/SCOPE.md` is the cut we actually build.
- Deliverables (organizers, mandatory): **GitHub repo + README + demo video + 7–10 slide deck.**

## 3. Locked decisions (do not relitigate — see `docs/DECISIONS.md`)

1. **Web-first. Mobile / on-device / streaming ASR / ASR fine-tuning are OUT this week**
   (reframed as Phase 1). No Kotlin, no Android.
2. **L1-centric.** The centerpiece is: transcript → risk score → **explained** alert.
   Level 2 (clustering into scam orgs) is a **lighter secondary** that carries the
   "why the government cares" narrative for the pitch.
3. **Classifier = hybrid.** Claude generates + labels the corpus → we **fine-tune a small
   multilingual transformer** (XLM-R base). The **LLM structured classifier ships first**
   as the baseline *and* the fallback, so a working demo exists from Day 1.
4. **Compute = free/limited Colab** → keep the model small and the corpus modest.
5. **FPR (false-positive rate) is the primary metric**, not recall. Hard negatives are
   first-class training data.
6. **Explainability is a hard requirement**, grounded in real features (token attribution
   + tactic tags), templated and localized RU/KK — never free-form LLM prose.
7. **Python-only, one repo, one deployable web demo (Streamlit).**

## 4. Tech stack (finalized)

| Layer | Choice | Notes |
|---|---|---|
| Language | Python 3.11+ | single language |
| Synthetic data + LLM classifier | **Anthropic Claude API** | `claude-sonnet` for quality, `claude-haiku-4-5` for bulk gen/labeling |
| Classifier (trained) | **XLM-RoBERTa base** + multi-label head | class-weighted (low FPR), fits free Colab |
| Calibration | temperature / isotonic (scikit-learn) | calibrated confidence for explainability |
| Attribution | **Captum** integrated gradients / attention rollout | → trigger-phrase spans |
| Text embeddings (L2) | **BGE-M3** (best KZ) or `multilingual-e5-large` | CPU batch over synthetic incidents |
| Clustering (L2) | **HDBSCAN** + phone-number co-occurrence overlay | no k to pick |
| Novelty (L2) | distance-to-nearest-cluster / IsolationForest | "new scheme" flag |
| Storage | **SQLite + FAISS** (in-repo) | no Postgres this week |
| Demo app | **Streamlit** (single app, both levels) | optional thin FastAPI only if needed |
| ASR (offline, black-box) | **faster-whisper** + reuse team's Vosk KZ stack | transcribe demo clips only; DO NOT rebuild ASR |
| Deploy | `docker compose up` **or** `pip install && streamlit run` | judges must self-deploy |

## 5. Repository layout (target)

```
pyproject.toml         src-layout install (`pip install -e .` → `qorgan` importable)
configs/               corpus.yaml + run configs
scripts/               demo_seed.py (seeds ~500 L2 incidents) etc.
models/                exported weights (gitignored)
src/qorgan/
  config.py            constants + thresholds (NO hardcoded values elsewhere)
  taxonomy.py          scam-tactic tags (loads data/taxonomy/tactics.yaml)
  data/                generate.py · label.py · build_corpus.py · schema.py
  classifier/          llm_classifier.py · train.py · calibrate.py · predict.py · attribution.py
  explain/             explainer.py  (attributions+tags → localized templated reason)
  analytics/           embed.py · cluster.py · novelty.py · rank.py   (Level 2, light)
  asr/                 transcribe.py  (faster-whisper/Vosk wrapper, offline)
  eval/                metrics.py · run.py  (FPR-first tables: test + real_heldout)
app/streamlit_app.py   the demo (L1 centerpiece + L2 panel)
data/                  taxonomy/ · raw/ · synthetic/ · processed/ · README.md (provenance)
tests/                 unit tests for deterministic modules
notebooks/             Colab fine-tuning notebook
docs/                  SCOPE.md · ARCHITECTURE.md · DECISIONS.md
```

## 6. Conventions (project-specific; global rules in `~/.claude/rules/` still apply)

- **Immutability, small files (<400 lines), no hardcoded values** — per global coding-style.
- **Two classifiers behind one interface.** `classifier/predict.py` exposes a single
  `score(transcript) -> {risk, tags, attributions}` used by the app and eval, backed by
  either the LLM or the fine-tuned model. Swapping the backend must not touch callers.
- **Explanations must be grounded.** Every trigger phrase shown to a user must be a real
  attributed span from the transcript, tied to a tactic tag with a weight. No hallucinated
  reasons. Templated strings live in `explain/`, localized RU + KK.
- **Report FPR first**, then precision/recall/F1/PR-AUC, always on `test` **and**
  `real_heldout` **separately**. The eval harness (`eval/run.py`) regenerates the tables.
- **Data provenance is graded** (ТЗ §9): every dataset/source, its structure, limits,
  cleaning, and features go in `data/README.md`.

## 7. Sprint adaptations to global rules

- **Testing:** the global 80%-coverage + strict-TDD rule is relaxed for this sprint.
  Deterministic modules (schema, `build_corpus`, `metrics`, cluster utils, explainer
  templating) get real unit tests written test-first. ML/LLM/ASR get smoke tests + the
  eval harness. Do not spend build days chasing coverage on stochastic components.
- **Research-first** still applies for any NEW instrument choice, but the stack in §4 is
  already decided — implement it, don't re-shop.

## 8. Milestone plan (front-loaded to guarantee a working demo)

- **Day 1** Skeleton + `tactics.yaml` taxonomy + data schema. Start LLM synthetic gen.
  Ship the **LLM classifier + minimal Streamlit** → *working transcript→risk→reasons demo today.*
- **Day 2** Finish corpus + LLM labeling (tags + spans) + splits (incl. small `real_heldout`).
  `data/README.md`. Eval harness + FPR baseline on LLM classifier.
- **Day 3** Fine-tune XLM-R on Colab (class-weighted) + calibration + Captum attribution;
  wire behind `predict.py`.
- **Day 4** Explainability polish (localized reasons, confidence, "where it can be wrong",
  human-decides). Full eval tables (LLM + trained) on test + real_heldout. FPR tuning + hysteresis.
- **Day 5** L2 light: embed ~500 synthetic incidents → HDBSCAN + number overlay + novelty +
  ranking → Streamlit analyst panel (cluster map, priority queue, drill-down, new-scheme flag).
- **Day 6** Integration, tests, one-command run + Docker, README, deploy, demo dry-run
  (3 scenes: live scam call; **hard negative = real bank call does NOT trigger**; analyst cluster).
- **Day 7** Demo video + 7–10 slides + final polish. **Submit before 23:59 GMT+5.** Buffer.

## 9. Definition of done (mapped to the 100-pt rubric)

Working `docker compose up` / `streamlit run`; a scam transcript triggers an **explained**
alert; a hard-negative bank call does **not**; FPR-first metric tables on test + real_heldout;
a Level-2 panel showing at least one clustered scam "organization" + a novelty flag;
`data/README.md` provenance; README run instructions; demo video; 7–10 slides.

## 10. Framing note for the pitch (not a scope change)

Program partner is **inDrive**; payment-redirect / fake-operator scams map directly to
ride-hailing fraud — worth one slide. The **government user** is the analyst/law-enforcement
persona seeing scam *organizations* (Level 2); the **citizen** is the Level-1 user. Keep both
personas explicit so Problem/Value/User scores land.
