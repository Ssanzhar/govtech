# ARCHITECTURE.md — lean 1-week design

Web-first, Python-only, L1-centric. This supersedes the two-subsystem/on-device design in
`DOCUMENTATION.md` for the sprint.

> **September 2026 layer (current).** Level 1 runs on the citizen's device (`site/core/`,
> the JS port of the classifier + `Xenova/multilingual-e5-base` int8 in a Web Worker); the
> FastAPI server (`qorgan.api`) hosts the site, a stateless `/api/analyze` fallback, and
> the analyst layer. The target diagram is `PLAN_2026-09.md §2`. Its invariants are
> **tests**, not prose (`tests/test_architecture.py`):
>
> 1. **No route accepts audio** — no WebSocket, no audio/multipart bodies (ADR D12).
> 2. **`/api/analyze` persists nothing** (snapshot of `data/` before/after).
> 3. **Level 2 has only consented ingresses** — `POST /api/reports` (citizen, ADR D13)
>    and `POST /api/v1/reports` (partner, ADR D19). Every module on the citizen/partner
>    path (`live/*`, `api.py`, `api_live*`, `api_reports*`, `api_partner.py`) is
>    import-graph-checked against the analytics write paths.
> 4. **Stored minimised** — numbers only as salted HMAC digests + `+7 700 ***`, transcripts
>    PII-scrubbed, receipts, `DELETE`, retention purge (ADR D14; schema-enforced).
> 5. **The partner ingress is not a bulk feed** — API key per partner, one report per
>    request, structured tactic hits preferred, transcripts accepted only pre-scrubbed,
>    `consent_basis` required, per-partner rate limit + rolling daily quota, content-free
>    audit log (`data/processed/audit_log.jsonl`), aggregates-only export.
>
> Modules: `api_reports.py` · `api_partner.py` + `api_partner_export.py` · `partners.py`
> (credential registry) · `audit.py` · `reports/` (store, purge, partner views) ·
> `privacy/numbers.py`. The rest of this file is the July design and still describes the
> classifier, explainer and Level-2 internals.

## Flow

```
                         ┌─────────────── Level 1 (centerpiece) ───────────────┐
 demo audio ─(offline)─▶ ASR ─▶ transcript ─▶ Classifier ─▶ Explainer ─▶ Streamlit alert
   (or pasted text) ─────────────────────────▲    │              (grounded reasons,
                                              │    │               tags, confidence)
                              LLM backend  ◀──┴────┴──▶  fine-tuned XLM-R backend
                              (baseline/fallback)        (Colab, class-weighted)
                                              │
                    ┌───── Level 2 (light) ───┴──────────────────────────────┐
  synthetic         │  embed (BGE-M3/e5) ─▶ HDBSCAN + number-graph ─▶ ranking │
  incidents (~500) ─▶                     └▶ novelty (new scheme) ────────────▶ analyst panel
                    └──────────────────────────────────────────────────────────┘
                              storage: SQLite + FAISS (in-repo)
```

## Components & responsibilities

### `src/qorgan/data/`
- `schema.py` — pydantic models: `Dialogue`, `Utterance`, `Label` (risk, `tactic_tags[]`,
  `trigger_spans[]`), `Incident`. Validate at every boundary.
- `generate.py` — Gemini generates KZ/RU/code-switch dialogues per tactic + variations,
  and **hard negatives** (bank call, relative asking for money, legit gov service).
  Deterministic (seeded), config-driven.
- `label.py` — Gemini labels each dialogue with tactic tags + **verbatim** trigger spans
  (spans must be substrings of the transcript — validate).
- `build_corpus.py` — assemble, dedupe, PII-scrub, split `train/val/test` + hold out a
  separate small `authored_heldout` (transcribed real anchors). Writes a manifest + hash.

### `src/qorgan/classifier/`
- `llm_classifier.py` — Gemini JSON-mode output → `{risk, tactic_tags, trigger_spans}`.
  Ships first; baseline + fallback.
- `train.py` — fine-tune XLM-R base, multi-label head, **class weighting for low FPR**.
- `calibrate.py` — temperature/isotonic → calibrated `confidence`.
- `attribution.py` — Captum integrated gradients / attention rollout → token spans.
- `predict.py` — **single interface** `score(transcript) -> ScoreResult` selecting backend
  via config. All callers depend on this, not on a specific model.

### `src/qorgan/explain/`
- `explainer.py` — turns `ScoreResult` into a UI-ready explanation: highlighted spans,
  contributing tags + weights, calibrated confidence, a **templated localized (RU/KK)**
  reason string, a "where this can be wrong" caveat, and the human-decides note. No
  free-form LLM generation in the explanation path.

### `src/qorgan/analytics/` (Level 2, light)
- `embed.py` — BGE-M3 (or e5) transcript embeddings, cached to FAISS.
- `cluster.py` — HDBSCAN over embeddings; overlay phone-number co-occurrence edges to
  merge/annotate clusters ("organizations").
- `novelty.py` — distance-to-nearest-cluster / IsolationForest → `is_novel` new-scheme flag.
- `rank.py` — priority = f(size, recency, growth). Prioritized queue for the analyst.

### `src/qorgan/eval/`
- `metrics.py` — FPR (primary), precision, recall, F1, PR-AUC, per-tactic F1;
  cluster purity/ARI on labeled synthetic groups.
- `run.py` — regenerates metric tables for `test` and `authored_heldout` **separately**;
  fixed seeds; logs configs.

### `src/qorgan/asr/`
- `transcribe.py` — faster-whisper / Vosk wrapper, **offline batch only**, for demo clips.

### `app/streamlit_app.py`
- Tab 1 (L1 centerpiece): input transcript (paste or pick a demo clip) → risk meter →
  explained alert with highlighted spans.
- Tab 2 (L2 panel): cluster map, priority queue, incident drill-down, novelty flag.

## Interfaces (stable contracts)

```python
# classifier/predict.py
def score(transcript: str) -> ScoreResult:
    """ScoreResult(risk: float, tags: list[TacticTag], attributions: list[Span])."""

# explain/explainer.py
def explain(result: ScoreResult, transcript: str, locale: str) -> Explanation:
    """Explanation(reason: str, highlights: list[Span], tags: list[...], confidence: float,
                   caveat: str)."""

# analytics/cluster.py
def cluster(incidents: list[Incident]) -> list[Organization]:
    """Groups incidents; each Organization has members, numbers[], representative_script,
       priority, is_novel."""
```

## Non-negotiable properties
- Deterministic corpus build (seeds + manifest) → reproducible metrics.
- Explanations grounded in attributed spans; localized RU/KK; never hallucinated.
- FPR reported on `test` **and** `authored_heldout` separately.
- Nothing "sends" or "decides" automatically — human-in-the-loop by construction.
