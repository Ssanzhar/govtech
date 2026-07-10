# SCOPE.md — 1-week sprint cut

**This file overrides `DOCUMENTATION.md` for the selection sprint.** `DOCUMENTATION.md`
is the full 10-week vision; below is what we actually build by **2026-07-17 23:59 GMT+5**.

## In scope (build this week)

| # | Deliverable | Serves rubric |
|---|---|---|
| 1 | Labeled KZ/RU/code-switch **scam-text corpus** (synthetic + real anchors + hard negatives), with provenance doc | Data (15) |
| 2 | **Scam-risk classifier** — LLM structured baseline **and** fine-tuned XLM-R; unified interface; FPR-first eval on `test` + `real_heldout` | AI/ML (20) |
| 3 | **Explainability** — token attribution → trigger-phrase highlights + tactic tags + calibrated confidence + localized RU/KK reason + "where it can be wrong" | Explainability (10) |
| 4 | **Streamlit demo (L1 centerpiece):** paste/play transcript → live risk meter → explained alert; hard-negative bank call does not trigger | Prototype (15), UX/demo (5) |
| 5 | **Level 2 (light):** embed ~500 synthetic incidents → HDBSCAN clustering into scam "organizations" + number co-occurrence + novelty/new-scheme flag + priority ranking → analyst panel in the same app | AI/ML (20), Value (15) |
| 6 | **One-command run** (Docker or `streamlit run`) + README + `data/README.md` + 7–10 slides + demo video | Prototype (15), Docs (5) |

## Out of scope this week → Phase 1 (main program) roadmap

- Android / Kotlin app, on-device inference, quantization (TFLite/ONNX/whisper.cpp).
- **Streaming** ASR and any **ASR fine-tuning** (KSC2 telephone-domain adaptation).
- **Speaker embeddings** (ECAPA/pyannote) and voice-based incident linking.
- Postgres + pgvector (use SQLite + FAISS), live report ingest, auth, retraining loop.
- Real government intake integration.

ASR this week = **offline black-box** (faster-whisper + the team's existing Vosk KZ stack)
used only to transcribe a few demo clips. Do not rebuild ASR.

## Risks & fallbacks

| Risk | Mitigation / fallback |
|---|---|
| Colab GPU unavailable / fine-tune stalls | **LLM structured classifier is the shipping fallback** — demo never depends on the trained model landing |
| KZ/RU synthetic data too "clean", high FPR on real anchors | Hard negatives from Day 1; report FPR on `real_heldout` honestly; tune threshold + hysteresis |
| Live demo fails on stage | Pre-recorded **fallback video** of all 3 scenes |
| L2 clustering looks weak on synthetic data | Seed incidents from a few known "scripts" so clusters are visibly meaningful; show novelty flag on an injected new scheme |
| Time overrun | L2 is the first thing to shrink (panel can go read-only/precomputed); L1 + explainability must ship |

## Demo script (3 scenes)

1. **Scam call** (RU or code-switch): risk meter climbs, alert fires with highlighted
   trigger phrases + tactic tags + reason + confidence.
2. **Hard negative** (real bank call / relative asking for money): stays low, no alert →
   proves low FPR.
3. **Analyst view (L2):** a scam "organization" cluster with linked numbers + one
   novelty-flagged new scheme, ranked by priority.
