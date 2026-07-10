# DECISIONS.md — resolved open questions (ADR log)

Resolves the "Open technical decisions" from `DOCUMENTATION.md §17` for the 1-week sprint.
Format: **Decision — Rationale — Revisit in Phase 1?**

### D1 — Scope: web-first, mobile out
Cut Android / on-device / streaming ASR / ASR fine-tuning from the sprint.
**Rationale:** ~5–6 build days; the rubric gives 0 points for "mobile" and the on-device
path is the riskiest, lowest-yield work. Web demo wins AI (20) + Explainability (10) +
Prototype (15) + Data (15) without it. **Revisit:** yes — mobile + on-device is Phase 1.

### D2 — Depth: L1-centric
Live transcript → risk → explained alert is the centerpiece; L2 clustering is a lighter
secondary. **Rationale:** L1 most directly demonstrates AI + explainability + UX in a live
demo; L2 kept present to carry the GovTech "why the state cares" story. **Revisit:** in
Phase 1, L2 (per `DOCUMENTATION.md`) becomes the primary value with real report streams.

### D3 — Classifier: hybrid (LLM labels → fine-tune small model)
Gemini generates + labels the corpus; fine-tune **XLM-RoBERTa base** (multi-label,
class-weighted). LLM structured classifier ships **first** as baseline + fallback.
**Rationale:** LLM path guarantees a working demo Day 1 and de-risks the fine-tune;
trained model gives real-ML credibility and an on-device path. **Revisit:** larger/distilled
model, quantization for device in Phase 1.

### D4 — Compute: free/limited Colab
Keep the model small (XLM-R base) and corpus modest so fine-tuning fits a free T4 session.
**Rationale:** stated constraint. **Fallback:** LLM classifier if Colab is unavailable.

### D5 — Explainability: attribution + tags, templated, localized
Token attribution (**Captum** integrated gradients / attention rollout) → trigger spans;
contributing tactic tags with weights; calibrated confidence; **templated RU/KK** reason
strings. **Rationale:** ТЗ requires grounded, non-black-box explanations; templated (not
free-form LLM) keeps them faithful to the model's actual features.

### D6 — Metric: FPR-first
FPR is the primary metric; hard negatives are first-class; probability calibration
(temperature/isotonic); alert hysteresis to avoid flicker. **Rationale:** a
false scam alarm on a real bank call destroys trust in a government tool.

### D7 — L2 signals: text + number graph (no speaker embeddings this week)
Cluster on **BGE-M3** (best KZ) / `multilingual-e5-large` text embeddings via **HDBSCAN**,
overlaid with phone-number co-occurrence; **IsolationForest / distance-to-cluster** novelty.
**Rationale:** speaker embeddings need per-incident audio and add heavy complexity for
little demo gain. **Revisit:** ECAPA-TDNN voice linking in Phase 1.

### D8 — Storage & deploy: SQLite + FAISS, Streamlit, one command
No Postgres/pgvector this week. Single Streamlit app for both levels; `docker compose up`
or `pip install && streamlit run`. **Rationale:** fewer moving parts → reliable
self-deploy (Prototype score). **Revisit:** Postgres + pgvector + FastAPI in Phase 1.

### D9 — Data versioning: seeded build + manifest (no DVC)
Deterministic `build_corpus.py` + a checked-in manifest/hash instead of DVC.
**Rationale:** DVC overhead isn't worth it for a 1-week corpus; reproducibility is what
matters and a seeded build + manifest delivers it.

### D10 — LLM provider: Google Gemini (not Anthropic Claude)
Data generation/labeling and the `llm` classifier backend use **Gemini** via the
`google-genai` SDK in JSON mode (`gemini-2.5-pro` quality / `gemini-2.5-flash` bulk); key
via `GEMINI_API_KEY` (or `GOOGLE_API_KEY`). **Rationale:** the team already has a paid
Gemini plan (no extra cost), and the LLM is only **build-time scaffolding + a temporary
baseline** — the shipped classifier is the offline fine-tuned XLM-R (D3), so the provider
choice does not affect the offline/transparent end goal. The client is dependency-injected;
swapping providers again is a one-file change in `llm_tools.py` + `_default_client`.
