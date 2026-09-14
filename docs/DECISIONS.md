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

### D12 — No server-side call audio; no operator interception on the roadmap (2026-09-13)
The served API never accepts audio: the microphone WebSocket (`api_live_ws.py`) and its
streaming client (`site/live_mic.js`) were removed; `GET /api/live/capabilities` states the
invariant. Speech recognition belongs on the citizen's device (PLAN_2026-09 B7 spike);
"listen to live calls through the operator" is off the roadmap. **Rationale:** the council
review — an architecture that streams call content to a server and forwards results to
government analysts is indistinguishable from interception infrastructure regardless of
intent, and it is not fixable by code review, only by design. Enforced by
`tests/test_architecture.py`. **Revisit:** never for audio; on-device ASR when B7 is green.

### D13 — Level 2 has a single ingress: consented reports (2026-09-13)
Nothing on the citizen path (`live/*`, `api_live*`, `api.py::analyze`) may import the
analytics write paths (`analytics.store`, `pipeline`, `cluster`, `intake.ingest_*`);
`/api/analyze` persists nothing. Enforced by `tests/test_architecture.py`. **Rationale:**
same as D12 — the analyst layer must be reachable only through an explicit, reviewable
report. **Revisit:** when the partner intake API lands (PLAN C5) it is the second
*consented* ingress, with a required `consent_basis`, quotas and audit — not a bulk feed.

### D14 — Reports are stored minimised: hashed numbers, scrubbed transcripts, receipts, retention (2026-09-14)
A consented report is persisted only after `reports.store.prepare_report`: the transcript is
PII-scrubbed (`data/scrub.py`), the caller number is reduced to a salted HMAC-SHA256 digest
(`privacy/numbers.py`, key `QORGAN_NUMBER_HMAC_KEY`) plus a coarse display prefix
(`+7 700 ***`), and the record gets an unguessable receipt. `Incident` and `StoredReport`
refuse anything that is not a digest, so a raw number cannot be persisted through the
schema. `DELETE /api/reports/{receipt}` forgets the report everywhere it reached (incident,
embedding cache, recomputed organizations); `python -m qorgan.reports.purge` applies
`QORGAN_REPORT_RETENTION_DAYS`. A server without a key refuses reports that carry a number
(503) rather than storing it raw. **Rationale:** the analyst layer needs to *link* numbers,
never to know them; the council's "one config line from surveillance" risk is answered by
making the raw data unavailable by construction. **Revisit:** key rotation strategy when a
real partner integration lands (linking breaks across keys).

### D17 — `multilingual-e5-base` stays; heads trained on the int8 ONNX embeddings the device ships; threshold 0.59 (2026-09-14)
**Model size.** `multilingual-e5-small` (118 MB int8) was measured, not assumed: same
harness, same threshold — authored recall 1.000 → 0.778, test/ood FPR 0 → 0.019/0.013,
overlapping score margins (lowest scam 0.155 < highest legit 0.205). Kept **e5-base**
(`Xenova/multilingual-e5-base` int8, 278 MB, self-hosted under `site/models/`).
**One graph on both sides (PLAN A4).** The server embeds with the same int8 ONNX graph
(`QORGAN_EMBED_BACKEND=onnx`, `classifier/embed.py::OnnxEmbedder`), **one text per run**:
dynamic int8 quantisation derives activation scales over the whole batched tensor, so a
batched embedding depends on its neighbours (measured cosine down to 0.98). The heads are
trained on those embeddings and the bundle records `embed_backend`; loading under another
backend is refused (`LinearFeatureMismatchError`).
**Operating point.** With int8-trained heads every FPR gate is 0 at **0.59** (the July
tuner's recommendation, previously left as headroom): test 0.000 [0, 0.068] / recall 0.938,
authored 0.000 [0, 0.142] / 1.000, ood 0.000 [0, 0.048] / 0.844 (fp32 heads at 0.55 had
test recall 0.953 and ood 0.889 — the honest cost). Hysteresis pair 0.59 / 0.49.
**Known residual.** A dynamically-quantised graph is not bit-stable across ONNX Runtime
implementations (Python ORT 1.27, onnxruntime-node 1.24, onnxruntime-web): cosine ~0.994
mean / ~0.98 min, tokenisation identical. With int8-trained heads that is decision-safe on
the golden set (0 flips / 28, |Δrisk| ≤ 0.055, tag-set Jaccard 0.977) and is the gate
`tests_js/integration/embedding.test.mjs` enforces. Bit-level parity would need **static
(calibrated) quantisation** — a follow-up, not a blocker. **Revisit:** when a real-call
training set exists (PLAN A8) — retrain, re-tune, re-gate.
