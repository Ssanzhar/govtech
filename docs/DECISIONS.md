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
implementations (Python ORT 1.27, onnxruntime-node 1.21/1.24, onnxruntime-web): cosine
0.985–0.994 mean / 0.97–0.98 min, tokenisation identical. With int8-trained heads that is
decision-safe on the golden set (0 flips / 28 on every runtime tried) but not tag-stable
(|Δrisk| up to 0.11, tag-set Jaccard 0.94–0.98); `tests_js/integration/embedding.test.mjs`
gates exactly that. Static (calibrated) quantisation was the candidate fix — measured and
rejected in D18. **Revisit:** when a real-call training set exists (PLAN A8) — retrain,
re-tune, re-gate.

### D18 — Static int8 quantisation rejected; the cross-runtime residual is accepted and gated at decision level (2026-09-17)
PLAN B8 hypothesised that the Node/Python/web drift of D17 came from *run-time* activation
scales (`DynamicQuantizeLinear`) and that a statically calibrated QDQ graph would make every
runtime do the same integer arithmetic. `scripts/quantize_embedder.py` built one (MinMax
moving-average, 128 train texts, MatMul + Gather quantised, the 24 activation-activation
MatMuls left fp32, 278 MB) and it was measured on 88 transcripts (28 golden + 60 test):

| graph | cosine vs fp32 (Python ORT) | Node ORT 1.21 vs Python ORT 1.27 |
|---|---|---|
| dynamic int8 (Hub, shipped) | 0.992 mean / 0.981 min | 0.985 mean / 0.970 min |
| static int8 (MinMax-MA) | **0.944 mean / 0.924 min** | 0.992 mean / **0.971 min** |

Two findings: (1) static per-tensor uint8 activations cost 5 pts of fidelity — more error
than the drift it was meant to remove (outlier hidden-state dimensions, the known BERT-family
problem); (2) the cross-runtime floor barely moved (0.971 vs 0.970 min), so the residual is
kernel/fusion differences between ONNX Runtime implementations, not scale computation, and
no calibration method can remove it. **Decision:** the dynamic Hub graph stays the single
artefact on both sides; the guarantee is the decision-level gate in
`tests_js/integration/embedding.test.mjs` (0 flips, |Δrisk| < 0.15, tag Jaccard ≥ 0.9) and
the eval report states it. The browser now loads the graph the server is configured with
(`qorgan-config.json::embedder`, derived from `QORGAN_EMBED_ONNX_DIR`), so the two can never
silently diverge. The script and the drift probes (`scripts/embed_probe.py`,
`tests_js/tools/runtime_drift.mjs`) stay as measurement tooling (`pip install -e ".[quant]"`).
**Revisit:** only with a different mechanism — e.g. running the *same* ONNX Runtime build on
both sides, or training the heads on the browser runtime's embeddings — and only if a real
tag-flip is ever observed in the field.

### D19 — The partner ingress is a consented, quota-bound, audited API — not a bulk-transcript feed (2026-09-17)
`POST /api/v1/reports` (`api_partner.py`) is Level 2's second ingress, next to citizen
reports, and it is shaped to make the wrong use awkward: **API key per partner**
(`QORGAN_PARTNER_API_KEYS`, parsed at startup, ≥ 16-char unique secrets, constant-time
lookup, closed when unset); **one report per request**; **structured tactic hits are the
first-class shape**; a transcript is accepted only if `scrub_text` is already a fixed point
on it — the server *refuses* an unscrubbed one (422, without echoing it) rather than
scrubbing silently, so a partner cannot use the API to move raw call content across the
boundary; **`consent_basis`** (a machine-readable code from the data-sharing agreement) is
mandatory; the caller number is HMAC-hashed on receipt exactly like a citizen report
(D14); **per-partner rate limit + rolling 24 h quota** (counted from the reports file, so
it survives restarts; `X-Quota-*` headers); **idempotent retries** by `partner_reference`;
a **content-free audit line** for every action (`audit.py` — the schema rejects any field
that `scrub_text` would change, so the log cannot become a copy of the data it accounts
for); partners **delete only their own** receipts (someone else's looks like an unknown
one). `GET /api/v1/organizations` exports **aggregates only** — ids, tactic-derived names,
counts, tactic counts, last-activity dates, novelty, priority; no numbers, not even digests,
no transcripts or excerpts — from a separate module so the ingress module stays under the
import-graph guard (`tests/test_architecture.py`, invariant 4).
**Hardening from the same-day security + code review:** the quota is charged by
**server receipt time** (`StoredReport.received_at`, set on both ingresses), never by the
partner-supplied `occurred_at` — a backdated report would otherwise never fall inside the
window (found as CRITICAL; regression-tested); `occurred_at` is bounded to
`[now − retention, now + 5 min]`; `tactic_ids` are capped (32) and de-duplicated on both
ingresses; every 422 in the app is rendered without pydantic's `input` echo
(`api_limits.validation_error_handler`); bodies are capped at 256 KB by `Content-Length`
and chunked bodies without a length are refused (`BodySizeLimitMiddleware`); the analyst
KPI `ingested` is actual incident membership and `signals_only` is reported separately.
**Accepted limitations (single-process demo server, same as the citizen path):** one full
scan of the reports file per request (bounded by quota × partners × retention; an index
comes with a real database), and idempotency by `partner_reference` is check-then-append
without a file lock, so two *concurrent* identical retries can both store.
**Scope cut, stated:** signals-only reports (no transcript) are stored, receipted,
deletable, counted and quota-charged, but `intake.pending_reports` skips them — there is
nothing to embed; placing them into organizations through the number graph alone is the
follow-up **C9**. **Rationale:** the council's "partner API becomes a bulk feed" risk
(PLAN §8) is answered by shape and limits, not by a policy document. **Revisit:** key
rotation and per-partner scopes when a pilot partner (PLAN §7 item 7) is signed; C9 when
the first signals-only reports arrive.

### D20 — Analysts see aggregates by default; a full transcript is an explicit, audited "open case" (2026-09-17)
The admin API returns organization aggregates (`/api/admin/overview`, tactic profiles,
counts, dates), and the drill-down carries **excerpts** (200 chars + `…`) — for the
representative script, the sample calls and the on-demand model analysis
(`GET /incidents/{id}/analysis`: verdict, ranked tags, trigger phrases, excerpt; no
transcript). The only way to read a whole call is `POST /incidents/{id}/open`, which
answers with the analysis plus the full (scrubbed) transcript **and appends a content-free
audit line** (`audit.py`: `analyst · case.open · incident:<id> · ok[: reason]`); the
analyst is named by `X-Analyst-Id` (the page passes `?analyst=<id>`), `anonymous-analyst`
otherwise, and a `reason` that carries a number or other content is refused (422) rather
than logged. **Rationale:** PLAN C4 / the council's exposure concern — an analyst layer
that shows every transcript on hover is a browsing tool; one that makes reading a call a
deliberate, recorded act is an investigation tool. **Known gap:** `/api/admin` has no
authentication in this demo (pre-existing); the header names, it does not authenticate —
SSO in front of the admin routes is the deployment's job and is listed in STATUS.

### D21 — A "novel scheme" needs support: a linkable number or at least two incidents (2026-09-17)
`analytics/novelty.py` keeps its distance rule (small org, centroid ≥ 0.06 cosine from every
large org) and adds a support gate: an organization with no number digest and a single
incident is never flagged. **Rationale:** the C8 stress table showed the distance rule
alone flags 29 number-less singletons as new schemes once half the callers rotate numbers —
one un-linkable call is an anomaly, not a scheme — while the measured distance margin
between true and false candidates (~0.013) is too thin to fix by tuning the threshold.
After the gate every stressed row flags 1 and the seeded novel family (which has a number)
is still flagged. **Consequence for the demo:** a citizen report with no number cannot
create a novel-scheme callout by itself; two such reports, or one with a number, can.
**Revisit:** with C9 (signals-only placement) and real data — growth over time
(`rank.py`) is the next support signal.

### D22 — Lexicon-free recall measured: the cue lexicon is an explainability instrument, not the recall engine (2026-09-17)
D15 said the device tier is public and the adversary should be assumed to hold the
lexicon; A9 measured what that costs. 109 test + ood scams paraphrased by Gemini to contain
none of the 35 cue phrases (verified locally, language preserved, 0 failures) are detected
at **0.917 [0.849, 0.962]** vs **0.899 [0.827, 0.949]** for their sources at the shipped
0.59 — no recall drop (−1.8 points; 3 flip to clear, 5 to scam). **Decision:** keep
publishing the lexicon (open device tier, ADR D15) — it buys grounded highlights and the
live meter's hard-signal floors, and hiding it would not protect recall because recall does
not depend on it. A10 (ASR-realism training) stays a "could". The split
(`data/adversarial/`, committed with its manifest) joins the harness as `adversarial` and
must be regenerated when the lexicon changes. **Caveat:** an LLM paraphrase is a fluent
adversary of the same family as the training corpus; **A9b** (rewrite calls to *sound
legitimate*, e.g. mimic institutional reassurance) is the harder attack and is not measured.

### D23 — Signals-only partner reports are placed through the number graph, with a zero embedding row (2026-09-17)
A partner report without a transcript (PLAN C5's preferred shape) becomes an incident with
an empty transcript and a **zero** embedding row (`analytics/embed.embed_transcripts`):
embedding the empty string would give every such report one constant vector and cluster
them together. Placement is by number graph only; HDBSCAN (opt-in text overlay) skips zero
rows; novelty ignores them (no text evidence, so such an org is never "far from
everything") and, by D21, a lone number-less report is never novel; the representative
script is the most common *non-blank* transcript or none. The analyst API marks such rows
`has_transcript: false`; analysis/open answer 409. **Rationale:** the partner story
("structured hits preferred") was hollow while those reports were stored but never
analysed. **Revisit:** tactic-profile-only similarity for signals-only orgs if partners
send many.

### D24 — Analyst feedback is append-only, follows the operation, and is applied at read time (2026-09-18)
`POST /api/admin/organizations/{id}/feedback` records **confirm / dismiss / merge** events
in `org_feedback.jsonl` (`analytics/feedback.py`). An event never stores the `org_<n>` id —
every ingest re-clusters and re-assigns those — but a snapshot of the *operation*: its
number digests, else its members; on read it is matched to whichever current organization
shares a number (or ≥ half its members). Effects: `dismissed` → priority × 0.2 and novelty
cleared; `confirmed` → badge; `merge` → the source unions into the target (members,
numbers; the target keeps its id). The latest event per operation wins and is computed
from the org's base state, so a later confirm fully undoes an earlier dismiss. Feedback is
never written into `organizations.jsonl`, so re-clustering cannot lose it; each event is
also an audit line (`analyst · org.<action> · org:<id>`), notes are content-free.
**Rationale:** PLAN C6 — the queue must learn from the analyst, and the analyst's verdict
must survive the next ingest. Built before the C7 interviews on purpose (the plan said
"built in W3 regardless"); D16 may reshape it (e.g. implicit feedback from opened cases).
**Revisit:** confirmed orgs as training signal for ranking once real reports flow.

### D25 — On-device ASR spike (B7): GO on desktop with Vosklet; Whisper no-go; Android unmeasured (2026-09-18)
Three candidates from PLAN B7, measured where measurable (`scripts/spikes/vosklet_bench/`):
**(a) Vosklet 1.2.1** (Vosk/Kaldi in WASM, < 614 KB, streaming partials) with the same small
KK / RU models the July server path used, repackaged as plain tarballs (46 + 60 MB):
loads in ~0.6 s from cache; **RTF 0.06–0.08** on a laptop (12–16× faster than real time,
criterion ≤ 2×); on synthesized speech (macOS `say`, the two demo scenes + a Kazakh scam
line) the same-language transcripts keep the cues intact — RU scam: «никому не говорите …
продиктуйте код из самая с … переведите деньги на безопасной счёт»; KK scam: «ешкімге
айтпаңыз … келген кодты айтыңыз»; RU legit: «никакие года и данные карты называть
ненужно» — and the shipped classifier decides correctly on all of them (RU scam 1.000
ALERT, KK scam 0.997 ALERT, RU legit 0.396 clear, wrong-language legit 0.081 clear).
Cross-language decoding degrades badly, so the July **dual recognizer + per-utterance
voting** design (`asr/vosk_stream.py`) must be ported, not simplified. **Hard
requirement found:** Vosklet needs `crossOriginIsolated` — the live page must be served
with `Cross-Origin-Opener-Policy: same-origin` + `Cross-Origin-Embedder-Policy:
require-corp`, and every cross-origin script tag needs `crossorigin` (jsdelivr is
CORS-enabled; model tarballs are same-origin). Vosklet caches models by id — bump the id
on every archive change. **(b) Whisper via transformers.js:** the only KK+RU fine-tune
(`KRASR/…whisper-small-full-ft`) ships safetensors, no ONNX, and its own card reports WER
0.54 (KK) / 0.72 (RU); generic `Xenova/whisper-small` is ~250 MB, chunked rather than
streaming, and weak on Kazakh — **no-go**. **(c) Web Speech API:** sends audio to the
browser vendor; not a default, at most a disclosed opt-in — not pursued.
**Decision:** GO for a Vosklet-based mic mode on desktop Chrome; **Android Chrome is the
open gate** (WASM threads + ~106 MB + two recognizers on a 3 GB phone) and must be measured
with the same bench before the mic button is enabled on phones. Follow-up **B9** in the
plan. Server-side ASR stays gone (D12).
