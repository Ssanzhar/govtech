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
rejected in D18. **Superseded on the parity claim by D28 (2026-09-18):** at 200 cases the
residual is ~1–1.5 % decision flips, not 0. **Revisit:** when a real-call training set
exists (PLAN A8) — retrain, re-tune, re-gate.

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

### D26 — Microphone mode ships on desktop as a dual-recogniser Vosklet port, gated by cross-origin isolation (2026-09-18)
`site/core/asr.js` ports the July design (D25's GO): the same audio feeds a Kazakh and a
Russian recogniser, the hypothesis with the higher mean word confidence wins per utterance.
Decisions forced by the runtime, each measured: **(1)** the prebuilt Vosklet has one worker
thread per module, so each language gets its own `loadVosklet()` instance (two instances
decode concurrently, combined RTF ≈ 0.07); **(2)** endpoints fire per recogniser 0.3–0.5 s
apart on the same pause, so the first `result` opens a 1.2 s alignment window and votes
against the other language's *partial* (which carries word confidences) if its result does
not arrive, suppressing the late duplicate — a 0.6 s window lost a Kazakh utterance to the
Russian model; **(3)** clean speech ties both models' confidence (0.95 = 0.95), and the
Russian model cannot emit Kazakh letters while the Kazakh model emits few of them on Russian
speech, so a near-tie (≤ 0.02) is decided by the Kazakh-letter share of the KK hypothesis
(≥ 8 % → Kazakh); **(4)** `stop()` tears the recognisers down first and flushes afterwards,
so every utterance reaches the session before the summary; **(5)** the live page and its
worker scripts (`/core/*`) are served with COOP `same-origin` + COEP `require-corp` and
`Cache-Control: no-cache` — a dedicated worker fails silently unless its own script response
carries the document's COEP, and a script cached before the headers existed is blocked
(`ERR_BLOCKED_BY_RESPONSE`) until revalidated; the service-worker shell cache is bumped for
the same reason and Vosklet keeps its own model cache (`/models/vosk/` is not double-cached).
Model tarballs are built USTAR with the root directory entry first (`asr/web_models.py`;
Vosklet needs both). Result: the three demo scenes through the recogniser in real time — RU
scam 81/100 critical, real bank call 14/100 low, KK scam 61/100 high — with nothing leaving
the browser (the network audit is unchanged). **Gate kept:** `isSupported()` refuses phones
until the Android bench (D25) passes; the chip explains why. **Same-day review fixes:** the Vosklet runtime is **self-hosted and hash-pinned**
(`asr/web_models.ensure_vosklet_runtime` → `site/vendor/vosklet/`, sha256 of the 1.2.1
release checked on download) — it sees the raw microphone audio, so it is never a live
third-party script, and the service worker can cache it; a language that endpoints twice
before the other commits its open window instead of overwriting it (a dropped utterance in
fast speech); duplicate suppression holds until it expires; the page disposes the module
instances on `pagehide`; `tests/test_architecture.py` asserts the ASR module contains no
network primitive. **Revisit:** endpointer mode (`FAST`) for shorter utterances on real
speech; a Vosklet build with `MAX_THREADS=2` if two module instances prove too heavy on
low-end laptops.

### D27 — The legit-sounding adversary beats the model; the answer is training data, not a weaker reassurance feature (2026-09-18)
A9b measured the attack D22 left open: scams paraphrased into a calm institutional
register, with the reassurances a real bank gives, and no cue phrase. Recall fell from
0.899 to **0.651** (−24.8 points; mixed-language 0.500), failing the plan's 15-point gate;
the reassurance feature fired on 36 / 109 of the attacks. Two responses were possible:
weaken or drop the reassurance feature (it is what fixed the July false positives on real
bank calls — dropping it trades a known FPR win for an adversarial recall win), or teach the
model the register with data. **Decision:** data — legit-style paraphrases of the *train*
scams only (`scripts/augment_legit_style_scams.py`, train-only augment like the July
reassurance negatives), retrain, and gate on **FPR first** (test / authored / ood at 0.59,
streaming false-latch ≤ 2/24) with the adversarial recall as the secondary number; if the
FPR gates move, roll back (`models/linear_prev`) and ship the number with the caveat. The
split stays in the harness so the trade-off is re-measured on every retrain. **Caveat:** the
attacker and the corpus share a generator family; the real distribution is unknown until
A3 data exists.
**Outcome (same day):** scam-side data alone failed the gate (authored FPR 0.083 with 40
examples, 0.208 with all 268 — the July false positives returned); pairing legit-style
scams with the 45 unused institutional-register legit negatives from July holds every
single-shot FPR gate. **Adopted: 20 scams + 45 negatives** (test recall 0.953,
adversarial_legit 0.826, cue-free 0.927, streaming alert-hit unchanged) over 40 + 45
(0.890 but one streaming alert-hit lost). **One gate is missed at every dose:** streaming
false-latch 3/24 vs ≤ 2/24 — a single hairline call (`real_neg_bank_fraud_alert_ru`, turn-2
meter 60.3 vs enter 59; the shipped model reads the same windows at risk 0.86 / 0.71) that
belongs to the meter's early-window problem (A6), with overlapping intervals. It was stated
as missed, not explained away — and resolved the next day by D29 (the meter arms from the
third utterance): with it the retrained heads meet every stated gate. `models/linear_prev`
remains the rollback, one env var away. The 268-example set is kept under
`data/synthetic/` (gitignored) for the next knee.

### D28 — The cross-runtime parity gate is judged on 200 transcripts at the plan's 99 % rule; the "0 flips / 28" claim is withdrawn (2026-09-18)
D17 stated the int8 runtime residual was "decision-safe: 0 flips / 28". Re-measured on a
200-transcript gate set, the *shipped* heads already flipped 3 / 200 (1.5 %) with |Δrisk|
up to 0.25 on borderline cases; the retrained heads flip 2 / 200 (1.0 %), max 0.27, p95 0.10.
The small golden set had hidden a ~1 % residual. **Decision:** the gate
(`tests_js/integration/embedding.test.mjs`) asserts what PLAN B3 specified — the same
decision on ≥ 99 % of cases — plus p95 |Δrisk| < 0.15 and max < 0.30 and tag Jaccard ≥ 0.9,
on `tests_js/fixtures/runtime_gate.json` (regenerated with the fixtures); the 28-case golden
trajectories remain the bit-level parity test for the JS port itself. **Rationale:** a gate
that cannot see the failure mode is not a gate; restating it at the measured level, with
both models measured, is the honest form of D17. The residual itself is still D18's
finding: runtime kernel differences, not fixable by calibration.

### D29 — The live meter arms only from the third utterance unless a hard signal fired (2026-09-19, PLAN A6)
Every transient false latch on the authored set happened on turn 2, and every one of them
had the same shape: a legitimate institution's opener («Здравствуйте, это банк. По вашей
карте была попытка оплаты…») reads like a scam opener until context arrives (risk 0.9 →
0.75 → 0.35 as the call unfolds). The meter was simulated offline on precomputed per-turn
scores — now `python -m qorgan.eval.meter_sweep`, which traces each dialogue once through
the same rolling window and scorer, caches the traces, and replays variants through the
*production* `meter.update` with an injected config (both the shipped and the retrained
heads, test + authored): `min_turns_to_arm = 3` removes the turn-2 latches (authored 3/24 → 2/24 on the
retrained heads) while keeping **every** alert (16/18 authored, 63/64 test) at exactly +1
median turn-to-alert (2 → 3, the plan's limit); short-window damping (`alpha_up` scaled by
`min(1, turn / N)`) cuts test false-latch further (6/52 → 3/52 at N = 2) but drops a
3-turn scam whose meter can no longer converge, so it ships **off** as a knob.
**Decision:** `QORGAN_METER_MIN_TURNS_TO_ARM = 3` (a confident hard signal still arms the
latch at once — «назовите код из SMS» on turn 1 is evidence), `QORGAN_METER_SHORT_WINDOW_TURNS
= 1` (off), both exported to the device (`qorgan-config.json::meter`, `site/core/meter.js`
1:1, golden fixtures regenerated). The demo scam scene now latches on its third line (87)
instead of its second; the hard-negative scene peaks at 37. **Consequence for D27:** with
this meter the retrained heads meet the streaming gate the previous day missed by one
hairline call. **Caveat:** selected on `test`, checked on `authored` — the same 24
negatives the July fixes were read against; the locked real set (A3) is the only
confirmation. **Revisit:** damping on window *length* (characters) rather than turns once
real ASR utterance statistics exist.

### D30 — Per-tactic decision thresholds for the tactic head, tuned on out-of-fold train + val; A11's "thin tail-tactic data" premise withdrawn (2026-09-20)
PLAN A11 assumed the weak per-tactic rows (`mule_recruitment`, `impersonation_telecom_delivery`,
`verification_ploy`) needed more training data. Inspection said otherwise: the tactic head
**over-predicts** — 2.86 tags per test dialogue against 1.91 true, `otp_request` precision
0.33 at recall 0.90, `safe_account` 0.26, authored `verification_ploy` precision 0.08 — while
recall was already high. A flat 0.5 cut on fifteen one-vs-rest balanced-class-weight LRs is
the wrong operating point for most of them; the fix is a decision-level one, not a data one.
**Decision:** every tactic gets its own threshold (`LinearBundle.tactic_thresholds`, exported
in `metadata.json` and `weights.json::tactic_head.thresholds`, honoured by
`labels.decode_tactics` and `site/core/head.js::decodeTactics` 1:1), chosen by
`calibrate.tune_tactic_thresholds`: max F1 over the grid **0.50–0.70**, ties to the lowest
cut, a tactic with **< 8 positives** in the tuning set keeps the default. The tuning set is
the **out-of-fold** tactic-head probabilities on train (seeded 5-fold of the same head,
`multilabel.out_of_fold_proba`) concatenated with the fitted head's probabilities on `val`:
`val` alone (8–31 positives per tactic) starved the tuner, and an early run on it alone
picked 0.80 for `impersonation_gov_police` from 4 positives, collapsing its test F1 to 0 —
hence the support guard and the grid cap (which also keeps every cut below the hard-signal
confidence floor). The risk head, its calibration and the alert threshold are untouched
(`risk_clf.joblib` byte-identical), so no FPR gate moves. **Measured end-to-end** (through the
cue merge; `docs/eval_report.md`): test micro-F1 0.733 → **0.770** (P 0.61 → 0.76, R 0.91 →
0.78), macro 0.705 → 0.737, tags per dialogue 2.86 → **1.98** (true 1.91); ood micro 0.396 →
0.496, tags 1.23 → 0.67 on a mostly-legit split; authored micro 0.580 → 0.602, macro 0.628 →
0.618 (42 dialogues, 1–8 positives per tactic — noise). The price is recall on `urgency`
(test 0.86 → 0.67) and `verification_ploy` (0.93 → 0.69): the 208 / 160 out-of-fold
positives say 0.60 is the optimum, test's 43 / 29 positives say 0.50 — the tuning set is the
larger sample, so it decides, and the disagreement is recorded rather than tuned away.
**Rejected:** val-only tuning (the first cut: test micro-F1 unchanged at 0.733, half its
moves reversed on test) and tuning on test (that is the evaluation). **Consequence:** the
tag stream that feeds the explanation, the advice engine and L2 cluster naming carries
fewer spurious tactics on legitimate calls; the JS port decodes identically (golden fixtures
and the 200-case gate regenerated: 2 / 200 flips as before, tag-set Jaccard 0.935 ≥ 0.9). **Revisit:** re-tune on real calls once A3 has them —
thresholds tuned on synthetic register may not be the real optimum, and the `urgency` /
`verification_ploy` question is answered only there.

### D31 — Train on the recogniser's register: an ASR-styled copy of every train row, chosen with the device runtime in the loop (2026-09-20, PLAN A10)
The heads were trained on clean Gemini text — capitalised, punctuated, digits — and the
on-device recogniser emits none of that (ADR D25). `qorgan.eval.asr_realism` scores every
eval dialogue twice, clean and through a deterministic styler (`data.asr_style`: lowercase,
punctuation → space, numerals → words via `num2words` ru / kz, `--drop-latin` as the worst
case for «SMS»/«CVV»/«Kaspi»), paired by id, FPR first, with cue and reassurance survival.
On the previously shipped heads the styled register **raises risk across the board**: test
FPR 0.000 → 0.096 (5/52) under the device-faithful runtime of D32 (0.038 under the server's
old one), the flipped calls all Kazakh / mixed legit openers. Cues and reassurance survive
except one Kazakh cue with a suffix hyphen, which a recogniser never emits.
**Decision:** (1) `build_corpus` adds an ASR-styled copy of every train row
(`QORGAN_ASR_STYLE_TRAIN_FRACTION = 1.0`, seeded id-stable, ids `<id>-asr`, train only,
711 → 1,422 rows, regenerated on build — nothing to commit); (2) the two hyphenated Kazakh
cues gain hyphen-free ASR forms (0 clean-train feature rows change; the styled rows are the
paired data); (3) the copies count as full rows. **Measured, not assumed:** dose (0 / 0.25 /
0.5 / 1.0) and weighting (clean / styled ∈ 1/1, 0.5/0.5, 1/0.5, 1/0.25, 0.75/0.75) were
swept on memoised embeddings through the production fit, with the browser gate of D32 in
the loop. Doubling the rows unweighted raises the risk head's ‖w‖ 20 → 27 (the loss term
doubles against a fixed C); pair-weighting 0.5 + 0.5 restores the norm exactly but halves
the clean register's evidence too and costs recall (authored 18 → 15/18, adversarial-legit
0.826 → 0.74) — rejected; every scheme flips 5–6 / 200 on the browser gate, so the gate does
not separate them; unweighted 1.0 is the best on every eval number. **Result, same runtime,
before → after:** styled test FPR 5/52 → **0/52**, authored 1/24 → 1/24 (one call at 0.594),
ood 0 → 0; streaming test false-latch 5/52 → **3/52**; adversarial-legit recall → **0.872**
(cue-free 0.936); clean recall unchanged (test 0.953, authored 17/18, ood 0.867). **Price, on
record:** `ood_neg_legit_bank_call_mixed_3` crosses 0.59 on the server (0.669) — the browser
scores the same call 0.154; and one more transient authored latch (`real_neg_bank_fraud_alert_ru`,
the A9b hairline, an inspected anchor: 3/24 against the A6 gate's 2/24, alerts unchanged at
16/18). Both are single calls inside every interval. **Not done:** modelling misrecognition
itself — the styler is a format model; the Vosk demo clips stay the B9 harness check.
**Found on the way:** three corrupted negatives (`neg_legit_gov_service_mixed_6` in test —
control characters inside Latin-script words, `ood_neg_legit_gov_service_mixed_2`,
`reassure_legit_bank_call_mixed_1` in train — literal `\u` escapes) and twelve rows wrapped
in `\r\n` + stray quotes; all negatives, all scoring correctly, left for their own entry.

### D32 — The device is the runtime the numbers must describe: WebGPU off, ONNX Runtime pinned to the device's version, the runtime gate moved onto captured browser embeddings; D17/D18/D28 restated (2026-09-20)
Chasing a runtime-gate failure of the D31 heads exposed three things about the int8
embedder, each measured on the 200-transcript gate set. **(1) The browser's WebGPU path is
broken for this graph:** in Chromium the embeddings come out at cosine **0.78** (min 0.70)
to the server's — the quantised ops are not all assigned to the GPU provider — and it is
slower than WASM (0.7 s vs 0.2 s per transcript). The worker auto-selected WebGPU wherever
`navigator.gpu` existed, so any such device has been scoring with an embedder the heads
were never fitted to. **Decision:** `embed-worker.js` runs WASM only and refuses other devices.
**(2) The "cross-runtime residual" of D17 / D18 / D28 was an ONNX Runtime *version* gap:**
Python 1.27 vs onnxruntime-node 1.21 gave cosine 0.980 / min 0.958 and 1–1.5 % flips;
with Python pinned to 1.21 the two are **bit-identical** (cosine 1.00000, |Δ| 0). The
patch version alone moved reported numbers — the same recipe reads test FPR 0/52 under 1.27
and 1/52 under 1.21. **Decision:** `onnxruntime==1.21.*` in `pyproject.toml`, the version
transformers.js 3.8.1 bundles for Node; `tests/test_runtime_pin.py` fails if they part;
transformers.js pinned exactly. **(3) The browser is a third build** (onnxruntime-web
1.22.0-dev, WASM): cosine 0.982 mean / 0.949 min to both 1.21 and 1.27 — the Node gate had
been measuring the wrong runtime. On the real browser embeddings both the previously
shipped heads and the D31 heads flip **5 / 200** (2.5 %) with p95 |Δrisk| 0.10–0.12 and a max
of 0.4–0.5 on one code-switched legit call (`ood_neg_legit_bank_call_mixed_3`: server 0.67,
browser 0.15). **Decision:** the runtime gate is `tests_js/integration/browser_gate.test.mjs`
over `tests_js/fixtures/runtime_gate_browser.f32` — the browser's own embeddings, captured by
`npm run gate:browser` (Playwright, headless Chromium, the site's real worker), fingerprinted
against the gate transcripts; it asserts the *measured* level — same decision on ≥ 97 %,
p95 |Δrisk| < 0.15, tag Jaccard ≥ 0.9, max reported — and runs in seconds. The Node test
now asserts bit-identity on the golden 28 (minutes, not half an hour). PLAN B3's "≥ 99 %"
was never met on the device; D28's "2 / 200" is withdrawn as a device number.
**Consequence for the report:** every table is regenerated under the pinned runtime; the
headline moves from "0.000 everywhere" to test 0.019 [0.000, 0.103] · authored 0.000
[0.000, 0.142] · ood 0.013 [0.000, 0.072] — single calls inside every interval, which is the
resolution these sets have, and which int8 runtime variation alone can cross. **Rejected:**
restating nothing (the browser is what citizens run); an fp32 / fp16 device graph (4× / 2×
the 278 MB budget, B3). **Follow-up (PLAN B10):** train and evaluate on the *device's*
embeddings — the browser can embed the whole corpus in minutes through the same tool — so the
reported numbers *are* the device's and the server becomes the 0.98-cosine proxy it
actually is.

### D33 — Train and evaluate on the device's own embeddings: the `device` embed backend over a headless-Chromium bridge (2026-09-21, PLAN B10)
D32 left a 2.5 % disagreement between what the server reported and what the browser
decided, on any heads, because the heads were fitted to the server's native ONNX Runtime and
the browser's WASM build is only a cosine-0.98 proxy of it. The principled end of that is to
stop training on the proxy. **Decision:** a third embed backend, `QORGAN_EMBED_BACKEND=device`
(`classifier/device_embed.py`): Python's `embed_texts` posts its already-prefixed texts to a
loopback bridge (`npm run device:serve`, `tests_js/tools/device_embed_server.mjs`) that runs
the site's real `embed-worker.js` in headless Chromium (WASM, several tabs in parallel, one
text per graph run as always) and returns the vectors the browser computes — bit-identical
to the gate fixture captured from the page (max |Δ| 0.0). Every vector is cached on disk
keyed by (model, browser build, exact text) (`data/cache/device_embeddings.sqlite`,
gitignored), so a corpus embeds once (~0.1 s per text on four tabs) and every later run is
free. The shipped heads are trained and every table computed with this backend; the
bundle records `embed_backend: device`, and `load_linear` allows the server's `onnx` backend
to load it as its documented proxy (`_PROXY_BACKENDS`) while still refusing any other pair.
**What it shows:** on the device, the previously shipped native-trained heads and the
device-trained heads decide the same within a call — test FPR 0.000 / recall 0.953,
authored 0.000 / 16/18, ood 0.000 / 0.87–0.89, styled FPR 0 on every split, cue-free
adversarial 0.917, legit-sounding ≈ 0.80, streaming 3/52 & 3/24 — so the "1/52 test" and
"1/75 ood" false positives of D32's tables were the *server's* errors; the device never made
them. The gain is not a metric, it is that the report now describes the device exactly: the
browser gate reads **0 / 200 flips, |Δrisk| 0.000, tag Jaccard 1.000**, which also proves the
JS scorer reproduces Python on real device vectors. The server's native runtime, now the
proxy, disagrees with the device on 6 / 200 (p95 |Δrisk| 0.114, max 0.51 on the same
code-switched call) — reported as the number that applies to `/api/analyze` and L2 scoring,
not gated, since citizens run the device. **Costs, on record:** the legit-sounding adversary
bites harder on device vectors (0.807, an 11-point drop, inside the 15-point gate); two
authored scams stay under the threshold on the device (16/18; the streaming alert-hit is
15/18 because one more never converges), and the three transient authored latches of D31
remain — all named in the report. **Rejected:** carrying the bridge into production (the
server keeps native ORT; the bridge is a training/eval tool), and an fp16/fp32 device graph
(budget, B3). **Revisit:** when transformers.js or the model moves, `runtime_id` changes, the
cache misses by construction, and the heads must be retrained on the new build — the version
pin of D32 (`tests/test_runtime_pin.py`) and the fixture fingerprint make that loud.

### D34 — Corpus repair: generation artefacts fixed where recoverable, rows dropped where letters were lost, `ood` given its provenance (2026-09-21)
Styling the corpus for ASR realism (D31) surfaced text no recogniser and no citizen would
ever produce: utterances wrapped in `"\r\n … "` with several turns jammed into one, backspace
characters typed over letters, literal `\uXXXX` escapes, and — the serious kind — rows where
the Kazakh-specific letters ә / ұ / ң / і had become control characters (`с\nлеметсіз`,
`С\t\nл\t\nшамын`), all in `mixed`-language dialogues. **Decision:** `data/clean.py`, run by
`build_corpus` on the synthetic set and the augment files and by hand
(`python -m qorgan.data.clean`) on the July `ood.jsonl` that `build_corpus` never produced:
recoverable artefacts are repaired (unwrapped, turns split back with alternating speakers,
backspaces deleted, escapes decoded, trailing whitespace stripped); a row with a lost letter
is dropped and named in `manifest.json.dropped_corrupted`; a corrupted hand-written anchor
raises. Effect: 11 synthetic / augment rows dropped and ~55 repaired (train 1,422 → 1,404
with the styled copies), `test` 116 → 115, `ood` 120 → 118 — among the ood drops
`ood_impersonation_gov_police_mixed_0`, which had been the largest server-vs-device delta in
every gate run: corrupted text is where int8 kernels disagree most. This is an evaluation
change — the eval sets' text moves — so it gets its own entry and its own regenerated
tables (`docs/eval_report.md`); split membership does not move (assignment is by id). The
committed augment files stay as generated for provenance; the drop is a build-time fact.
**Also recorded:** `ood`'s provenance, which `data/README.md` had never stated. **Rejected:**
reconstructing the lost letters (each `\n` stands for a different letter; guessing would
fabricate evaluation data) and hand-editing the eval files (the rule is reproducible, the
edit would not be).

### D35 — A second-generator evaluation split (`shift`): the headline recall was the generator's style (2026-09-21, PLAN A12)
Every evaluation split so far shared a generator with `train`: `test` is Gemini with the same
prompts, `adversarial` / `adversarial_legit` are Gemini paraphrasing Gemini, `ood` is a July
Gemini stress set; `authored_heldout` (42 rows, 18 scams) is the only exception and it was
partly used for feature engineering. **Decision:** a hand-authored split written by a
*different* generator (Claude, Opus 5) from its own knowledge of Kazakhstani calls, with
different prompting and no access to the corpus, the prompts or the lexicons — 66 dialogues,
33 scams / 33 confusable legit, 22 per language, all 15 tactics per language, with subtle,
short and long scams on purpose (`data/authored/shift/`, built by `qorgan.data.shift_set`,
proven disjoint from every other split before it is written). It is evaluation-only and is
recorded as inspected by construction. **What it measured, on the device runtime, at 0.59:**
recall **0.242 [0.111, 0.423]** (8 / 33) against 0.953 on `test`; FPR 0.030 (1 / 33 — a
teacher collecting money for a school trip, 0.62); ru 0.455 · kk 0.182 · mixed 0.091; tactic
F1 near zero. The cue lexicon fired on 6 / 33 scams (the requests are phrased as "read me what
the app shows", "защищённый счёт", "оқып беріңіз"); the embedding head scored textbook
prize / customs-fee / relative-in-trouble scams at 0.00–0.10. The reassurance feature did its
job (the three real fraud-alert calls and the courier scored ≤ 0.01). **Consequence:** the
0.95 / 0.92 recall figures describe Gemini's register, not scam calls in general; the number
the README leads with is now this one until real calls (A3) exist. The threshold is not the
lever (0.40 catches 14 / 33 and fires on 2 legit). Utterance-level pooling is not the lever
(D36). The lexicon must **not** be extended from this set's phrasing (that would make it a
second `authored_heldout`); a third generator or real calls are the only ways to move it
honestly. What can move it: a stronger embedder (D37), the cloud second opinion (the `llm`
backend — measured the same evening, see the eval report: 33 / 33 and 0 / 33 on this set), and training data from
more than one generator. **Rejected:** treating `shift` as a data top-up (it is the only
cross-generator signal we have), and re-authoring it "harder" or "easier" after seeing scores.

### D36 — Utterance-level heads: measured no-go for tactics, a recorded knob for risk (2026-09-21)
Both inference paths already embed every utterance (for the highlights), so utterance-level
heads cost nothing extra on the device. Tried on the device embeddings with weak labels
(evidence utterances = those overlapping a trigger span; styled copies inherit the source's
mask; 2 MIL relabel rounds), grouped out-of-fold threshold tuning exactly like D30, cue merge
exactly like `predict`: **tactics** — max / top-2 / top-3 / softmax pooling, balanced or
unbalanced instance LR, wide threshold grid, stacking, 30 / 50 % ensembles. Best ensemble test
micro-F1 0.816–0.823 vs 0.804 (device baseline), authored 0.60–0.63 vs 0.609, ood ≤ 0.490:
noise; every stand-alone utterance head is worse (max pooling over-fires: 4.7 tags per
dialogue). The spans are not tied to tactics, so the instance labels are too weak to add to
what the whole-transcript embedding already carries. **Decision: no.** A tactic-per-span
relabel (needs the LLM key) is the only version worth retrying. **Risk** — a separate
utterance-level calibrated risk head (same weak labels, MIL-refined), mixed into the dialogue
risk at weight *w*: as an extra feature or as pure max pooling it fails the FPR gate (1–2 ood
negatives fire, one at 0.98); as a mix, at *w* = 0.15 nothing moves except legit-register
adversarial recall 0.807 → 0.853 and the authored margin (0.51 → 0.46); at *w* = 0.35 test
0.969 / authored 17 / 18 but ood loses two calls and its margin drops 0.31 → 0.40; at
*w* = 0.5 adversarial-legit 0.936 with the ood margin at 0.52 — a hairline the runtime
history says not to trust. On `shift` the mix moves recall 0.24 → 0.27 at best and removes
the one FP. **Decision: not shipped** — a second head, a JS port, new fixtures and a gate
re-capture for ≤ 5 adversarial dialogues; recorded as the knob it is, to be re-judged on real
calls. Scripts and JSON in the session scratchpad; the numbers are in `docs/eval_report.md`.

### D37 — A stronger server-side embedder (`multilingual-e5-large`) does not earn a tier (2026-09-21)
The council's publication posture (D15) left room for "a stronger private model behind the
partner API". Measured: the same hybrid heads on `intfloat/multilingual-e5-large` (fp32,
1024-d, 2.2 GB, `QORGAN_EMBED_BACKEND=sentence-transformers`, bundle `models/linear_e5large`,
gitignored) against the shipped e5-base device bundle. On every Gemini-generated split the
two are the same model to within a call: test 0.953 / 0.953, authored 0.889 / 0.889, ood 0.864
/ 0.886, cue-free adversarial 0.917 / 0.917, legit-sounding adversarial 0.853 / 0.815, FPR 0
everywhere. On the second-generator split (D35) it catches **the same 8 of 33** (kk 0 / 11,
mixed 3 / 11, ru 5 / 11 — Kazakh got worse, the others slightly better), with the one
e5-base false positive gone and a better threshold-free ranking (PR-AUC 0.953 vs 0.870).
**Decision: no server tier.** Three times the parameters do not move cross-generator recall,
so the ceiling is the training distribution — one generator's register — not the embedder;
a tier that costs 2.2 GB, a PyTorch runtime and ~15 min of CPU for six eval splits, and
that the partner API has no scoring endpoint for, buys nothing measurable. Re-open only
with training data from more than one generator or real calls (A3), where a larger encoder
might finally have something different to learn from. **Not tried:** BGE-M3 (same reasoning
applies: the data is the bottleneck), contrastive fine-tuning of the encoder (would learn
the same register better — the wrong direction until A3).

### D38 — The cloud second opinion is the tier that generalises: `llm` backend grounded in the taxonomy and made robust (2026-09-22)
The re-benchmark that D35 asked for. Two defects first: the LLM classifier's system prompt
never listed the taxonomy, so Gemini tagged calls with ids of its own (`impersonation.bank`,
`code_request`, `phishing.credentials.2fa_code`) that the explainer cannot template and the
harness cannot score; and `_default_client` built a fresh SDK client with no timeout on every
cache miss, which leaked connection pools and, because httpx's read timeout is the gap
*between* bytes, hung an evaluation for 40 minutes on one idle socket — twice. **Decision:**
the system instruction enumerates the 15 tactic ids with their descriptions and declares the
list closed; `_build_tags` drops any id outside the taxonomy (the verdict is untouched); the
prediction cache is versioned (`v2`) so old free-form answers are not reused; one live client
per process, built with the transport timeout; a 180 s wall-clock deadline per call that
abandons the call in its worker thread, drops the cached client and retries on a fresh pool;
three attempts with backoff on transient transport / 408 / 429 / 5xx errors. **Measured
(`gemini-2.5-pro`):** `shift` **33 / 33 recall, 0 / 33 FPR**, every language, taxonomy tags at
F1 ≥ 0.9 for 12 of 15 tactics; `authored_heldout` 18 / 18 and 0 / 24; `test` 1.000 recall with
**4 / 51 false positives** (kk / mixed synthetic legit calls that open by confirming identity —
`verification_ploy` in the model's reading). **Consequence:** the device model (D35: 8 / 33)
is the privacy tier and the cloud model is the accuracy tier; the product story in D11 — score
on the device, offer the cloud second opinion on the citizen's explicit request — is now the
story the numbers support, not just the privacy one. **Not decided here:** making the cloud
tier automatic (it sends the transcript off-device, which D11/D12 forbid without the citizen's
request), or using it as a labeller to close the device model's gap — the latter is the
obvious next experiment (relabel spans per tactic, generate training data with a second
generator) and needs its own entry. **Caveat on record:** a Gemini judge on Gemini-written
`test` negatives cuts both ways; the hand-written sets are the ones to read.

### D39 — Cue matching that survives the recogniser: bounded-edit matching + behaviour-based Kazakh cues (2026-09-23, STT Tier A)
The hard-signal lexicon is matched against a transcript that, in microphone mode, is
recogniser output — but matching was exact substring containment, so one mis-recognised
character killed the hard-signal floor. **Measured first, on real recogniser output**
(`scripts/spikes/asr_cue_survival/`: 272 corpus utterances spoken by macOS `say` and decoded
by the very Vosk models the browser ships; `data/asr_capture/pairs.jsonl`). A synthetic error
model was rejected as circular — it would prove whatever it was designed to prove. The
capture showed three distinct failures: **moved word boundaries** («три цифры на обороте» →
«три цифры наоборот де»), **morphology** («сотрудникам» → «сотрудником»), and **Latin brand
names** the recogniser transliterates («AnyDesk» → «аны десіп» / «не доски», «TeamViewer» →
«сиам жібер»), which no edit distance can bridge across alphabets.

**Decision, two parts.** (1) `classifier/cue_match.py` + its 1:1 port `site/core/cue-match.js`:
a cue is searched verbatim first and, only if that fails, as a bounded-edit substring of the
**de-spaced** normalised text, which makes a boundary shift free. Budget ≈ one edit per ten
characters and **zero below twelve characters**, so short cues cannot drift. A pigeonhole
prefilter (with `budget + 1` disjoint blocks, one must survive intact) skips the quadratic
search for almost every pair and keeps the browser fast; a test asserts the prefilter never
changes an answer. (2) Three **behaviour-based** Kazakh `remote_access` cues — «қашықтан
кіру», «қашықтан қосыл», «бағдарламасын орнат» — cueing the action instead of the brand.

**Measured (112 cue-bearing utterances, 160 cue-free):** cue recovery **67 % → 76 %**; the
two parts are separable — the lexicon fixes Kazakh (76 % → 86 %), the matcher fixes Russian
(74 % → 84 %). **False fires stayed 0/160 throughout**, in all three languages. Under the
ASR-styled register the shipped model now keeps 8/8 cues (was 6). **Clean-text drift** — the
thing that makes a lexicon change dangerous (the July KK-boundary rollback, ADR D27) — is 5
dialogues in 2,075 for the matcher and 20 for the new cues, and **every one of them is a scam
row: zero hard negatives drift, on any candidate tested.** That is why this did not repeat
July's failure. Rejected candidates, on that evidence: «кодты айтыңыз» (+1 recovery for 50
drifted rows), «қауіпсіз шотқа» (no recovery, 25 rows), «удалённый доступ» (no recovery).

**Gates, retrained on device embeddings:** test 0.000 / 0.953 · authored 0.000 / 0.889 · ood
0.000 / 0.886 · adversarial 0.917 · adversarial-legit 0.807 (0.815, one call) · shift 0.030 /
0.242 · streaming test 0.059 & 0.969, authored clean 0.053 & 0.833 · styled FPR 0 everywhere ·
1,122 pytest, 35 npm including a 3,200-case Python↔JS fixture. The bundle now records
`cue_matcher_version` and `load_linear` refuses a bundle trained under a different one — the
cue block is a model input, so the matcher is part of the feature contract. Rollback:
`models/linear_d38_rollback` (gitignored). **Limits, on record:** the capture is synthesised
speech, cleaner than a speakerphone in a kitchen, so these are lower bounds; the `mixed` row
(55 %) is unreliable because a Russian voice reading Kazakh produces «ешь ком я и тп а», not
what a bilingual speaker produces. Real speakerphone recordings are the missing input.
**Found in review, fixed before shipping (the reason for the extra rules above):** de-spacing
also dissolved the **utterance boundary**, so the tail of one turn and the head of the next
could concatenate into a cue neither contained — «…назовите три цифры на» + «обороте карты…»
scored a `credentials_request` hit at weight 1.0, bypassing the calibrated head. Structurally
new (exact matching kept the newline) and squarely against the FPR-first rule, so the fuzzy
pass now runs per utterance; measured cost: **zero dialogues in 2,075, scam or negative,
relied on bridging**, and recovery stayed 85/112. Also from the review: the browser never
checked the matcher version against the weights it loads (`weights.json` now carries
`cue_matcher_version`, `createScorer` refuses a mismatch — the server-side guard alone was
half a rail); the exact branch's slice-back verification, dropped when the fuzzy branch was
added, is restored for exact hits only; the prefilter-soundness test skips instead of
crashing when the capture corpus is absent, and the scratch audio directory is gitignored
while `pairs.jsonl` is tracked as the ADR's evidence. **Accepted, not fixed:** Python offsets
are code points and JavaScript's are UTF-16 code units, so they would differ for
supplementary-plane characters — which `normalize` drops in both languages, and which no
Kazakh/Russian recogniser emits; the docstrings now say this instead of claiming
byte-identical offsets. Independent fuzzing (8,000 randomised cross-language cases, 20,000
prefilter cases) found no other divergence.

**Not done here (STT Tier A, remaining):** reading the recogniser's N-best alternatives
instead of only the top hypothesis, and A/B-ing the browser's `noiseSuppression` / AGC, which
are tuned for human listening and may hurt recognition. Both now have a harness to measure in.

### D40 — Language locking: measured, implemented, shipped OFF until real bilingual audio exists (2026-09-24, STT Tier B)
Microphone mode runs a Kazakh AND a Russian recogniser for the whole call and votes per
utterance (ADR D26). Two WASM instances is the memory half of the phone gate (ADR D25), so
the question is whether the loser can be switched off once the language has settled.
**Measured** by capturing what EACH recogniser said for EACH utterance in order
(`capture_dual.py` → `data/asr_capture/dual.jsonl`, 108 dialogues / 584 utterances) and
replaying policies offline — the A6 meter-sweep pattern, decode once and simulate many
(`lock_sweep.py`). Policy chosen: **lock after 3 voted utterances to the language that won
most of them, and reopen both whenever the locked recogniser's mean word confidence falls
below a floor** — a speaker switching language looks exactly like a confidence drop.

**Result at `after=3, floor=0.80`:** 0 cue detections lost, text differs from the
both-recognisers baseline on 25/584 utterances (4.3 %) — of which 19 are `mixed` rows whose
audio is a TTS artefact, leaving 5 of ~390 ru+kk utterances (1.3 %); **85 % of dialogues never
reopen** after locking (ru 39/40, kk 34/35) and 91 % of post-lock utterances use one
recogniser. Savings are **18 % of decodes on these dialogues (median 5 utterances) but ~39 %
at 20 utterances and ~42 % at 40** — a real call, not a demo one; the short-dialogue figure
understates it. A stricter floor of 0.85 trades savings for fidelity (15 differing utterances,
77 % never reopen).

**Implementation.** Locking is expressed purely as narrowing `state.languages` to one, so
every alignment rule of D26 still holds untouched: with one language `everyLanguageFired` is
immediate and the vote has a single candidate. The browser runtime then feeds PCM only to
`state.languages`. **This saves decode work, not memory** — the losing Vosklet instance stays
loaded. Freeing memory would mean disposing it and reloading (~0.6 s) on the 15 % of calls
that reopen, i.e. a mid-call stall exactly when the language changes; not taken.

**Shipped OFF** (`QORGAN_ASR_LOCK_AFTER=0`; set it >0 with `QORGAN_ASR_LOCK_CONF_FLOOR` to
enable), like the D29 damping knob. **The reason is honest rather than cautious:** the one
risk that matters is code-switching, and it is the one thing this capture cannot measure — a
Russian TTS voice reading Kazakh produces «ешь ком я и тп а», which is a property of the
voice, not of a bilingual speaker. Flip the default when the Android bench runs on real
bilingual speakerphone audio and the reopen rate holds. Until then the microphone path keeps
both recognisers, and the knob is there to be measured with.

### D41 — Grammar-constrained decoding is not worth a third decode (2026-09-24, STT Tier B)
Proper domain biasing — rebuilding the decoding graph with a language model interpolated
toward scam vocabulary — needs a Kaldi/OpenFST toolchain that is not on this machine
(`compile-graph`, `arpa2fst`, `fstcompile` all absent), so it stays untested and remains the
technically right version of this idea. Its reachable cousin was tested: Vosk builds a small
LM on the fly from a phrase list, giving a recogniser that can emit only those phrases or
`[unk]` — a keyword spotter rather than a bias (`grammar_probe.py`). **Measured against the
open recogniser on the same synthesised audio:** Russian, cue recovery **6/12 vs 11/12** and
1/30 false fires vs 0/30 — worse on both sides; Kazakh, recovery **8/9 vs 7/9** but **2/30
false fires vs 0/30**. So the only gain anywhere is one Kazakh utterance, bought with a
6.7 % false-fire rate on speech that carried no cue, under a project whose primary metric is
FPR — and it would cost a **third** decode per utterance, pulling directly against D40.
**Decision: no.** Constraining the vocabulary removes the surrounding context that makes an
open hypothesis matchable, which the bounded-edit matcher (D39) already exploits better.
**Caveat on record:** 12 and 9 cue-bearing utterances respectively — small, but the direction
is consistent and the cost is certain. Revisit only with a real toolchain, as biasing rather
than restriction.

### D42 — Widening the training register closes half the cross-generator gap; the threshold move that looked free was not (2026-09-24)
ADR D35 measured the device model at 8/33 on calls written by a second generator, against
0.95 on `test`. The hypothesis was that the model had learned **one generator's house style**
rather than the scam, since every split shares its generator and prompts with `train`.
Tested by generating through the same Gemini pipeline with a **sampled register persona**
injected into `generate.py`'s existing `style` hook — caller manner, callee manner, how the
call opens, verbal texture, length (`scripts/augment_register_diversity.py`). Deliberately
**not** written by Claude: `shift` is Claude-authored, so training on Claude text would turn
the only cross-generator test into a test of the model's own author.

**The hypothesis holds.** With 45 register-varied scams + 74 negatives folded into train:
`shift` recall **0.242 → 0.364** (8 → 12 of 33), `test` 0.953 → 0.984, `ood` 0.886 → 0.932,
cue-free adversarial 0.917 → 0.945, legit-sounding adversarial 0.807 → 0.835 — every split up,
at FPR 0.000 on test / authored / ood. So a real part of that gap was register, not semantics.

**The first attempt broke the primary metric, in the exact way ADR D27 predicted.** Scams in
new registers plus *generic* negatives put `authored_heldout` FPR at **0.083** — the same
number, and the same two anchors (`real_neg_bank_fraud_alert_ru`,
`real_neg_telecom_tariff_ru`), as July. The lesson is sharper than "pair augmentation with
negatives": pair it with negatives carrying **the specific counter-signal the new positives
would otherwise drown**. Here that is institutional reassurance ("we will never ask for your
code"), which the taxonomy's negative categories do not prompt for on their own. Regenerating
with 40 % of negatives carrying the reassurance instruction **in the new registers** returned
authored FPR to 0.000 with the recall gains intact.

**Rejected: moving the alert threshold to 0.65.** The remaining cost is one extra false
positive on `shift` (0.030 → 0.061; `shift_legit_kk_11`, a pushy but legitimate bank deposit
sales call, at 0.623), and a threshold of 0.65 removes it. On the A5 tuning set (val +
authored) FPR is 0.000 anywhere in 0.55–0.65, so it looked free. It is not: (1) it is not
Pareto — it saves that one false positive but loses three true positives (adversarial 0.945 →
0.927, legit-sounding 0.835 → 0.826); (2) `SINGLE_HARD_SIGNAL_FLOOR` is **61**, so at enter
0.65 one confident hard signal — a verbatim OTP request — would score 61 and **no longer latch
the live meter**, silently deleting a designed behaviour (the meter test caught it). Threshold
stays **0.59**; the honest targeted fix for that one call is sales-call negatives in varied
registers, not an operating-point change.

**Shipped:** train 1,404 → 1,642, eval splits byte-identical, retrained on device embeddings.
test 0.000/0.984 · authored 0.000/0.889 · ood 0.000/0.932 · shift **0.061/0.364** ·
adversarial 0.945 · legit-sounding 0.835 · streaming test 0.059 & 0.984, authored (clean)
0.053 & 0.889. **On record:** `shift` FPR doubled (1 → 2 of 33, overlapping intervals) — the
one metric that got worse, named rather than buried. Rollback: `models/linear_d41_rollback`.

### D43 — `shift` gets the inspection ledger too; its false positives are not fixed with data (2026-09-24)
The inspection ledger (A2) exists because a number computed on records a developer read while
debugging is not a generalization signal. It was built for the authored anchors, but
`eval.run` applies it to any split, and by now two `shift` negatives have been read:
`shift_legit_mixed_05` (0.62, the highest-scoring negative when the split was introduced,
ADR D35) and `shift_legit_kk_11` (0.623, the false positive register diversity added, ADR
D42). Both are now in the ledger. **Effect:** `shift` 0.061 / 0.364 splits into
**`shift (clean)` 0.000 [0.000, 0.112] / 0.364 (n=64)** and `shift (inspected)` 1.000 (n=2).
Both rows are still reported — the inspected subset is the *hardest* two negatives by
construction, so its 1.000 is not a scandal and the clean 0.000 is not a clean bill of health;
the split simply stops presenting inspected rows as generalization.

**Rejected: generating sales-call negatives to remove `shift_legit_kk_11`.** It is the
obvious targeted fix and it would have worked. But the premise was checked first and is
false: sales-flavoured calls are already **183 of 459 (40 %)** of train negatives, so this is
not a coverage gap. Generating more would be fitting to one named evaluation row — exactly
the "mild eval circularity" `docs/STATUS.md` already admits to twice. The call stays a false
positive on the record. A genuine fix needs real calls, or a category the corpus actually
lacks. **Rule going forward:** before generating data to fix a named evaluation failure,
measure whether the category is under-represented in train; if it is not, the failure is
information, not a task.

### D44 — The citizen reviews, edits and explicitly approves a report before anything is sent (2026-09-25)
`task.md` §8 requires reporting to be "never automatic … review before submission … editable
report contents"; the live page sent the whole draft on one click, showing it only *after*
it was stored. The post-call panel now opens a review (`site/live.js` + the pure
`site/core/report.js`): an editable transcript, a live preview of exactly what the server
will store (numbers / cards / IINs / e-mails redacted by `scrubText`, a 1:1 port of
`scrub_text` pinned by `tests_js/fixtures/scrub.json`, generated from Python and
drift-checked by `tests/data/test_scrub_fixture.py`), per-tactic checkboxes, the optional
number, and a consent checkbox — Send stays disabled (and visibly dimmed) until it is
ticked. **Edits can only remove:** tactics are a subset of the detected ones and flagged
phrases survive only while still verbatim in the edited, redacted transcript
(`finalizeReport`), so a report never carries evidence the model did not produce. The
server contract is unchanged and still re-scrubs (it never trusts the client). Verified in
Chrome (`tests_js/tools/e2e_report_review.mjs`): 0 content-carrying requests before Send,
exactly one POST equal to the edited draft, number stored as `+7 700 ***`, delete by
receipt works. Same pass: `.btn[hidden]` now hides (it also left the mic "End call"
button visible), and `sw.js` precaches `core/cue-match.js` + `core/report.js` (shell v3;
`tests_js/sw.test.mjs` keeps the list complete).
**Not done:** the report offer is still shown for every call, not only above a
configurable threshold (§8), and the page chrome is English-only.

### D45 — PII glued to Cyrillic is redacted; nothing is published unless it is a scrub fixed point (2026-09-25)
**Gap 1 — scrubbing.** Card and IIN rules used `\b`, and in Python's Unicode `re` a
Cyrillic/Kazakh letter is a word character, so `карта4400123456789010` or
`ЖСН940101300123` (ASR output, hand-edited reports) were stored verbatim. The edges are now
"not an ASCII letter, underscore or digit": Cyrillic-glued PII is redacted, while ASCII system
identifiers (digests, receipt ids) stay a fixed point for the audit / feedback / report
validators that use `scrub_text` as a "no PII" check (tests on both sides). Phones already
used digit-only lookarounds. Python and JS change together.
**Gap 2 — publishing.** `ood.jsonl` bypasses `build_corpus` and was never scrubbed: one
synthetic legit call carried a 12-digit IIN next to a full name, and it is on the public Hub.
The local split was repaired with `build_corpus.scrub_dialogue` (1 / 118 dialogues; spans
re-grounded); the stricter rule changes nothing else in 19,137 corpus utterances, so no
split hash other than `ood`'s moves and no retrain is needed. `qorgan.data.publish_guard`
now makes the invariant a gate: `scripts/hf_upload.py` refuses to upload unless every
published split and augment file is a `scrub_text` fixed point (findings name file / id /
utterance, never the value). **Open:** republishing `ood.jsonl` to `sanzh-ts/govtech_ds`
needs the maintainers' write token (outward-facing; not done from this workspace).

### D46 — The analyst console authenticates per person, gates whole calls behind a role and a purpose, and its audit log is a keyed hash chain (2026-09-25)
The council's deal-breaker was that Level 2 is "architecturally indistinguishable from
surveillance infrastructure". In code `/api/admin` had no authentication at all: the analyst
was whoever `X-Analyst-Id` / `?analyst=` claimed, and the audit log was plain JSONL anyone with
file access could edit. This is not new Level-2 scope (the council said to stop investing
there): it closes a liability the demo already carried.
**Change.** `QORGAN_ANALYST_KEYS` (`id:secret:role`, parsed fail-fast like D19's partner keys,
secrets ≥ 16 chars, never shared with a partner, constant-time compare) — identity comes only
from `X-Analyst-Key` (`src/qorgan/api_admin_auth.py`). The console fails closed: no analyst
keys or no audit key ⇒ 503; bad key ⇒ 401; wrong role ⇒ 403. `analyst` sees aggregates and
excerpts; only `investigator` may `open` a whole call, and only with a purpose from a closed
list (`pattern_review` / `citizen_request` / `partner_request`; no law-enforcement code
without a legal basis) and within 30 opens per hour. The audit line (who, which incident,
why) is on disk before the transcript leaves; if it cannot be written, nothing is released.
Wrong keys (never the secret), refusals and every revealing or state-changing action are
audited. **The excerpt view is now a real boundary:** it used to return every trigger phrase
and a reason quoting them — a median 75 % (max 97 %) of each transcript on the 60 seeded
incidents, with no audit line. Phrases outside the 200-char excerpt are now counted, not
quoted (median exposure 36 %, the excerpt itself). **Audit log:** each line carries
`seq` / `prev` / `mac` (HMAC-SHA256, separate `QORGAN_AUDIT_CHAIN_KEY` — a verifier must not
hold the number key, which would reverse phone digests); legacy unchained lines are allowed
only as a sealed leading prefix; appends are serialised (thread lock + `flock`);
`python -m qorgan.audit verify [--anchor SEQ:MAC]` names the first broken entry. The partner
API writes through the same path and is closed without the key. `site/admin.*`: sign-in
panel, key in `sessionStorage` only, role chip, purpose picker, CSP `script-src 'self'`.
Architecture invariant: every admin route carries the analyst dependency, any route returning
a transcript carries the investigator dependency, and a runtime sweep checks 401/503 on all.
Same pass: report-derived incident ids (~0.25 %) tripped the audit scrubber and made `open`
fail — that id shape is exempt, like receipts.
**Evidence.** 8 threads × 15 appends verify as one chain (without the lock it breaks at entry
1). On copies of a real run's log an edit, a deletion, a reorder and a legacy-line edit are each
named at the right entry; tail truncation is caught only with an anchor. Chrome
(`tests_js/tools/e2e_admin_auth.mjs`, re-run independently on scratch data): no key / claimed
id / wrong key → 401; analyst `open` → 403 and locked in the UI with the reason; investigator
opens only after a purpose; revoked key → sign-in; unconfigured → 503. `pytest` 1204 passed.
**Not done.** SSO/MFA, key expiry/rotation, a key id for the audit key; shipping lines to WORM
storage (the key holder can still rewrite history); an auditor role / audit view; limits are
in-process; the Streamlit harness reads files directly (no auth — it is a dev tool, D4); the
purpose list and open budget need the legal owner (`docs/LEGAL_ASSESSMENT.md` §5).

### D47 — One language control drives the whole citizen page; a mid-call switch re-renders in place (2026-09-25)
`site/live.html` served Kazakh- and Russian-speaking citizens English chrome; its ru/kk radio
switched only the advice language, and a mid-call switch left advice, evidence and the summary
in the old language (known wart) because strings were written once and the session locale was
fixed at start. **Change.** All chrome lives in one pure module, `site/i18n.js` (kk / ru / en,
keyed placeholders); markup carries `data-i18n*` keys. Model content (tactic names, advice,
reason templates, human note) is **not** copied there: it is re-rendered from saved call state
through the same pure `recommend` / `summarize` / `renderReason`, and the running session's
locale switches too (scoring never reads it). English chrome shows the reviewed Russian
content and says so rather than showing unreviewed English advice. The review form is never
rebuilt, so edits and consent survive a switch. Initial language: stored choice, else kk/ru
from the browser, else ru; `<html lang>` follows. Citizen copy: "Не спешите. Проверьте
звонок." plus three promises — the call stays on your device, a human decides and the AI can
be wrong (Law 230-VIII disclosure), it shows why — with the pipeline behind "how it works". A
one-time download notice (~300 MB; ~106 MB more for the microphone) appears before anything
is fetched. Accessibility: `role="meter"` with a spoken value, labelled controls, visible
focus, PT Serif for Cyrillic (Fraunces has none). Fixed on the way: white-on-white
`.btn-ghost` buttons, an unretryable failed model load, a summary sentence that quoted every
flagged phrase (now three).
**Evidence.** `tests_js/i18n.test.mjs` (same keys, no empties, placeholder parity, Kazakh not
copied from Russian, every key the page uses exists); `sw.test.mjs` now requires page imports
to be precached (shell v4); `npm test` 60/60. Chrome (`e2e_live_i18n.mjs`, re-run
independently): RU scam scene switched to kk at turn 3/6 → the same tactic's advice in Kazakh,
call runs to 6/6, edits kept across kk → ru → en; fresh kk-KZ phone profile at 360 px all
Kazakh, no overflow; no Latin words on ru/kk pages; no `/models/` request before Start and no
request carrying call content before Send. `e2e_report_review.mjs` still passes.
**Not done.** Native review of the new Kazakh strings (listed in `docs/STATUS.md`); legal
wording of both consent sentences; the landing page and `try.js` are English; Cache Storage
refused the 279 MB model in a throwaway profile, so "stays on your device" is unverified on real
profiles and phones; the ONNX runtime still loads from jsdelivr; the report is still offered
for every call (D44).

### D48 — The runtime installs only what the shipped product imports (2026-09-25)
`pip install -e .` and the Docker image pulled CPU torch, transformers, captum,
sentence-transformers, hdbscan, faster-whisper, Streamlit and google-genai, although the
served path is the int8 ONNX embedder + sklearn heads + FastAPI. **Change.** `pyproject.toml`
dependencies are that runtime (13 packages); the rest are extras — `cloud` (Gemini: the
consented second opinion and data generation), `harness` (Streamlit dev harness, D4), `live`,
`research` (xlmr, fp32 embeddings, HDBSCAN overlay), `quant`, `dev`, `all`. The Dockerfile no
longer installs torch; `requirements.txt` points at `pyproject.toml`.
`tests/test_runtime_deps.py` blocks every extra's top-level module *and records each attempt*
(`/api/analyze` degrades to `mock` on any exception, so a swallowed import would otherwise
pass), then imports the server, the deploy bootstrap and the retrain path and scores a call.
**Evidence.** The probe records 0 attempts and scores with `linear`; blocking `networkx` makes
it fail (mutation check). `pip install -e ".[dev]" --dry-run` resolves on the lean venv with
nothing but dev tools to add; that runtime is 588 MB of site-packages on macOS arm64.
**Not measured.** The Docker image size before/after (Docker was not running here).

### D49 — Close the data paths the product story says do not exist (2026-09-26)
`docs/LEGAL_ASSESSMENT.md` (§1, gap list M1/M3/M4) found, and the code confirmed, four paths
that contradicted "call content leaves the device only on an explicit, reviewed, consented
report; `/api/analyze` persists nothing; one citizen ingress; reports expire":
1. **Caller-selectable cloud tier.** Any caller could pass `backend=llm` to `/api/analyze`, the
   live-session API or the admin analysis routes — text to Gemini, outside Kazakhstan, with no
   notice — and the classifier cached the verbatim trigger phrases on disk. Now
   (`qorgan.cloud_tier`): `QORGAN_CLOUD_TIER=off` by default → 403, nothing sent (a configured
   `llm` default cannot bypass it); on → `/api/analyze` also needs the requester's explicit
   `cloud_consent` (422 without); request-time scoring never writes the cache
   (`predict.score(use_cache=False)`; batch eval still caches); analyst routes refuse `llm`
   outright (422) — the cloud is a citizen's choice, not an analyst's.
2. **A second, consent-free citizen ingress.** `/api/live/session/*` held every utterance in
   server memory (LRU, no TTL) and its `/report` stored a report with no review or consent. No
   product page used it (the citizen page scores on the device; the Streamlit harness drives
   `qorgan.live` in-process), and PLAN B6 had scheduled its deletion. Retired with its
   in-memory store; `GET /api/live/{capabilities,scenarios}` remain. **New invariant:** every
   write route is named in `tests/test_architecture.py`, and exactly two create reports
   (`POST /api/reports`, `POST /api/v1/reports`) — a new one has to be argued there.
3. **Retention on the client's clock.** `purge` aged reports by the client-supplied
   `timestamp` (unbounded — a future date never expired) and nothing scheduled it. Now
   `reports.retention`: age runs on the server's `received_at`; a client `timestamp` outside
   [now − retention, now + 5 min] is refused (422); the server purges at startup and every
   `QORGAN_REPORT_PURGE_INTERVAL_HOURS` (default 24; 0 = cron); report writes and the purge
   share one lock. Deleting or purging a report also removes its incident id — and, unless
   another incident or report still carries it, its number digest — from `org_feedback.jsonl`.
4. **No proof of what was agreed to.** `POST /api/reports` now requires a registered
   `consent_version`; `reports.model.CITIZEN_CONSENT_VERSIONS` maps each version to the SHA-256
   of the exact consent wording in every page language (`site/i18n.js`), and a test fails if the
   wording changes without a new version. Stored with the report, echoed with `expires_at` in
   the receipt; the page sends it.
Also: config secrets (number HMAC key, Gemini key) are `Secret*` fields — no secret appears in
`repr`/`str`/JSON dumps of the config (tested per field).
**Review fixes to D46 (independent review of the D44–D48 snapshot: 0 critical/high, 2 medium,
3 low).** *Medium:* the partner API stored/deleted **before** auditing, and an audit failure was
an unhandled 500 with the data already written — it now audits first and maps audit failures
to 503 with nothing changed (tests). *Medium:* a routine append silently sealed any unchained
lines at the head of the log, so someone without the key could plant invented "history" naming
a real analyst and have the next request vouch for it — appends now refuse unsealed legacy
lines; sealing is an explicit, logged `system` step (`python -m qorgan.audit seal`), and
`verify` notes that no key protects legacy content. *Low:* `load_audit` split on U+2028 inside
JSON strings (now bytes on `\n`); a failed analysis no longer spends the investigator's open
budget. Not changed (documented): the failed-auth limiter keys on the client address (behind a
proxy it is shared) and secrets are length-checked, not entropy-checked.
**Evidence.** `pytest` 1214 passed / 10 skipped (the retired session-route and store tests went
with the code; their meter/session logic is covered in `tests/live/`); `npm test` 60/60; new
tests `tests/test_cloud_tier.py`, `tests/test_purge_schedule.py`, `tests/reports/test_retention.py`,
`tests/reports/test_consent_versions.py`, the write-route invariant, partner audit-first,
explicit seal. Chrome on a scratch copy of `data/`: report review (payload carries
`consent_version`), live i18n and admin auth e2e all pass; `POST /api/live/session` → 405;
`/api/analyze` with `backend=llm` → 403 while the tier is off.
**Not done.** Spoken-number redaction (the on-device recogniser writes numbers as words:
`scrub_text('мой номер восемь семьсот один два три…')` is unchanged, so a microphone-mode report
can carry a number) — the top open privacy item; it needs Python + JS parity and a corpus-impact
measurement before it can change what the model trained on. The audit/access-log retention
period is a legal decision (LEGAL_ASSESSMENT §6). The Streamlit harness still stores reports
without a consent version (dev tool, D4).

### D50 — The publish gate gains its repair; the Hub's unscrubbed IIN is fixed (2026-09-27)
D45 made the invariant explicit — every published utterance must already be a `scrub_text`
fixed point — but shipped only the *gate*. There was no tool to establish it: `hf_upload.py`
refused the upload and named a **function** (`build_corpus.scrub_dialogue`), leaving the
operator to hand-edit a split, which is how the row got published in the first place. Finding
without fixing is half an invariant.

**The row.** `ood.jsonl`, `ood_neg_legit_gov_service_ru_1`, utterance 1: a legitimate
ЦОН/eGov caller reads back a fabricated 12-digit IIN (`is_hard_negative`, risk 0.02) — the
exact file the guard's own docstring names. It is also the live public copy on the Hub, so
legal gap **M11** was real and not merely local. `ood.jsonl` is a July artifact with no live
producer (repaired by hand per D34), so there is no generator to correct: the data plus the
gate *is* the complete remediation.

**The repair.** `publish_guard.scrub_file(path)` reuses `build_corpus.scrub_dialogue`, so
trigger spans are re-grounded and a span that *was* the PII is dropped rather than left
pointing at `[IIN]` (the schema's verbatim invariant). It leaves a file that is already a fixed
point completely untouched — no rewrite, so a clean corpus never churns and the call is
idempotent — and raises on a line that is not a valid `Dialogue`, because a split we cannot
parse is one we must not silently publish. `python -m qorgan.data.publish_guard` reports and
exits non-zero; `--fix` repairs. `hf_upload.py`'s refusal now names that command.

**Measured cost: none.** The repaired utterance moves the dialogue from 0.0324 to 0.0391 on the
onnx proxy against a 0.59 threshold, so it cannot change a decision. Split-level `ood` after
the fix: recall **0.932** (unchanged), FPR 0.014 [0.000, 0.073] — and the single false positive
is `ood_neg_legit_bank_call_mixed_3` at 0.714, the pre-existing server-proxy artifact already
recorded in D31 (0.67 server / 0.15 browser), not this row. Exactly one line of `ood.jsonl`
changed; the other 117 are byte-identical. Numbers are the onnx proxy, not the device backend
the headline uses — the device restatement needs `npm run device:serve` and is not required
here, since the decision margin is ~0.55.

**Still open.** The Hub copy is unchanged: republishing is outward-facing and needs sign-off,
and M11 also asks for a `LICENSE` plus HF cards stating "synthetic, no real persons". Until
then the guard keeps the local corpus honest and blocks any upload.

**Evidence.** Six new tests in `tests/data/test_publish_guard.py` (fixed point, reported
findings, no-churn, span dropped, idempotent, missing file), written before the implementation;
`pytest` **1241 passed / 2 skipped** (was 1234 / 1 failed / 2 skipped); `npm test` 60/60.

### D51 — The OTP cue lexicon spelled SMS only in Latin; real Russian and real ASR write «СМС» (2026-09-27)
Found while probing the shipped lexicon against real-world Russian scam phrasing (the external
dataset survey, `docs/DATA_SOURCES.md`): the single most canonical real OTP-phishing line,
«Продиктуйте код из СМС», matched **nothing**. Every SMS cue was Latin-spelled
(`"код из SMS"`, `"SMS-тегі кодты айтыңыз"`), and `normalize()` correctly keeps the scripts
distinct (`кодизсмс` ≠ `кодизsms`), so the D39 bounded-edit matcher cannot bridge it — a script
change is not an edit-distance problem.

**Rejected alternative.** Folding Latin↔Cyrillic homoglyphs inside `normalize()` would have
invalidated `MATCHER_VERSION` and therefore every trained cue feature, and it would not even
work: `SMS` → `СМС` is *acronym transliteration*, while homoglyph folding maps Cyrillic `С` to
Latin `C`, not `S`. A lexicon variant is the honest fix, and the file already has a documented
slot for it (`# ASR form`).

**Live, not hypothetical.** In `data/asr_capture/`, `secrecy_mixed_3#6` was decoded correctly by
Vosk except for the script — «код из смс» — and lost its `otp_request` cue, so the 61/81
hard-signal floor never armed. Corpus-wide, **53 positive rows** voiced a request-shaped
«код из СМС» while firing zero OTP cues (46 `train`, 3 `test`, 2 `val`, 2 `ood`).

**The change, measured per candidate.** `"код из СМС"`: +36 positives, +6 negative rows (3 unique
dialogues plus their ASR-styled copies). `"код из пуш"` / `"код из пуш-уведомления"`: **+0 / +0**,
so not added — the same bug class with no instances is not worth the surface. The two Cyrillic
**KK** forms also gain +0 today and were added anyway, deliberately: the cue flag is per *tactic*,
not per phrase, so the `otp_request` weight is already learned from the Latin-form rows and
widening what sets the flag has no training-time effect, while closing the identical bug for
Kazakh ASR output. That is reasoning, not measurement — on record as such.

**Why the 6 negative hits are acceptable.** All are reassurance rows where the sentence is a
negation («Мы никогда не спросим у вас ПИН-код, CVV или код из СМС»). Cue-hits on legit calls are
already the status quo, not a new class: **8 of 1,136** negatives fire existing cues (e.g.
«SMS кодтарын ешкімге айтпаңыз»), and only 4 of those 8 carry the reassurance counter-signal;
2 of the 3 new ones do. Separately noted: `reg_neg_reassure_legit_bank_call_ru_72` reads as
textbook reassurance but does not trip the reassurance lexicon — a gap to fix on its own, not
bundled here, so a gate movement stays attributable.

**No new training data was added**, because the 38 rows that gain the cue are existing positives
whose feature was silently zero. That makes this the lexicon-only shape the July sprint lesson
warns about, and the predicted symptom duly appeared (below).

**Results** (device backend, threshold 0.59, ADR D33 methodology):
`test` FPR 0.000 / recall **0.984 → 1.000** · `authored_heldout` 0.000 / 0.889 held ·
`ood` 0.000 / 0.932 held · `shift` 0.061 / 0.364 and `shift (clean)` 0.000 / 0.364 held.
Streaming: `test` false-latch 3/51 held and alert-hit **62/64 → 63/64**; `authored` false-latch
**3/24 → 2/24**, clean 1/19 held, alert-hit **15/18 → 16/18**. The three `test` positives that
were losing the hard signal are exactly the ones recovered.

**Cost, on record.** ASR-styled `ood` FPR **0.000 → 0.014** (1/74): `ood_neg_legit_bank_call_mixed_3`,
clean 0.132 → styled 0.612. **No cue fires on that row at all** — clean or styled, old lexicon or
new — so the joint-LR refit moved it, precisely the effect the July lesson describes. It is not
fixed with data, because it is an eval-split row and targeting it would be tuning on held-out
data. It is also the row D31 already recorded as the corpus's weakest negative (0.67 server /
0.15 browser); it is now in the inspection ledger, so the split reports `ood (clean)` beside the
full row. Styled `test` and `authored` FPR stay 0.000, and cue/reassurance survival is 11/11,
13/13, 10/10 and 5/5, 9/9, 6/6.

**Ledger.** Two `ood` rows were read during this work and are recorded (ADR D43 treatment):
`ood_neg_legit_gov_service_ru_1` (read only to repair the published IIN, D50) and
`ood_neg_legit_bank_call_mixed_3` (read to attribute the movement above).

**Rollback.** `models/linear_d50_rollback` (device-trained, pre-D51 lexicon; it hash-checks the
lexicon, so it must be restored together with the old YAML).

**Evidence.** `pytest` **1241 passed / 2 skipped**; `npm test` **60/60**; `weights.json`, the
parity, runtime-gate and scrub fixtures regenerated; corpus rebuilt with an unchanged
`content_hash` (the lexicon does not alter corpus content).

### D52 — English becomes a reviewed content locale, not an alias for Russian (2026-09-27)
The citizen page has offered English chrome since D47, but only the chrome: `CONTENT_LOCALES`
was `["ru", "kk"]` and `contentLocale` mapped `en` onto `ru`, so an English-speaking user read
English buttons and then **Russian** tactic names, advice, explanations and summary, under a
notice admitting it. The seam was already named in the code; this closes it.

**Scope, deliberately narrow.** This is the *content* layer only. `SupportedLanguage`
(`ru|kk|mixed`) is the **corpus** enum and is untouched: the language a call is spoken in and
the language its explanation is rendered in are independent axes, and English scam *data* is a
separate piece of work. Nothing about the model, the lexicons or the corpus changes here, so no
retrain and no eval movement — the risk head is byte-identical.

**What changed.** `explain/templates_en.yaml` and `explain/advice_en.yaml` (all 15 tactics,
verification questions, low-confidence note), an `en:` display name per tactic in
`data/taxonomy/tactics.yaml`, `TacticDefinition.en` with `display_name` dispatching on the
locale code, `_SUPPORTED_LOCALES` and `_DEFAULT_SUPPORTED_LOCALES` gaining `en`, and
`QORGAN_SUPPORTED_LOCALES=ru,kk,en` in `.env.example`. On the client, `CONTENT_LOCALES` gains
`en`; `contentLocale` needed no logic change because it was already generic.

**Kazakhstani, in English.** The advice keeps the local institutions and glosses them —
"IIN (your national ID number)", "the regulator (ARDFM)" — because the reader is an English
speaker *in Kazakhstan*, not a generic English audience.

**A string that would have become a lie.** `call.content_fallback` told English users that
advice "is shown in Russian: they are written and reviewed in Kazakh and Russian only". The
notice now never renders for a shipped locale (it is `hidden` whenever `locale === contentLocale`),
but the trailing clause was false in a shipped bundle, so it was removed. The mechanism is kept:
it is correct defensive behaviour for any future chrome-only locale.

**Not changed, on purpose.** `pickLocale` still only auto-selects `kk` or `ru` from the browser
language, so an English browser still *starts* in Russian and the user picks ENG. That looked
like a deliberate choice in D47 rather than an oversight, and reversing a default is a product
decision, not a content one. Worth an explicit answer now that English is complete.

**Tests.** The locale-coverage tests are parametrised over `ru/kk/en` (advice covers every
taxonomy id; every locale's templates are complete), plus a new test that each tactic has three
*distinct* display names, so English can never silently become an alias again. Four tests used
`"en"` as their example of an *unsupported* locale and now use `"de"` — the intent is preserved,
only the example moved. The i18n unit test asserts `contentLocale("en") === "en"` and that an
uncovered locale still falls back.

**Evidence.** `pytest` **1246 passed / 2 skipped**; `npm test` **60/60**; the live i18n
end-to-end suite passes in real Chrome (`node tests_js/tools/e2e_live_i18n.mjs`, scenes A/B/C),
updated to assert the reviewed English summary note and English review chips and the **absence**
of the fallback notice. Verified visually in the browser: verdict band, all eight tactic chips
and the advice render in English while the Russian call transcript stays Russian with its
trigger spans highlighted.

### D53 — The shared Gemini seam gets a wall-clock deadline; generation retries (2026-09-27)
Adding English (D54) meant a first generation run since the CLIs were written, and it **hung
for 83 minutes at 0 % CPU with an empty output file** before being killed. The cause was
already written down in this repo: D38 hardened the *classifier's* client because "httpx's
read timeout is the gap between bytes, so a socket that stays ESTABLISHED while the server
sends nothing never trips it (observed: 40-minute hangs)". That fix lived in
`classifier/llm_classifier.py`. The **generate and label CLIs go through the same
`llm_tools.generate_json` and never got it** — they had only the 120 s transport timeout that
D38 says is insufficient.

**Fixed at the shared seam, not per caller.** `generate_json` now runs the SDK call on a worker
and abandons it after `deadline_s` wall-clock seconds (default 180), raising `TimeoutError`.
Deliberately *not* a `with ThreadPoolExecutor(...)` block: its `__exit__` joins the worker,
which is precisely the hang being escaped — the first implementation passed its test only
after 30 s for that reason. A thread blocked in a socket read cannot be cancelled, so it is
abandoned and ends when its socket does; it holds no state we keep. Every caller is now
covered, including `label.py`, which had the same exposure and no test.

**A batch must survive one bad socket.** `generate._call_tool` retries `TimeoutError` and
`LLMResponseError` (a truncated or unparseable payload — a fresh sample usually fixes it) up
to 3 times with linear backoff, then raises `GenerationError` naming the cause. Before this, a
single blip discarded every dialogue generated so far: `generate_batch` accumulates in memory
and writes only at the end, so failure at call 169 of 170 lost all 169.

**Verified, not assumed.** After the fix a live single call answered in **1.1 s**, so the
83-minute stall was a genuine silent socket and not a broken key or endpoint.

**Then it happened again, differently.** The retry caught `TimeoutError` and
`LLMResponseError` only, so the next run died on `httpx.ReadError: Connection reset by peer`
after ~10 minutes — and because `generate_batch` accumulated in memory and wrote once at the
end, the whole batch was lost a *second* time. Two fixes, both at the shared seam:
`llm_tools.is_transient` (moved out of `llm_classifier`, which now delegates to it, so
"transient" means one thing across the classifier and both CLIs) already covered
`httpx.HTTPError`; and `generate_batch` gained a `sink` callback plus `skip_ids`, so the CLI
appends and flushes each dialogue as it is produced and re-running **resumes** from what is
already on disk. A dropped connection now costs one dialogue, not a run.

**`label.py` had the same two holes and was fixed before it could bite.** Labelling is the
*next* few-hundred-call pass over the same seam: it had no retry at all (only `LLMResponseError`
converted to `LabelingError`) and `main` wrote once at the end. It now retries transient
failures and appends/resumes exactly as generation does. Fixing it pre-emptively rather than
after a third lost run is the point of writing the failure down.

**Evidence.** New `tests/test_llm_tools.py` (the seam had no tests at all): the deadline fires
on a client that never answers and returns in well under 10 s, a normal call still returns,
and an unparseable payload is still `LLMResponseError`. Plus two generation tests: a transient
failure is retried, and exhausted retries raise loudly rather than yielding a short corpus.
`pytest` 1250 passed / 2 skipped.

### D54 — English becomes a third call language; what it fixed, and what it exposed (2026-09-28)
D52 made English a reviewed *content* locale (the UI and the explanations). This makes it a
**call** language (the third: ru, kk, en — `mixed` is kk/ru code-switching, not a separate
language): `SupportedLanguage` gains `en`, and the corpus gains 170 English dialogues
(120 positive / 50 hard negative, all 15 tactics), generated through the existing Gemini
pipeline with a language instruction that keeps the setting local — "Kaspi, Halyk, eGov, tenge,
IIN; only the language is English". 168 of 170 carry a Kazakhstani marker, so this is
**Kazakhstan in English**, not a US corpus.

**Generated separately, on purpose.** `configs/corpus_en.yaml` writes its own file and
`scripts/merge_synthetic.py` merges by id, so adding a language did not regenerate ru/kk/mixed
and invalidate every number in this report. `configs/corpus.yaml` lists `en` so a full
regeneration still reproduces the corpus.

**Cues: measured one at a time, and two were rejected on evidence.** Candidates were derived
from `train` only. `"your IIN"` fires on **20 legitimate calls** and `"confirm your IIN"` on 3 —
the bare-mention trap the lexicon's own invariant warns about — so both were rejected in favour
of request-shaped forms. `"the code"` and `"a one-time code"` measured **0** false hits and were
*still* rejected: our English negatives contain **zero** reassurance language, so the corpus
cannot reveal that risk. A clean measurement over a blind spot is not evidence.

**A language-independent bug found along the way.** `AnyDesk` was a `remote_access` cue;
**`TeamViewer` never was** — although the comment beside it names TeamViewer and the corpus uses
it 35 times across ru/kk/mixed/en. Adding it helps Russian and Kazakh, not just English.

**Re-labelling was overruling a structural label.** A hard negative is legitimate *by
construction*, but `label.py` let the model overwrite that: on a legitimate bank call verifying
an identity it returns risk 1.0 with tactic tags — **32 of 50** English negatives came back
tagged, one at risk 1.000. The existing ru/kk corpus is clean only because the historical run
relabelled positives only; the protection was a property of how it was run, not of the code.
`_build_label` now returns the structural label for hard negatives (see the commit).

**Results** (device backend, threshold 0.59):
`test` FPR 0.000 / recall **1.000** (n=143, English included) · per language all four at
0.000 / 1.000 · `authored_heldout` 0.000 / 0.889 and `ood` 0.000 / 0.932, both unchanged ·
**`shift` 0.061 → 0.030 FPR and 0.364 → 0.455 recall.** State this carefully: the intervals
overlap heavily (0.364 [0.204, 0.549] → 0.455 [0.281, 0.636], 12/33 → 15/33), it is a single
run, and **three things changed together** — English training data, the `TeamViewer` cue, and
the retrain itself. So this is a co-occurrence worth recording, not a demonstrated effect of
adding a language; isolating it needs an ablation we have not run. Styled FPR: test and authored 0.000, `ood`
0.014 — unchanged from D51, not a new regression. Streaming: test false-latch 3/60, alert-hit
0.988; authored clean 1/19 unchanged, inspected 1/5 → 2/5 (cost, on record).

**The honest caveat, and it is a big one.** English has **no `authored_heldout` anchors and no
`shift` split**, so its 1.000 shares a generator with its training data — exactly the weakness
D35 exists to expose. Measured against an independent generator
(`BothBosu/scam-dialogue`, Apache-2.0, evaluation only):

| English probe | before English training | after |
|---|---|---|
| recall | 0.475 | **0.739** |
| FPR | 0.078 | **0.227** |

Recall improved everywhere (refund 0.094 → 0.394, reward 0.225 → 0.606, ssn → 1.000, support
0.631 → 0.956). **But FPR tripled**, and FPR is the primary metric. Diagnosed rather than
guessed: the 145 false positives are **entirely insurance (66) and telemarketing (79)** — pushy
but legitimate sales calls — and **not one of the new cues fired on any of them**. The cues are
innocent; the model has simply never seen a legitimate hard-sell call in English. Our 50
English negatives cover delivery, family, chit-chat, bank and e-gov, with no sales register and
no reassurance language at all.

**Therefore: English is not at Russian/Kazakh quality and must not be presented as if it were.**
The fix is known and scoped — English negatives in the pushy-sales and insurance registers,
~40 % of them carrying the reassurance counter-signal, exactly as D27/D42 did for ru/kk. A first
attempt stalled on repeated silent sockets (the API throttles after a burst and the connections
hang rather than returning 429), and because
`augment_register_diversity.py` still writes once at the end, that batch was lost. Note the
probe's own limit: its legitimate calls are US-register, so some of that 0.227 is a distribution
we do not serve — but the reassurance and sales-register gaps are real regardless.

**Tooling.** `--languages` on both augmentation scripts (they hardcoded `ru/kk/mixed`), and
`--tag` on `augment_register_diversity.py`, because an untagged rerun **overwrites the committed
ru/kk/mixed batch** — caught before running it, not after.


### D55 — The server scores with the weights the browser ships; a deploy never swaps the model (2026-09-29)
Found on a fresh pull of `main`. D51/D54 changed the cue lexicon and committed the retrained
browser weights (`site/models/weights.json`), but the server's sklearn bundle (`models/linear`,
gitignored) still comes from the Hub, which holds the 2026-09-24 bundle. Its lexicon hash no
longer matched, so **every server-side score silently fell back to the keyword `mock`**: a
textbook «…никому не говорите. Продиктуйте код из СМС» call scored **0.05**, and the analyst
drill-down and the landing widget's server path showed mock verdicts. Worse, the deploy
bootstrap (run by the Docker build and at every container start) saw the stale bundle,
**retrained a different model** (older corpus without English, server embeddings) and copied
its export over the committed browser weights: a deployment would have shipped every browser
a model nobody evaluated. On Sanzhar's machine nothing failed, because his local bundle is
current; every other checkout was affected.
**Change.** `web_bundle.load_linear_from_web` rebuilds a serving `LinearBundle` from the
browser's JSON: the numpy `WebScorer` (already pinned to sklearn) behind sklearn-shaped heads,
and the lexicons *embedded in the file* — what the browser computes its features from — after
checking they match their own recorded hashes, the cue-matcher version and a compatible
embedder. `QORGAN_LINEAR_WEIGHTS` picks the source: `bundle` (a retrain being evaluated before
export), `web` (deployments: server and device score identically by construction) or `auto`
(default: the bundle when it is valid for the current lexicons, else the browser weights — never
`mock` while a real model exists; logged). `/api/health` reports `linear_model_source`. The
bootstrap now asks which source serves: when the browser weights serve, it neither retrains nor
downloads under `web`; it never overwrites usable committed weights (developers publish a
retrain explicitly with `scripts/export_parity_fixtures.py`); a missing or unusable file is
replaced only by the export of a bundle that is valid now. The drift test now checks, on every
checkout, that the committed weights' embedded lexicons are the repo's, and compares against a
local bundle only when that bundle is current.
**Evidence.** `tests/classifier/test_web_serving.py` (11: whole-verdict parity with the sklearn
bundle — risk within 1e-6, identical tags and evidence; tampered lexicon and foreign matcher
refused; auto/web/bundle policy; health field) and three bootstrap tests (never overwritten,
replaced only when missing/unusable, no retrain when the browser weights serve). `pytest`
1250 passed / 11 skipped, `npm test` 60/60. Live server: `linear_model_source: "web"`; the RU
scam call → `linear`, risk 1.000; an English TeamViewer scam → 1.000 with `remote_access` (the
D54 cue); the demo hard-negative bank call → 0.007.
**Not done.** The Hub bundle and corpus are still the 2026-09-24 ones; uploading the current
bundle, corpus and cards needs the maintainers' token (outward-facing). Until then a fresh
clone evaluates the committed weights on the older local splits.

### D56 — Measured, not fixed: a bank's own "never share the SMS code" warning can trip the meter (2026-09-29)
Probing the pulled model with hand-written legitimate calls (written for this check, not from
any split) through the Python live pipeline (1:1 with the browser):

| probe (legitimate) | meter per utterance | window risk |
|---|---|---|
| card ready · «мы никогда не спрашиваем код из СМС и ПИН-код» | 44 · 41 · **61 latched (High)** · 61 · 61 | 0.143 |
| new-device notice · «Никому не сообщайте код из СМС, даже сотрудникам банка» | 6 · 12 · 37 · **69 latched** · **84 Critical** | 0.986 |
| the same with Latin «SMS» | identical | 0.998 |

Two mechanisms. (1) The meter's hard-signal floor (61/81) ignores the reassurance counter-signal:
in the first call the risk head is right (0.143, the reassurance feature fired) but the bare-mention
cue «код из СМС» alone latches High. (2) «Никому не сообщайте … код» is not reassurance vocabulary,
so in the second call the risk head itself says scam. The Latin spelling behaves identically, so
this predates D51; D51 extended it to the Cyrillic spelling real Russian uses.
**Why no rule change now.** A static count (no model) of dialogues whose cues fire *only* inside
reassurance-bearing utterances: train — 3 of the 14 legitimate rows that contain any cue (918
legitimate in total) vs 2 of 180 cue-bearing scams; val — no legitimate row contains a cue, 0 of 11
cue-bearing scams; **test — no legitimate row contains a cue, 2 of 11 cue-bearing scams**. Muting the floor
inside reassurance would therefore fix nothing measurable on the evaluation splits and cost the
instant alert on the "we never ask for your PIN — just read me the SMS code" scams, a known script.
And «никому не сообщайте код» is itself scammer phrasing («…кроме меня»), so adding it as
reassurance is exactly the lexicon-only shape D27 and D43 warn against.
**The finding is a data gap:** only 14 of 918 legitimate training calls contain any hard-signal
cue, so the model has barely seen a real bank reading out its own security warning. The fix is
the D27/D42 method: Gemini-generated legitimate bank/operator calls carrying the standard
warnings (ru/kk/mixed/en; Latin and Cyrillic SMS), then re-measure whether a negation-aware
floor is still needed. It needs a Gemini key (not available in this workspace) and must not use
Claude-written text for training (`shift` is Claude-authored). Until then: the demo's
hard-negative call is unaffected (0.007), and this probe set should become a small,
ledger-tracked evaluation file once a second author reviews it.

### D57 — The image builds and starts on Railway, not only on a local BuildKit (2026-10-01)
The deploy of `2d0354b` (PR #8, the hardened image) failed on Railway. The old single-stage
image had deployed there. Three things in the new `Dockerfile` and entrypoint are specific to a
local Docker setup:
1. **BuildKit cache mounts.** There were three: the pip, Hub and Vosk caches. Railway refuses
   any `--mount=type=cache` whose id is not `s/<service-id>-<path>` (*"Cache mount ID is not
   prefixed with cache key"*, docs.railway.com/builds/dockerfiles). This was the build failure.
2. **`COPY --exclude`.** The `moby/buildkit:v0.12.5` frontend rejects it: *"dockerfile parse
   error on line 64: unknown flag: exclude"* (reproduced). `HEALTHCHECK --start-interval` has the
   same portability problem.
3. **A non-root image on a platform volume.** Railway mounts volumes root-owned. Run as uid
   10001, the entrypoint's first `cp` into the volume fails (*"Permission denied"*) and the
   container exits at once, which Railway reports as a crash loop. Reproduced with a root-owned
   volume.

**Change.** The cache mounts and `--exclude` are gone. `PIP_NO_CACHE_DIR=1` is set, and the
bootstrap `RUN` deletes its download caches in its own layer, so the image size is unchanged
(1.73 GB vs 1.70 GB). One `COPY site` replaces the two-layer split. `--start-interval` is gone
from `HEALTHCHECK`; compose keeps its own. The entrypoint, when started as root
(`RAILWAY_RUN_UID=0`), gives the state directory to uid 10001 and re-executes itself through
`setpriv`, so the server never runs as root. Run as non-root on a directory it cannot write, it
stops with exit 78 and a message that names the fix. `.dockerignore` drops
`site/models/qorgan`, the D18-rejected static graph (282 MB of local build context).
`docs/DEPLOY.md` §8 is the Railway runbook. `tests/test_dockerfile_portability.py` pins these
rules: it fails 4 of 5 on the `2d0354b` files and passes 5 of 5 now.
**Evidence.** The build is from a clean clone of `2d0354b` plus this change, so it has no local
models, like Railway's clone. It builds on `moby/buildkit:v0.12.5` (linux/amd64), which rejected
the original, and on buildx 0.37. The runtime was checked on native arm64, with secrets as env
vars and `PORT=8080`:
- no volume: healthy in 22 s, server uid 10001, `/api/admin/overview` 200, report `POST` 201;
- root-owned volume as root: healthy in 22 s, server uid 10001, files in the volume owned by
  10001 with mode 0640; a restart on the same volume is healthy in 4 s ("L2 seeds: present");
- root-owned volume as non-root: exit 78 with the message;
- the compose hardening flags (read-only root filesystem, `cap_drop ALL`, no-new-privileges):
  healthy in 25 s.

`/api/health` reports `linear_model_source: "web"` in every case.
**Not verified here.** No Railway log was read: there is no CLI or token in this workspace. The
cache-mount failure is inferred from Railway's documented rule, not observed. Railway's real
BuildKit version is unknown, which is why the Dockerfile no longer relies on newer syntax.

### D58 — After a deploy, the analyst console reloads itself instead of serving a cached copy (2026-10-01)
After the D57 deploy, the analyst console showed *"Dashboard offline — API returned 401"* with
no sign-in form, although the server was healthy (`/api/admin/*` answers 401 without a key, as
designed). Prod served the current files with `no-cache`; a clean browser got the sign-in.
Three faults, all on the client:
1. **The old service worker served the old page.** Shells v1–v5 were served cache-first
   (stale-while-revalidate). On the first visit after a deploy, the old worker answers with its
   cached `admin.html` and `admin.js`. An `admin.js` cached before the sign-in existed (ADR C4)
   sends no key and prints exactly that message. The v6 worker installs and activates right
   after, but the page on screen stays old until a manual reload. Reproduced in Chromium
   (Playwright): old v4 site → visit → deploy `2d0354b` → visit. The page stays on the old
   "analyst dashboard" after v6 activates.
2. **The language switch loaded the console before sign-in.** `admin-lang.js` applies the
   stored or browser language by firing `change` on the content radios. `admin.js` reloaded on
   every `change`, so a keyless request got a 401, read as a revoked key. A first visit in
   English or Kazakh showed a red *"Your key is no longer accepted — sign in again."*
   Reproduced on prod.
3. **Overlapping loads (latent).** A signed-in reload starts the session's load in the HTML's
   default language, then the stored language starts a second. The slower answer rendered last.
   It had no visible effect on the current seeds: the overview is the same in ru and kk, and
   stats were already fetched in the final language. Its only cost was a duplicate stats request.

**Change.** `sw.js` v7: when activation replaces a cache-first shell (`qorgan-shell-v1…v5`),
the worker reloads its open pages once (`client.navigate`). Replacing a network-first shell
(v6+) reloads nothing. The reload runs **outside** `waitUntil`: a navigation is handled only
once the worker has activated, so awaiting it inside activation deadlocked. The first attempt
did exactly that, and the page hung on "Loading" in Chromium. A content-language change
reloads the console only when `me` is set. `load()` carries a sequence number, and an
overtaken load renders nothing.
**Evidence.** `tests_js/sw.test.mjs` runs the real `sw.js` activate handler in a fake worker
scope where navigation completes only after activation. It caught the deadlock (*"activation
never finished"*), and now passes v4 → reload both pages, first install and v6 → none.
`tests_js/pages.test.mjs` pins the guard and the sequence check. `npm test` 70/70, `pytest`
1311 passed. Chromium, same origin, v4 site → fixed site, one visit: the first render is the
cached old page, then exactly one reload lands on the new console with sign-in. `admin.html`
was requested twice in total and 0 times in the following 6 s, so no loop. Local API with a
throwaway key on a scratch data dir: a fresh visit with a stored `en` or `kk` makes **no**
`/api/admin` request and shows no note; sign-in → `session`, `overview?locale=ru`, `stats` all
200; switching to ҚАЗ → `overview?locale=kk`, `stats?locale=kk`.
**Not covered.** A browser whose worker cached the old page and that never visits again keeps
it; nothing can reach it. Browsers that visited before this change and already moved to v6 are
unaffected: v6 is network-first.

### D59 — Open demo access to the analyst console, off by default (2026-10-01)
The jury has no analyst key, so on the hosted demo it cannot see Level 2. The team decided to
switch the restricted access off **for the demo** and to state in the README that it exists.
**Change.** `QORGAN_ADMIN_OPEN_ACCESS` = `off` (default) | `analyst` | `investigator`. When it
is set, `require_analyst` serves a request carrying **no** key as the reserved identity
`public-demo` (now in `RESERVED_IDS`, so no configured analyst can take it) in that role. The
rest of invariant 7 is untouched:
- every route still goes through `require_analyst`;
- every action is audited, under `public-demo`;
- the audit-chain key is still required (503 without it);
- the role and the stated purpose still gate a full transcript;
- the per-identity rate limits and the hourly open budget still apply, shared by all visitors;
- a presented key keeps its own identity, and a wrong key is still a 401.

Personal keys become optional while the switch is on. `GET /api/admin/session` reports
`open_access`. The page tries a keyless session when it holds no key and skips the sign-in only
when the server reports `open_access: true`. It then shows "open demo access" and the audited
identity, with no sign-out.
**Why a switch, not a deletion.** Requiring keys again is one deleted variable. The default stays
fail-closed, so `tests/test_architecture.py` and the auth tests keep guarding it. The privacy story
is still true for a real deployment.
**Cost, on record.** While it is on, anyone with the URL holds the configured role. With
`investigator` they can open full (scrubbed) transcripts of every report on the server, including
any a real visitor submits on the public demo. Acceptable only while the server holds fabricated
data; switch it off before any real report could arrive (`docs/DEPLOY.md` §7).
**Evidence.** `tests/test_api_admin_open_access.py` (9): off by default → 401; an unknown value
is a config error; the identity is reserved; `analyst` → console 200 and a full transcript 403,
with the refusal audited as `public-demo`; `investigator` → open with a purpose, audited as
`(public-demo, pattern_review)`; a personal key keeps its identity; a wrong key → 401; no
personal keys needed, but no audit key → 503. `tests_js/pages.test.mjs` pins the page's probe.
`pytest` 1320 passed, `npm test` 71/71. Local API with `QORGAN_ADMIN_OPEN_ACCESS=investigator`
in Chromium: a fresh visit opens the console without sign-in (500 incidents, 5 organizations),
the `/api/admin` requests are session, overview and stats, all 200, and the audit log reads
`public-demo session.start ok:investigator`.

### D60 — The server listens first; Level 2 is seeded in the background (2026-10-01)
The deploy of `36c53fd` (PR #10) built, then failed Railway's health check. Eleven attempts on
`/api/health` over the 5-minute window all got "service unavailable", and the previous deploy
kept serving. Since `QORGAN_NUMBER_HMAC_KEY` was set, every start seeded Level 2 (500
transcripts embedded and clustered) **before** uvicorn bound the port. That took 25 s on a
laptop, but reproduced at `--cpus=0.5` it took **507 s**: longer than the window, so the
container could never be healthy in time. The same gap caused the earlier 502 "Application
failed to respond".
**Change.** `deploy_bootstrap.py --phase serve|seeds|all` (default `all`, so the build and local
use are unchanged). The entrypoint runs `serve`, which is everything except the seeding:
corpus, pool, embedder, model probe, weights and speech models. It then starts `seeds` in the
background and execs uvicorn. Incidents and organizations are written atomically, and the
console already answers `available: false` until they exist. Its empty-state hint now says a
fresh server is preparing the data, instead of telling a jury to run scripts.
**Evidence.** `tests/test_deploy_bootstrap.py`: `serve` provisions everything but the seeds,
`seeds` only seeds, the default runs every step, and the entrypoint backgrounds the seeding
before `exec uvicorn`. `pytest` 1324 passed, `npm test` 71/71. Container at `--cpus=0.5` with
`QORGAN_ADMIN_OPEN_ACCESS=investigator`: healthy after **20 s**, the overview answered
`available: false` at once, seeds ready at **507 s** (500 incidents, 5 organizations), health
200 throughout.
**Cost.** For the first minutes after a deploy, the console shows "no Level-2 analysis yet".
The seeding process is not reaped by uvicorn (PID 1), which leaves one zombie entry after it
exits; harmless. Compose runs with `init: true`, which reaps it.

### D61 — The browser embedder's runtime is self-hosted; large binaries bypass the service worker (2026-10-01)
Reported: "Could not turn on the microphone (embedding worker failed to load (check the model
files and the script imports))". Prod served every file correctly, and Chromium on prod embedded
fine (4.2 s warm-up, 768-d). The failure depends on the browser engine. Two independent defects:
1. **WebKit (Safari, every iPhone browser).** `embed-worker.js` imported transformers.js from
   jsDelivr. `live.html` is cross-origin isolated (COOP/COEP `require-corp`, for threaded WASM),
   and WebKit refuses a worker's cross-origin import under that policy: *"Refused to load
   'https://cdn.jsdelivr.net/…/transformers.min.js' worker because of
   Cross-Origin-Embedder-Policy"*. The worker never started, so the message is exactly the
   reported one. Reproduced on prod with Playwright WebKit. Serving the file same-origin was
   not enough: `/vendor/` responses carried no COEP. WebKit then refused the same-origin
   import, and Chromium blocked the ORT `.mjs` that starts the WASM threads
   (`net::ERR_BLOCKED_BY_RESPONSE`).
2. **Firefox.** The service worker routed `/models/` through itself (cache-first with a cache
   `put`). Firefox stops a service worker about 30 s into an event and cuts off what it is
   still streaming. The 278 MB model failed with *"Error in input stream"* after about 32 s,
   reproducibly on prod. With service workers blocked it loaded (65 s). Reproduced locally
   behind a 5 MB/s proxy: the v7 worker failed at 30.0 s.

**Change.**
- `qorgan.web.embed_runtime` installs `transformers.min.js`,
  `ort-wasm-simd-threaded.jsep.mjs` and `.jsep.wasm` under `site/vendor/transformers/3.8.1/`,
  each checked against its sha256. The bytes are identical to the CDN and to
  `node_modules` 3.8.1, so parity (ADR D32) is untouched.
- `deploy_bootstrap` makes this a **required** step: a build without the runtime fails instead
  of shipping a page that cannot score.
- The worker imports from there and sets `env.backends.onnx.wasm.wasmPaths` to the same
  directory. Nothing in the worker is cross-origin any more.
- `/vendor/` joins `/core/` in `CROSS_ORIGIN_ISOLATED_PREFIXES`.
- `sw.js` v8: `.onnx` and `.wasm` bypass the service worker. transformers.js keeps the model in
  its own Cache API store. `qorgan-models-v2` drops the v1 copy (about 278 MB) on activation.

**Evidence.**
- Tests: `tests/web/test_embed_runtime.py` (pinned install, refusal of a tampered file, hashes
  equal to `node_modules`, nothing cross-origin in the worker); `test_api_limits.py` (COEP on
  `/vendor/`); `sw.test.mjs` (the large binaries are not intercepted, the shell still is).
  `pytest` 1328, `npm test` 72/72.
- Browsers, locally behind a 5 MB/s proxy with the v8 worker in control: Firefox, WebKit and
  Chromium each warmed up and embedded (61–64 s, essentially the download time).
- The microphone button with a fake audio device: Chromium and Firefox reach "Слушаем". WebKit
  gets past the worker and the recogniser to the permission prompt, which headless WebKit
  cannot answer.

**Cost.** The page no longer works offline for the WASM runtimes. They now depend on the HTTP
cache, which revalidates (`no-cache`), and offline mic mode was never verified anyway. The
image gains about 22 MB.

### D62 — English joins the voice mode as a handicapped third recogniser (2026-10-01)
The voice mode recognised only Kazakh and Russian, while English was already a content locale
(D52) and a call language for the classifier (D54). Requested: English in the voice mode, "even a
little". Added `vosk-model-small-en-us-0.15` (41 MB tarball) as a third Vosklet recogniser in the
per-utterance confidence vote.
**The risk, measured first.** On Kazakh/Russian speech the English model returns confident-looking
garbage, for example "the grass to you mean yes" at 0.71 for a Russian sentence. Measured on `val`
(never a held-out split), with every utterance synthesised (macOS voices Aru, Milena, Samantha)
and decoded by all three models (`scripts/spikes/asr_english/capture_tri.py`, 882 kk/ru/mixed + 165
en utterances in `data/asr_capture/tri_val*.jsonl`). The shipped `voteFinal` was replayed with an
English handicap h (English ranks with confidence − h; `vote_sweep.mjs`):

| h | English wins on kk/ru/mixed | English wins on English |
|---|---|---|
| 0 (plain vote) | 4.8 % | 97.0 % |
| 0.10 | 1.6 % | 85.5 % |
| **0.15** | **1.0 %** | **65.5 %** |
| 0.20 | 0.7 % | 40.6 % |

**Downstream, through the live meter** (`meter_replay.py`, the Python pipeline 1:1 with the
browser, served weights):
- English `val` dialogues, voice mode vs. reference text: scams latched **15/15** at h = 0.15,
  the same as the reference and up from **6/15** with English off; legitimate calls 1/8 vs 1/8
  on the reference (the D54 weakness, not added by ASR).
- kk/ru/mixed: at h = 0.15 the vote winner changes on 9 of 882 utterances, all in **2
  legitimate dialogues** (fragments such as "алло с" → "hello"). Replayed: English off, 1 of the
  2 falsely latched; English on, 0 of 2. The other 110 dialogues have identical winners, so
  identical outcomes. h = 0.10 gave the same English result for more kk/ru churn, so 0.15
  ships.

**Change.**
- `voteFinal(hypotheses, preferred, handicap)` ranks with confidence − handicap and reports the
  real confidence, which the meter uses.
- The reducer and the `stop()` flush pass the handicap through.
- `createDeviceAsr({optional})` skips an optional language whose model fails to load, so English
  can never take down kk/ru; a required one is still fatal.
- Config: `QORGAN_VOSK_MODEL_EN` (`""` = off) and `QORGAN_ASR_EN_HANDICAP` (0.15). **Correction,
  same day:** the page reads the *committed* `site/core/qorgan-config.json`, and nothing regenerates
  it at build or start, so these variables take effect only after `python -m qorgan.web.client_config`
  and a commit. A Railway variable alone changes nothing on the page. Setting `""` there only stops
  the bootstrap from packaging the English tarball; the page then fails to load it and continues
  with kk/ru (it is optional). To take English off prod: roll back on Railway, or revert.
  `qorgan-config.json` gains `asr.order` (kk, ru, en) because the writer sorts keys and English
  would otherwise be the first language and win ties; `asr.handicap` and `asr.optional` are
  added too.
- Bootstrap packages the model. i18n gains `lang_name.en`, `lang_short.en` and
  `mic.model_skipped`; the mic description says English is in test mode. The speech download
  disclosure goes from 106 to 147 MB.
- The Python `vosk_stream` dev harness stays kk/ru.

**Evidence.**
- Tests: `asr.test.mjs` (handicap, kk/ru vote unchanged, the reducer, optional skip vs required
  fatal), `test_client_config.py` (order, handicap, off switch), `pages.test.mjs` (the page
  builds in `asr.order`), `i18n.test.mjs` (every voice-mode language named in every locale — it
  caught the missing `lang_short.en`). `pytest` 1329, `npm test` 78/78.
- Real page in Chromium, synthesised calls through the fake microphone:
  - English scam call: 3 of 5 sentences recognised as English almost verbatim ("read me the code
    now quickly", 100 %), meter 50/100 on a 5-sentence call.
  - Russian scam call: 5/5 "рус", 90–100 %, meter **96/100**.

**Limits, on record.** Synthesised, clean audio, not a speakerphone. About a third of English
utterances still go to kk/ru (the handicap's price). English classification keeps its D54
false-alarm rate on sales calls. "Test mode" in the UI says so.


### D63 (taxonomy v2) — Three-layer taxonomy and a Claude-written generation grid; phase 1 ships the code, the pilot and the plan, not a retrain (2026-10-08)
_Number note: written on branch `feat/taxonomy-v2-datagen`. If another D63 lands first, renumber
this one at merge. Code comments cite "ADR D63"._
_Proposal status: the choices marked **(approve)** need the maintainers' sign-off before phase 2._

**Why.** On `shift`, recall is 0.455, and the cause is the low-arousal pretexts the 15 tactics cannot
name: parcel, clinic and utility "codes", eGov EDS renewal, KGD tax, telecom contract expiry, the mirror
loan, cash couriers, erroneous transfers, task jobs and voice-cloned relatives or bosses
(`docs/RESEARCH_2026-10.md` §2.3, `docs/research/2026-10/model_taxonomy.md` Part B). Nineteen
observed caller roles do not fit three `impersonation_*` ids. The team has no LLM API budget, so new
data is written by Claude. The pilot (below) puts a number on the gap: the shipped model catches
**0 / 27** low-arousal pilot scams.

**Layer 1 — tactics (what the heads learn).**
- `data/taxonomy/tactics.yaml` becomes **version 2**. It holds two label spaces in one file:
  - `v1` is the 15 shipped ids.
  - `v2` keeps the 12 non-impersonation ids, replaces the three `impersonation_*` with `imp_bank`,
    `imp_law_enforcement`, `imp_gov_service`, `imp_regulator`, `imp_telecom_delivery_utility` and
    `imp_personal`, and adds `loan_inducement`, `cash_courier_handoff`, `refund_request`,
    `task_advance_fee` and `callback_instruction`. That makes **23 heads**.
- `active_label_space: v1` stays until the phase-2 retrain. `Taxonomy.tactic_ids()`, `hard_signal_ids()`
  and `active_tactics()` describe the active space, so training, decoding, evaluation, the mock heuristic
  and the LLM prompt are byte-identical today. A test pins `weights.json::label_space ==
  tactic_ids()`.
- Display names, advice and report/partner/intake validation run over `all_tactic_ids()`, so v2 tags
  already render and validate. The retrain flips one line.
- **(approve)** No v2 tactic is a hard signal. The cue block stays at 5 columns; promoting, say,
  `cash_courier_handoff` would be its own ADR with a retrain.

**Negatives.** Six new hard-negative classes, each the legitimate twin of a new scheme:
`legit_delivery_code`, `legit_refund_request`, `legit_employer_request`, `legit_clinic_appointment`,
`legit_telecom_offer` and `legit_utility_notice`. Negatives carry `label_spaces` too, so the v1 Gemini
generator (`generate.py`) still writes only the five v1 classes and a full regeneration of the existing
corpus stays reproducible (D54).

**Layer 2 — role slot.**
- `data/lexicon/roles.yaml` lists 19 roles. Each has an `imp_tactic`, reviewed RU (instrumental) / KK
  (nominative) / EN names and claim-shaped cues.
- Matching uses the same `find_cue` as the hard-signal cues, in both runtimes. The first cue per role
  wins, a longer overlapping span beats a shorter one, and the earliest claim wins.
- `explain()` gains `role_id`, `role_span` (verbatim) and `role_line`, rendered from the reviewed
  `role_template` («Звонящий представился {role}: «{span}».» / «Қоңырау шалушы өзін {role} деп
  таныстырды: …» / "The caller claimed to be {role}: …"). The reason string is unchanged.
- `site/core/role.js` is the 1:1 port. It is pinned by `tests_js/fixtures/roles.json`
  (`scripts/export_role_fixture.py`, 62 cases) and drift-checked from Python.
- **This lexicon is display-only. It is not a feature, not in the bundle hash
  (`weights.json::lexicon` = cue + reassurance only — tested) and editing it does NOT trigger the
  "lexicon edit forces a retrain" rule.** Its only other consumers are the scheme rules and the
  one-time label migration, whose output is frozen into the corpus.

**Layer 3 — schemes.**
- `data/taxonomy/schemes.yaml` defines 21 rule-derived schemes in priority order, in a five-operator
  rule language (`all` / `any` / `tactic` / `any_tactic` / `role`) evaluated by `qorgan.explain.schemes`.
  The first match wins.
- Advice is keyed by scheme (`advice_{ru,kk,en}.yaml::scheme_advice`, 21 × 3 reviewed lines). Eleven
  new per-tactic lines are added for each locale.
- The shipped v1 tags are read through the migration below, so the layer works before the retrain.
- A test proves every scheme's generation spec derives back to itself. It caught three ordering
  bugs while writing: prize vs benefits, mule vs fake boss, and egov vs clinic.
- **(approve)** It is not yet in the UI or the JS core. Without v2 heads its input is v1 tags. Phase 2
  ports it with fixtures once the heads exist.

**Migration of existing labels — (approve).**
- `qorgan.data.migrate_v2` is deterministic. Each `impersonation_*` maps to the successors whose
  family matches a role the transcript claims. Otherwise it maps to the first successor (bank →
  `imp_bank`, gov_police → `imp_law_enforcement`, telecom_delivery → `imp_telecom_delivery_utility`)
  and the row goes on a review list (ids, reasons and roles only, no text).
- Dry run on train: 686 rows carry an impersonation tag. Transitions: bank → imp_bank 404,
  → imp_regulator 44, → both 2; gov_police → law 116, → regulator 6; telecom → telecom 110, → gov 4.
- **246 rows go to review**: 146 bank and 72 gov_police with no role found, 10 role outside the family,
  18 telecom with no role.
- Val: 49 rows with an impersonation tag, 11 to review.
- The list is `data/taxonomy/migration_v2_review.json`. A human pass over it (or a sampled audit) comes
  before phase-2 training.
- Held-out splits are migrated by the same function with **aggregate counts only**; their review rows
  are never opened.
- **The five new tactics are not inferred on old rows.** An old row is *unlabelled* for them, not
  negative. The phase-2 trainer masks v1-sourced rows out of those five heads ("label masking")
  instead of guessing.

**Thin heads — (approve).** D30 stays as written: a tactic with < 8 tuning positives keeps 0.5. Phase-2
acceptance additionally requires ≥ 8 out-of-fold positives for **every** v2 id. A head that still
misses that is dropped from the v2 space (a yaml edit plus a note here) rather than shipped at an
untuned 0.5. No untuned head reaches users.

**Generation pipeline (`python -m qorgan.data.gen`).**
- `configs/gen_grid.yaml` defines a seeded, quota-exact grid over:
  - scheme or negative class, and role;
  - register: kk 30 %, ru 30 %, mixed 25 %, en 15 %; negatives ≥ 35 %;
  - persona (age band, region, occupation), outcome, channel, stage and an explicit length target;
  - direct vs indirect phrasing, and confusable negatives.
- Scam tactics are the scheme's core plus 0–2 optional, kept only if the rules still derive the cell's
  scheme.
- Chunk specs carry instructions, the JSON schema and an English glossary of names and descriptions.
  They never carry example phrases or cue lists, and no Russian or Kazakh lexicon phrase reaches the
  writer (tested). English definitions unavoidably share a few words with English cues.
- Rows are the existing `Dialogue` plus per-utterance `stage` and verbatim `{id, phrase}` tactic spans.
- QC gates, all automated (`qc.py`):
  1. schema and cell agreement;
  2. scrub fixed point;
  3. e5 near-duplicate, with the threshold calibrated so that no train pair exceeds it;
  4. ROUGE-L < 0.7, computed with a bit-parallel LCS;
  5. disjointness from all eight processed splits, by hash only;
  6. lexicon-leak share per scheme;
  7. a generator-fingerprint LR (char n-grams and e5), also restricted to v1 topics.
- No new dependency is needed (numpy and sklearn are already in), so no `tooling` extra was added.
  semhash is unnecessary at this size.

**Pilot (120 dialogues, 3 sub-agents; full report `docs/gen_pilot_qc_2026-10.md`).**
- Gates 1–5: 120/120 valid, 0 scrub, 0 near-duplicates (threshold 0.974; the 0.92 starting point is
  meaningless for e5 because train's own median nearest neighbour is 0.937), 0 ROUGE-L flags (max
  0.307), 0 eval overlaps.
- Cue-free scams 64/72 = 0.889. Scheme agreement 72/72. Role-slot recall 0.42.
- **Fingerprint AUC 0.975 (char) / 0.948 (e5); on shared v1 topics, 0.91 / 0.91.** The style leak is
  real, so §3.2's held-out redesign is mandatory, not optional.
- Shipped v1 model on the pilot (diagnostic): recall **0.208 [0.122, 0.320]**, 0/27 on the low-arousal
  schemes, FPR 1/48.

**Not done, on purpose.** Nothing is added to the corpus and nothing is retrained. `parity.json`,
`runtime_gate*` and weights are untouched (they regenerate after the retrain). The scheme layer is not
in the UI. `live.js` and `asr.js` are untouched.

**Phase 2 (after review).**
1. Fix the three grid issues (persona coherence, role-consistent optional `imp_*`, stage vs core).
2. Freeze `shift` as `shift_v1` (validation). Build `shift_local` (KazLLM, offline) and `court_authored`
   as the honest held-outs **before** training on Claude rows.
3. Generate about 1,600–2,000 rows: about 70 chunks of 25, written by sub-agents in batches of ≤ 16.
   About 60 % scams over 21 schemes (≥ 40 per new tactic, ≥ 8 OOF positives per head) and about 40 %
   negatives (≥ 60 per v2 guardrail class, the reassurance counter-signal in ~40 %, D42).
4. QC every chunk. Human-review the migration list.
5. Build the corpus with v2 labels and masking, flip `active_label_space`, retrain on device
   embeddings, export, and regenerate fixtures and the browser gate.
6. Gates: FPR first on test / authored / ood / styled / browser, plus the new held-outs.
7. Port the scheme layer to JS with fixtures. Add any v2 cue additions only from train, paired with
   negatives, in a separate measured step.

**Rejected.**
- Renaming the ids in place with a flag day. It would break the shipped weights' label space and every
  stored report.
- Inferring the 5 new tactics on old rows by keyword. That writes the lexicon into the labels.
- Making the role a learned head. It would add FPR exposure for a display string.
#### D63 phase 2 — generation, corpus fold-in and a dry-run retrain (2026-10-09/10)
_Approved by the user on 2026-10-09: the six phase-1 "(approve)" items, all as proposed. No
held-out split was evaluated; nothing shipped; `site/models/weights.json` is unchanged._

**Grid v2** (`configs/gen_grid.yaml`). Four changes:
- age band is drawn first, then an occupation that fits it;
- an optional `imp_*` tactic is drawn only when it is the role's own `imp_tactic`;
- `hook_only` is allowed only when every tactic of the cell is hook-compatible;
- the scam share is 0.48, the train base rate (896/1880 = 0.477). The pilot used 0.60.

A new field, `lookalike_of`, gives 80 % of the negatives the pretext of a scheme in proportion to
that scheme's scam count. Each such negative is that scheme's `twin` class, with the scheme's role
when that role can call legitimately (police yes; regulator and antifraud never).

**Role slot.** Self-introduction by organisation. Python and `role.js` are 1:1, and the fixture
was regenerated. Each role gets `orgs`, plus intro markers and Kazakh ablative affixes. A claim is
an organisation with a marker within `ROLE_INTRO_WINDOW` = 4 tokens on the same line, or a
line-opening ablative. The longest organisation wins. Personal names are never matched. The lexicon
stays display-only.

| | role recall | fires on role-less rows | wrong role |
|---|---|---|---|
| pilot, before → after (designed on it) | 43/102 (0.42) → 75/102 (0.735) | 1/18 → 1/18 | 9 → 4 |
| phase-2 batch, before → after (**not** designed on it) | 399/1136 (0.351) → 749/1136 (0.659) | 31/189 → 47/189 | 70 → 63 |

**Migration review (user item 3).** I read every phase-1 review row: 123 train sources plus their
-asr copies, and 11 val rows. I also read every row whose automatic mapping moved with the
widened role lexicon, 141 sources in all. Each decision has a one-line reason in
`data/taxonomy/migration_v2_review.json`:
- imp_bank 75, imp_law_enforcement 41, imp_telecom_delivery_utility 8, imp_regulator 7,
  imp_gov_service 6;
- 4 rows dropped: a private "investment/financial firm" caller impersonates no institution.

10 of the 141 differ from the automatic mapping. `apply_review` applies the decision per source id.

**Generation.** Run `p2`: 1,800 cells, 72 interleaved chunks of 25, written by writer sub-agents.
Each writer saw only its own chunk specs, and resume state comes from disk
(`python -m qorgan.data.gen status`). Usage limits and network drops killed several writers, so
**19 chunks were skipped** at the coordinator's instruction ("stop at ≥ ~1,400"):
chunk_31, 34, 38, 41, 42, 45, 46, 58, 61–71. Chunks are interleaved samples of the grid, so the
balance holds.
- Written: 53 chunks = 1,325 rows. With the pilot that is **1,445 QC-accepted dialogues**; no row
  was rejected.
- Whole-run QC (`qc --run p2`):
  - schema 1325/1325; 0 eval-split hash overlaps; 0 near-duplicates (threshold 0.9749);
  - ROUGE-L max 0.382;
  - cue-free scams 0.871, no scheme under 0.30;
  - scheme rules 636/636;
  - 46 of 689 negatives fire a hard-signal cue (confusables, as designed).
- Language split, scam/legit: kk 213/223, ru 228/238, mixed 161/164, en 106/112.
- **Fingerprint: char AUC 0.9999, e5 0.990. On v1 topics only: 0.9998 / 0.986.** Claude style
  remains perfectly separable from Gemini style. That is why the label balance below matters.
- Every v2 tactic clears the ≥ 40 generation floor:
  - the five new tactics: loan 47, cash courier 42, refund 46, task fee 40, callback 54;
  - imp_personal 57.
- A scrub change is no longer a rejection. Scrubber v2 (D66) masks self-introduced names, and
  the fold scrubs every row. 126 rows changed.

**Corpus fold-in.**
- `build_corpus --gen-run pilot --gen-run p2 --label-space v2 --processed-dir …`:
  - generated rows go to train/val only, never test (`GEN_VAL_FRACTION` = 0.15, seeded id hash);
  - they get ASR-styled copies like any train row;
  - v1 rows are migrated, with the review applied;
  - `provenance.json` records each row's generator.
- Rebuilt first under scrubber v2 (`publish_guard --fix`). ASR copies are now scrubbed after
  styling: styling turned "[NAME]" into "name", which v2 re-masks.
- **P(scam | generator):**

  | split | Claude | Gemini |
  |---|---|---|
  | train (incl. ASR copies) | 1204/2454 = 0.491 | 896/1880 = 0.477 |
  | val | 106/218 = 0.486 | 58/135 = 0.430 |

- `QORGAN_LABEL_SPACE=v2` switches the space for a process. **The yaml stays `v1`**: flipping it
  without new weights would break the shipped model. It flips together with an export.

**Trainer.**
- Label masking: Gemini rows are unlabelled for the five new tactics in fit, OOF and threshold
  tuning (1,880 train rows masked).
- Thin heads: `V2_MIN_OOF_POSITIVES` = 8 (config.py), counted on train + val labelled
  positives. No head was dropped; the smallest are callback 54 and cash courier 42.

**Dry run** (validation only, risk 0.59). Embeddings come from the **ONNX server proxy, not the
device bridge**: inside the agent sandbox, headless Chromium cannot reach its loopback server
(ERR_SOCKET_NOT_CONNECTED), and running it unsandboxed was refused.
- 23 heads, `models/linear_v2_dryrun/web/weights.json`, sha256 `7f1c4299…f0b610`.

| val, threshold 0.59 | FPR [95 % CP] | recall [95 % CP] |
|---|---|---|
| shipped v1, all | 4/189 = 0.021 [0.006, 0.053] | 84/164 = 0.512 [0.433, 0.591] |
| shipped v1, Claude rows | 3/112 = 0.027 [0.006, 0.076] | 26/106 = 0.245 [0.167, 0.338] |
| shipped v1, Gemini rows | 1/77 = 0.013 [0.000, 0.070] | 58/58 = 1.000 [0.938, 1.000] |
| **v2 dry run, all** | **16/189 = 0.085 [0.049, 0.134]** | 145/164 = 0.884 [0.825, 0.929] |
| v2 dry run, Claude rows | 10/112 = 0.089 [0.044, 0.158] | 88/106 = 0.830 [0.745, 0.896] |
| v2 dry run, Gemini rows | 6/77 = 0.078 [0.029, 0.162] | 57/58 = 0.983 [0.908, 1.000] |

Validation threshold sweep for the dry-run model (validation only; nothing is chosen yet):

| threshold | 0.59 | 0.70 | 0.80 | 0.85 | 0.90 |
|---|---|---|---|---|---|
| FPR | 16/189 | 9/189 | 6/189 | 1/189 | 0/189 |
| recall | 145/164 | 136/164 | 121/164 | 110/164 | 100/164 |

**Verdict: not sane to ship at 0.59.** Recall on the new pretexts rises a lot, but validation FPR
quadruples, including on the Gemini rows (1/77 → 6/77). The model would need either a new
operating point, for which 0.85 is where validation FPR returns to baseline, or a fix of the
cause. That choice belongs to the user. Per the protocol, no held-out split was run.

#### D63 phase 2 — diagnosis and PRE-REGISTRATION of the held-out run (2026-10-11)
_Written and committed before any held-out split was evaluated. Everything below was chosen on
train / validation only._

**Device embeddings.** The model is retrained on the browser's own vectors (the device bridge,
transformers.js 3.8.1 / ORT-web wasm). On validation they tell the same story as the ONNX proxy.

**Diagnosis 1: ranking.** v2 ranks better than the shipped model, so the regression is an
operating-point shift.

| validation ROC-AUC / PR-AUC | shipped v1 | v2 (device) |
|---|---|---|
| overall | 0.872 / 0.881 | **0.956 / 0.963** |
| Claude rows | 0.759 / 0.763 | **0.919 / 0.933** |
| Gemini rows | 1.000 / 1.000 | 0.9996 / 0.9994 |

**Diagnosis 2: the 13 validation false positives of v2 at 0.59.**
- 11 are Claude rows and 2 are Gemini rows (one legit_gov_service, one delivery_notification).
- 9 of the 11 Claude false positives are Russian.
- 9 of the 11 are twin / look-alike negatives:
  - legit_delivery_code ×2;
  - one each of legit_refund_request, legit_employer_request, legit_telecom_offer and
    legit_gov_service (eGov);
  - legit_bank_call ×2;
  - general_conversation look-alike of investment_broker ×2.
- 2 of the 13 fire a hard-signal cue.
- The top heads are the pretext's own: imp_telecom_delivery_utility, otp_request,
  refund_request, task_advance_fee, imp_gov_service, investment_scam.

So v2 learned the pretexts. At 0.59 their legitimate twins cross the line.

**Operating-point rule (stated by the orchestrator before computing it).** enter is the lowest
threshold on a 0.01 grid at which v2's validation false positives are ≤ the shipped model's
(4/189), on device embeddings. This gives:
- **enter = 0.78, exit = 0.68**; the single-shot `risk_threshold` is also 0.78;
- **single hard-signal floor = enter + 2 = 80**; double floor = max(81, single) = **81**.
  These come from `meter.hard_signal_floors`; JS reads the same numbers from the config;
- shipped stays at 0.59 / 0.49 with floors 61 / 81.

| validation (device) | FPR | recall |
|---|---|---|
| shipped v1 @ 0.59 | 4/189 = 0.021 [0.006, 0.053] | 86/164 = 0.524 [0.445, 0.603] |
| v2 @ 0.59 | 13/189 = 0.069 [0.037, 0.115] | 146/164 = 0.890 [0.832, 0.934] |
| **v2 @ 0.78 (rule)** | **4/189 = 0.021 [0.006, 0.053]** | **132/164 = 0.805 [0.736, 0.863]** |
| v2 @ 0.78, Claude rows | 3/112 | 75/106 |
| v2 @ 0.78, Gemini rows | 1/77 | 57/58 |

**Live meter on validation** (`eval.stream`, ONNX proxy for both models, each at its own point):
- shipped @ 0.59/0.49: false-latch **0.085 [0.049, 0.134]** (16/189), alert-hit 0.573;
- v2 @ 0.78/0.68: false-latch **0.138 [0.092, 0.195]** (26/189), alert-hit 0.848.

**The streaming false-latch rate is worse for v2 on validation even at the rule point.** This is
recorded here before the held-out run. The ship criteria below are the orchestrator's and do not
include it; it goes to the user as a known risk.

**Frozen model** (`models/linear_v2_device`):
- 23 heads; no thin head dropped. The minimum out-of-fold positive count is ≥ 8 for all; the
  smallest are cash_courier_handoff 42 and callback_instruction 54;
- `_RISK_C` = 4.0 (unchanged); label masking on; per-tactic thresholds re-fit by D30 (OOF +
  val), 0.55–0.70;
- embed_backend `device`;
- `web/weights.json` sha256 **7fc405ac52057d3ed4188a748884b6f1a8ffb17d5bf57917674c9c4e22b2bf92**;
- meter settings otherwise unchanged (alpha 0.5/0.12, min turns to arm 3).

0 of the 1,445 generated rows hash-collide with `shift_local`.

**Ship criteria (the orchestrator's, fixed before looking):**
1. FPR does not rise on test / authored_heldout / ood / ASR-styled / the browser gate.
2. shift_local FPR stays within the baseline's 0.062 (4/64) plus one false positive.
3. shift_local recall improves on the baseline's 0.548.

If any fails: report and stop, no iteration.

**Role line, extra fires on role-less rows (31 → 47/189 on the batch).** Of the 16 new fires,
14 are wrong:
- Kaspi named as a channel, not as the caller: "Kaspi support", a line-opening «Kaspi-ден»
  transfer, «Kaspi-ге»;
- these sit in erroneous_transfer ×7, legit_refund_request ×5, prize ×1 and
  general_conversation ×1 rows.

The other 2 are OLX buyers who name the listing site; that is factual. The lexicon is
display-only, so these only mislabel the role line. Dropping the markers "support" / "here" and
requiring a marker for the ablative is a later display-only fix, re-measured on a new batch.

#### D63 phase 2 — the one-shot held-out result (2026-10-12): the ship criteria FAIL, so v2 is not shipped
_The run used the frozen model and operating point of the pre-registration commit (f038cde).
Shipped v1 ran at 0.59/0.49 and v2 at 0.78/0.68, in the same harness, on device embeddings.
The streams ran on the ONNX server proxy for both models: on the device bridge, v1's streams
alone took more than 10 h. Numbers are pasted from `data/synthetic/gen/_heldout/*.log`._

| split | shipped v1: FPR · recall | v2: FPR · recall |
|---|---|---|
| test | 0.000 [0.000, 0.060] · 1.000 | **0.033 [0.004, 0.115]** · 0.976 |
| authored_heldout | 0.000 [0.000, 0.142] · 0.889 | 0.000 [0.000, 0.142] · 0.944 |
| ood | 0.000 [0.000, 0.049] · 0.932 | **0.041 [0.008, 0.114]** · 0.909 |
| adversarial (all scams) | recall 0.936 | recall 0.945 |
| adversarial_legit (legit-sounding scams) | recall 0.862 | recall **0.716** |
| shift_v1 (contaminated, validation only) | 0.000 · 0.455 | 0.000 · 0.727 |
| **shift_local** (headline) | 0.047 [0.010, 0.131] · 0.548 [0.417, 0.675] | 0.047 [0.010, 0.131] · **0.661 [0.530, 0.777]** |
| styled test / authored / ood, FPR | 0.000 / 0.000 / 0.014 | 0.000 / 0.000 / 0.000 |
| stream test, false-latch · alert-hit | 0.050 · 0.988 | 0.050 · 1.000 |
| stream authored, false-latch · alert-hit | 0.125 · 0.889 | 0.000 · 0.833 |
| stream shift_local, false-latch · alert-hit | 0.125 (8/64) · 0.694 | 0.094 (6/64) · 0.774 |
| court_quotes, risk ≥ threshold / latched@1 | 29/47 / 1/47 | 41/47 / 7/47 |
| court_quotes pseudo-calls, latched | 4/7 | 5/7 |

**Criteria verdict:**

| criterion | result |
|---|---|
| 1. FPR does not rise on test / authored / ood / styled / browser gate | **FAILS** |
| 2. shift_local FPR within 0.062 + one false positive | passes (3/64 both) |
| 3. shift_local recall improves | passes (0.548 → 0.661) |

- Criterion 1 fails on two splits: **test** (0/60 → 2/60) and **ood** (0/74 → 3/74). Authored and
  styled hold.
- The browser gate was not run: it is a parity check of exported weights, and the criteria had
  already failed.
- Per protocol: no export, no re-tuning, no data edits aimed at these rows. `weights.json`,
  config defaults and fixtures are unchanged. The user decides.

#### D63 phase 2 — USER DECISION (2026-10-12): ship v2 under an ADR exception if, and only if, three pre-stated checks pass
_Copied verbatim from the decision and committed before any of the checks was run. If any check fails: stop, keep v1, report._

CHECK 1, the nature of the 5 new held-out FPs (test 2, ood 3). Read only those 5 rows, add them to data/anchors/inspection_ledger.yaml, and classify each.
- PASS for a row if it is a legitimate call whose pretext matches a scam pretext: a bank / courier / delivery / telecom / utility / government / employer contact that mentions codes, payments, transfers, accounts, cards, verification, security or personal data.
- FAIL for a row if it is an ordinary call with no such pretext (appointment, family, casual, generic sales without money/credential talk).
- Check 1 passes if ≥ 4 of 5 rows pass AND none is a plain everyday call.
- Report each row's one-line gist (no PII), v1 and v2 risk, and which heads fired. Do not change anything based on what you read.

CHECK 2, export and the browser gate.
- Export v2: export_parity_fixtures + client_config, active_label_space → v2, the new floors/thresholds into qorgan-config.json, sw.js SHELL_CACHE → v11.
- Run the existing int8 browser runtime gate on device embeddings via the bridge (http://127.0.0.1:8765; restart it with `--pages 4` if it is down). It passes under that test's existing assertion; do not loosen it.
- Full `pytest -q` and `npm test` green.

CHECK 3, the demo scenes in Chromium (live.html replay, the three shipped scenes + the code-switch scene).
- RU scam and KK scam must reach the alert band.
- The real bank call must stay LOW (below the new exit threshold, no alert).
- New tactic names, reasons and advice must render in kk/ru/en (no missing i18n/YAML keys).
- Screenshots go to docs/research/2026-10/taxonomy_v2/.

### D64 — Word-level voice mode: live per-word language chips + a GigaAM pass per phrase (2026-10-08)
The voice mode tagged a whole utterance with one language (the per-utterance Vosk vote, D62), so a
code-switched sentence was transcribed by one monolingual model, and English was the weakest leg.
Requested: three recognisers in parallel, a language decision **per word**, better English. The
literature has no support for per-word ROVER across monolingual models; what works is one joint
model for the final text plus a calibrated tagger for the live view. Measurements in
`docs/research/2026-10/stt_wordlevel.md` and `scripts/spikes/asr_wordlevel/*/REPORT.md`; browser
evidence in `docs/research/2026-10/voice_d64/`.

**Change.**
- **Endpoint pass:** at each Vosk endpoint, GigaAM-Multilingual 220M CTC int8 (Sber, MIT, 225 MB)
  decodes the utterance's audio once more, in a dedicated worker running a single-threaded
  sherpa-onnx WASM build (`scripts/sherpa_wasm/build.sh`, pinned v1.13.8 + emsdk 4.0.23,
  reproducible byte for byte, hashes in `asr/gigaam_web.py`). Its text **replaces** the Vosk text,
  and only that text reaches the session and the meter. Per-word kk/ru/en tags come from the word
  LID (`site/core/word-lid.js`: letters first, then a char-trigram LM + lexicon + Viterbi).
- **Live pass:** while the person speaks, the calibrated span vote over the three Vosk recognisers
  (`site/core/span-vote.js`: 13 features, per-model logistic calibrators, 3-state Viterbi, 1.0 s
  freeze) shows provisional words in same-language runs with a chip. English is shown dashed.
- **Fallback:** while GigaAM loads, when it is unavailable, times out or returns nothing, the
  per-utterance vote passes through, marked as the quick transcript. Its English handicap goes
  **0.15 → 0.05**.
- Both ports are 1:1 with the spike code, pinned by golden fixtures (`scripts/export_asr_wordlevel.py`).
  `tests/test_architecture.py` asserts that no module audio passes through has a network primitive,
  and that the GigaAM worker's only fetch is a same-origin GET. Service worker v9 never streams or
  deletes the model's own cache.

**Evidence.**
- Real read speech (FLEURS) WER: GigaAM **kk 0.074 / ru 0.070 / en 0.130** vs the vote's 0.217 /
  0.203 / 0.592. The 600M variant was rejected (collapsed on 5/31 phone code-switch clips, 2.4×
  slower, 866 MB).
- Headless Chromium, no cross-origin isolation: load 1.0 s (0.43 s cached), RTF 0.25 clean / 0.23
  phone-channel, 49/60 browser outputs identical to native (the rest differ by a word),
  endpoint → replaced text median ~2.3 s (max 3.2 s).
- Code-switch TEST (authored, synthesised, tuning on DEV only), word-language accuracy: span vote
  **0.873** vs per-utterance 0.570. Live English recall is only 0.29, hence the dashed display.
- Handicap 0.05 on FLEURS: English wins 0.910 of English utterances (0.545 at 0.15), 0 false wins on
  real kk/ru (0.020 on synthetic Aru code-switch rows).
- Demo scenes, meter before → after: RU scam 98 → 98, KK scam 96 → 96, real bank call 18 → 18.
  Transcripts get better («код из самая с» → «код из смс»). A bug found in the run (an
  un-endpointed recogniser dragged the previous phrase into the next) is fixed, with a test.
- `pytest` 1341 passed / 2 skipped, `npm test` 115/115.

**Costs and limits, on record.**
- Download +239 MB (model + runtime), disclosed on the page before it starts. Renderer RSS peaks
  ~1.15 GB during a call on desktop Chromium, which is a phone risk: no phone measured (B9).
- The sherpa-onnx browser build has no hosted mirror yet. Until one is approved, deployments run
  the fallback (the page says so).
- All audio is synthesised or read speech. TTS English comes out of GigaAM in Cyrillic (a TTS
  artefact, ~100 % Latin on real speech); real Kazakh-accented English needs the team's recordings
  (`docs/RECORDING_SCRIPT.md`). Cue survival on real speech is unmeasured.
- Not done: a GigaAM + VAD mode for Android WebView, where Vosklet cannot run (no cross-origin
  isolation).

### D65 — An independent held-out from a local open model (`shift_local`); `shift` becomes `shift_v1`, validation only (2026-10-10)
**Why.** `shift` (66 calls) was written by Claude. Claude is now becoming a *training* generator
(track 2), and phase 1 showed a classifier tells Claude rows from Gemini rows at AUC 0.975, so the
moment Claude rows enter train `shift` stops measuring cross-generator generalisation.
**Decision.**
- `shift` is documented as **`shift_v1`** from now on (the code split name stays `shift` to avoid
  churn). Once Claude-written data is in train it is a *validation* number, never the headline.
- **`shift_local`** is the new honest headline: calls written offline on the Mac by an open-weights
  model from a different family, with no access to the corpus, prompts, lexicons or any split.
- `court_authored` (track D68 / the team) and `real_asr` (the team's recordings,
  `docs/RECORDING_SCRIPT.md`) are pending; the headline becomes the worst of the three, with
  intervals, as soon as they exist (`docs/RESEARCH_2026-10.md` §3.2).
- **Rules:** nothing is tuned on `shift_local`, no threshold is picked on it, and the lexicon is
  never extended from it (as D35/D43). Rows a person reads go into the inspection ledger. It is
  never published: `publish_guard.NEVER_PUBLISHED` + `assert_publishable` make
  `scripts/hf_upload.py` refuse any allow pattern that would match it (tests), and its raw and
  processed files are gitignored (test). Track-2 prompts must not be built from its briefs.

**How it was built.** `configs/shift_local_briefs.yaml`: 48 one-to-two-sentence briefs (24 scam,
24 legit; 21 legit are look-alikes with the same pretext and none of the asks), written from public
descriptions of the 2024–26 KZ/RU waves without opening the lexicons, corpus or prompts.
`qorgan.data.shift_local_plan` fixes every slot (language × brief × persona × victim outcome ×
length × seed; plan seed 20261009, temperature 0.8, top-p 0.95, ≤ 3 attempts per slot) before any
text exists; the label comes from the brief. `scripts/gen_shift_local.py` runs llama.cpp
(`llama-server` 0.6.0, build 11429, Metal); `qorgan.data.shift_local_checks` accepts or rejects
each attempt on **automated checks only** — format, 8–22 turns, both speakers, turn length, prompt
echo, script/language (Cyrillic vs Latin, Kazakh-letter share, Russian function words), loops,
near-duplicates — never on a classifier score. `python -m qorgan.data.shift_local_set` reuses the
`shift` pipeline (clean, scrub, dedup, proven disjoint from every split incl. `shift`), then
refuses to write unless the file is a scrub fixed point (D45). Provenance with hashes, seeds,
thresholds and rejection counts, no text: `data/provenance/shift_local.json`.

**Generator — one model family, stated plainly.** `mradermacher/Qwen3.5-9B-Kazakh-GGUF` @
`b6810719`, `Qwen3.5-9B-Kazakh.Q4_K_M.gguf`, sha256 `4e3ca9f6…02021e2`, Apache-2.0; a GGUF of
ISSAI's `issai/Qwen3.5-9B-Kazakh` (Qwen3.5 family — neither Gemini nor Claude). **It wrote every
row in every language.** The preferred `issai/LLama-3.1-KazLLM-1.0-8B-GGUF4` is gated
(CC-BY-NC-4.0, access must be requested from the user's own HF account — not done on their
behalf). A second family (`yandex/YandexGPT-5-Lite-8B-instruct-GGUF`, custom licence) was planned
for ru/en and is still registered in the plan, but the link delivered 0.01–7 MB/s for a day
(including an outage and a corrupted first download, fixed with range-verified resumes and a
final sha256 check), so it was dropped. Consequence: `shift_local` carries Qwen3.5-Kazakh's house
style; it is independent of our training generators, not of style in general.

**No code-switched (`mixed`) calls.** Planned at 40. In the pilot (plan seed 7, outside the
split) the Kazakh fine-tune answered in Kazakh whatever the instruction language, and with
native-language directives it produced Russian and English but **never kk–ru code-switching**
(0 / 8 calls, both intra-sentence and speaker-level prompts — it falls back to pure Kazakh). The
acceptance check would have rejected every one, so `mixed` is 0 and stays pending for
`court_authored` / `real_asr`. `mixed` therefore has no independent held-out at all today.

**Accepted / attempts (rejection reasons):** kk scam 24/32 (turn_count 8) · kk legit 25/25 ·
ru scam 24/31 (language 6, turn_count 1) · ru legit 24/33 (language 7, turn_count 2) · en scam
14/18 (turn_count 4) · en legit 15/15. **126 calls** (62 scam / 64 legit; kk 49 · ru 48 · en 29),
4 slots unfilled after 3 attempts. Russian rejections are the model drifting back into Kazakh.

**Label validity.** A seeded 10 % sample (13 rows, `random.Random(65)`) was read only to check
call ↔ brief: 13 / 13 match; two have blurred speaker roles (a scam still). All 13 are in the
inspection ledger, so the harness reports `(clean)` / `(inspected)` rows. No score was looked at
before the sample was read.

**Baseline — the shipped model, evaluated, not tuned** (`QORGAN_CLASSIFIER_BACKEND=linear`,
server ONNX embedder as the documented proxy, `linear_model_source=bundle`, threshold 0.59):

| Split | FPR [95% CI] | Precision | Recall [95% CI] | F1 | PR-AUC [95% CI] | N |
|---|---|---|---|---|---|---|
| shift_local | 0.062 [0.017, 0.152] | 0.895 | 0.548 [0.417, 0.675] | 0.680 | 0.883 [0.808, 0.944] | 126 |
| shift_local (clean) | 0.067 [0.018, 0.162] | 0.879 | 0.547 [0.404, 0.684] | 0.674 | 0.873 [0.795, 0.938] | 113 |
| shift_local (inspected) | 0.000 [0.000, 0.602] | 1.000 | 0.556 [0.212, 0.863] | 0.714 | 0.950 [0.828, 1.000] | 13 |
| en | 0.133 [0.017, 0.405] | 0.800 | 0.571 [0.289, 0.823] | 0.667 | 0.833 [0.610, 0.984] | 29 |
| kk | 0.080 [0.010, 0.260] | 0.846 | 0.458 [0.256, 0.672] | 0.595 | 0.872 [0.742, 0.966] | 49 |
| ru | 0.000 [0.000, 0.142] | 1.000 | 0.625 [0.406, 0.812] | 0.769 | 0.927 [0.834, 0.979] | 48 |

Streaming (`eval.stream`, live meter replayed turn by turn):

| Split | False-Latch Rate [95% CI] | Alert-Hit Rate [95% CI] | Median Turns | P90 Turns | N+ | N- |
|---|---|---|---|---|---|---|
| shift_local | 0.125 [0.056, 0.232] | 0.694 [0.563, 0.804] | 3.000 | 8.000 | 62 | 64 |
| shift_local (clean) | 0.133 [0.059, 0.246] | 0.679 [0.537, 0.801] | 3.000 | 8.000 | 53 | 60 |
| shift_local (inspected) | 0.000 [0.000, 0.602] | 0.778 [0.400, 0.972] | 3.000 | 5.000 | 9 | 4 |

The live meter latches on 8 / 64 legitimate calls at some point (0.125) — twice the
full-transcript FPR; this is the number a citizen would feel.

Read with care: recall 0.548 is in the same range as `shift_v1` (0.455, D54) and far from `test`;
FPR 0.062 is double `shift_v1`'s 0.030, with overlapping intervals; Kazakh recall (0.458) is the
weakest language, as on `shift_v1`. The device-runtime number (`QORGAN_EMBED_BACKEND=device`) was
not run (the Mac's browser was in use by other tracks); the server proxy is what `eval.run`
reports elsewhere too.

**Open (needs a decision):** request KazLLM access and/or fetch YandexGPT to add a second family
for ru/en; a code-switching generator (or the team) for `mixed`; whether 126 is enough — at this
size FPR intervals are ±0.07.

### D66 — The scrubber masks spoken numbers and self-introduced names; 3-year audit retention; SBOM and scans (2026-10-09)
The eGov/NITEC review (`docs/egov/SECURITY_CHECKLIST.md`, `DATA_FLOW_PII.md`) and the legal gap list
(`docs/LEGAL_ASSESSMENT.md` §6, M7 / M4) left code-side gaps. The worst: voice mode writes numbers as
words («восемь семьсот ноль один…»), so a report could carry a full phone, card, IIN or account
number past the digit-only scrubber.

**Scrubber (`data/scrub.py` + `scrub_numbers.py`, `scrub_names.py`, `scrub_tokens.py`; 1:1 JS port
`site/core/scrub-*.js`).**
- *Number runs.* Number words in ru / kk / en, ASR spellings («семсот», «пятдесят») and Kazakh
  without its special letters («сегиз жуз»), optionally mixed with digit groups, separated only
  by spaces, commas or hyphens (up to 3 characters, line breaks included). Each run is composed
  as a listener writes dictation down («семьсот ноль один» = 701, «сегіз жүз жетпіс» = 870,
  «он бір» = 11). A run with a word is masked from **7 digit-equivalents**. 7 is the shortest
  phone number (city); a dictated 6-digit code stays, like the digit rule's OTP. Placeholder by
  shape: 16 `[CARD]`, 12 `[IIN]`, 10/11 starting 7/8 `[PHONE]`, else `[NUMBER]`. A run split by
  the endpointer across two utterances is judged as one and masked line by line.
- *Left alone:* short quantities («два раза», «код из шести цифр», «назовите шесть цифр»),
  amounts (multiplier words such as тысяча / мың / thousand break a run; a following currency
  word or sign keeps it), em-dash prose, dates, times.
- *Digits:* KZ IBAN (`KZ` + 2 digits + 16 alphanumerics, optional spaces) → `[IBAN]`. Digit
  groups split by single spaces or hyphens in a card / IIN / phone shape («940101 300123»,
  «707 123 45 67») are masked; ASCII-glued ids stay fixed points.
- *Names, conservatively.* Strong triggers («меня зовут», «моё имя», «моя фамилия», «атым»,
  «есімім», "my name is") take the next word in any case (ASR is lower case), plus up to two
  capitalised or surname-ending words. Weak triggers («это», "this is", "I'm") need a
  capitalised name and a following «из / с / от» / "from / calling / with"; two words only if
  the second has a surname ending. A stop list keeps organisations and brands out; a rank stays
  visible («это майор [NAME] из полиции»). Only the name is replaced.
- *Addresses are out of scope.* Scammers rarely need the victim's address, and address patterns
  in kk/ru free text collide with branch and organisation names, which are the evidence. On the
  corpus the only address is one synthetic delivery row. The citizen reviews and edits the text
  before sending, and only an audited investigator opens a full transcript. Also not masked:
  addressing someone by name («Здравствуйте, Аида Кайратовна»), IBANs said as words.

**Evidence survives.** Measured by `scripts/scrub_impact.py` over every published split plus
`data/augment/`: **0** rows change their hard-signal cue tactics, **0** cue spans change text, **0**
reassurance flags flip, and **0** gold trigger spans are lost. 38 gold spans (train 26, val 3,
test 8, ood 1) survive in masked form. That needed two changes after the first measurement on
train: "this is Kazakhstan Post calling" and "this is Major Ivanov from…" were read as names.
Flagged phrases and gold spans are now grounded in this order (`data/span_grounding.py`, mirrored
in `report.js`): as-is, then scrubbed alone, then scrubbed in their original context. Before, a
phrase that held PII was dropped. «Назовите код из СМС: [NUMBER]» keeps the request. Parity:
fixture 102 cases, plus a corpus-wide check of 21,714 texts with 0 Python/JS mismatches.

**Impact analysis.** The scrubber runs on: the report path (`prepare_report`; the browser preview
and `finalizeReport`), partner ingress (unscrubbed text → 422, so partners must apply the same
rules), the content-free checks of the audit log, feedback, analyst ids and stored reports,
`build_corpus` / `incident_seed` / `real_intake`, and the publish guard (D45). **It does not run
on scoring.** `/api/analyze` and the browser score raw text, so the served model sees no change
at runtime.
- *Corpus rows the new scrubber changes* (not rebuilt, on purpose): train **286 / 1,880** (125
  scam, 161 legit; 276 `[NAME]`, 11 `[PHONE]`, 7 `[NUMBER]`), val 24 / 135, test 29 / 143,
  authored 0 / 42, ood 18 / 118, shift 14 / 66, adversarial 13 / 109, adversarial_legit 17 / 109.
  Augment: 52 rows in 4 of 5 files. Almost all are names.
- *Score effect on the served weights* (`weights.json`, server ONNX proxy, `qorgan.eval.run` on
  re-scrubbed copies vs the current splits; measured, nothing tuned). FPR: test 0.017 → 0.017,
  val 0.013 → 0.013, ood 0.014 → **0.000**, shift 0.030 → 0.030, adversarial_legit 0 → 0.
  Recall: test 0.988 = , ood 0.909 → 0.932, shift 0.394 → 0.455, adversarial 0.936 = ,
  adversarial_legit 0.908 = . No split or language gets a higher FPR. The ood false positive
  that goes away is a ledger-inspected row.
- *What the scheduled retrain (after taxonomy-v2) must do:* (1) re-scrub `data/augment/*.jsonl`
  and the separately built splits (`ood`, `shift`, `adversarial*`) with
  `python -m qorgan.data.publish_guard --fix`; (2) `python -m qorgan.data.build_corpus`, whose
  manifest now records `scrubber_version: 2`; (3) retrain, export, regenerate fixtures and
  re-gate as usual. No lexicon change is needed: cue matching is unaffected. Until then
  `test_local_published_splits_are_scrubbed` and `test_every_augment_file_is_already_scrubbed`
  **xfail** as "rebuild pending" (the manifest says v1), and `scripts/hf_upload.py` refuses to
  publish (the guard finds the rows).

**Stored records under a stricter scrubber.** A record on disk was checked by the scrubber of its
day. Re-checking it on read would break `qorgan.audit verify` with no tampering, and make the
reports file unreadable (console, deletion by receipt, purge). Loaders now validate with
`STORED_RECORD_CONTEXT`, which skips only the content check; every write is still checked
(`tests/test_scrub_evolution.py`). Stored reports are **not** re-scrubbed: their incident ids
derive from the transcript, so a rewrite would orphan incidents. Reports stored before D66 keep
their old scrubbing until they expire (≤ 180 days). A pilot starts with an empty store.

**Audit retention.** `QORGAN_AUDIT_RETENTION_DAYS`, default and floor 1096 (3 years with a leap
day; lower is refused at startup). The purge and every deletion path are pinned never to touch
`audit_log.jsonl`. A purge leaves it byte-identical and `verify` passing
(`tests/test_audit_retention.py`). Archiving older entries is an operator procedure (WORM plus the
last `head`), never deleting lines (`docs/DEPLOY.md` §4).

**Supply chain** (`docs/egov/security/`).
- *SBOMs:* CycloneDX for the image pins, the npm toolchain and the self-hosted browser runtime
  (`scripts/generate_sbom.py`).
- *pip-audit:* image pins **0 findings**. The dev venv has 61 advisories in 9 packages that are
  not in the image.
- *npm audit:* 2 high, 4 moderate, all in transformers.js 3.8.1's dev tree. Parity-pinned, so
  documented and not bumped; none of it runs in production.
- *bandit:* 2 HIGH fixed (SHA-1 id marked non-cryptographic, same digest; a path-checked zip
  extract annotated). 10 MEDIUM triaged. semgrep over the own JS in `site/` was not run (install did not complete here): open.
- Not touched: the `onnxruntime` / `onnxruntime-node` / transformers.js pins.

**Stays organisational.** KZ hosting and platform access-log retention (M5), whether the audit
log must be kept longer than 3 years, and whether addresses or forms of address must be masked
(lawyer, M1 / M4). Also: MFA/SSO, the ТД ИБ, the incident procedure, the responsible person, the
SOC contract, and CI integration of the scans plus Trivy.

### D67 — A WebView-safe voice mode (GigaAM + an energy VAD) and an Android build path (2026-10-09)
Android WebView, including the one inside eGov Mobile, never grants cross-origin isolation. So
Vosklet (WASM threads + SharedArrayBuffer) cannot start there, and the D64 voice mode dies. Paste
and replay worked. GigaAM's single-threaded sherpa-onnx build does not need isolation (D64), so
the only missing piece was what Vosk's endpointer did: cutting the audio into phrases.

**Change.**
- `voiceMode()` (`site/core/asr.js`) picks one of three modes:
  - **full** (the D64 word-level mode): needs isolation + SharedArrayBuffer and a non-phone.
  - **light**: GigaAM only. It runs whenever the full mode is blocked by isolation,
    SharedArrayBuffer or the phone gate.
  - **off**: no secure context, AudioWorklet, microphone, WebAssembly or Worker.

  The D25 phone gate still holds for the full mode: a phone gets the light mode, even an
  isolated one such as a TWA. `live.html?voice=full|light` forces a mode for evidence runs;
  `full` lifts only the phone gate and cannot make a page isolated.
- Light mode pipeline: `capture-worklet.js` (AudioWorklet, ~43 ms chunks) → audio ring +
  `vad.js` → each segment goes through the **unchanged D64 endpoint pass**
  (`createEndpointPass`: padded span, GigaAM worker, word LID) → the session and meter, the
  same path as D64. `voice-light.js` wires it (`createLightPass` is the testable core,
  `createLightAsr` the browser runtime).
- **VAD:** deterministic energy + zero-crossing, with no model. 30 ms frames. A frame is active
  when its RMS clears an adaptive floor by 12 dB to open a phrase (6 dB to stay in one) and clears
  −50 dBFS. The floor is the minimum level over 2.5 s, capped at −35 dBFS. Frames crossing zero
  fewer than 150 times/s are excluded, which removes mains hum. Inside a phrase, a quiet "hissy"
  frame still counts (fricatives).
  - Onset 150 ms, hangover 600 ms (250 ms past 8 s), hard cut at 15 s, minimum phrase 300 ms,
    pre-roll 150 ms.
  - Parameters were set from first principles and checked only on synthetic signals and the
    demo scenes. Nothing was tuned on held-out data. Silero-VAD was not needed (below).
- **No fallback in light mode:** there is no Vosk text to fall back on. A phrase whose decode
  fails, times out or comes back empty is dropped and reported ("one phrase could not be
  transcribed"); it is never scored as empty text.
- The meter's step uses confidence 1, the same as a pasted transcript, because GigaAM's CTC
  decode exposes none.
- If GigaAM itself cannot load, the voice mode turns off with the reason. Paste and replay stay.
- UI (kk/ru/en in `site/i18n.js`): a light-mode note saying phrases are transcribed after a
  pause and no live words appear, "hearing speech… / transcribing…" while it works,
  light-specific loading notes, and a 239 MB disclosure (no Vosk tarballs) instead of 147 + 239.
- `tests/test_architecture.py`: `capture-worklet.js`, `vad.js` and `voice-light.js` join the
  no-network audio modules. A new test fails if any `site/core` file that touches an audio API,
  the ring or the VAD is missing from that list. The worklet's only outlet is its own port.
- Service worker **v10** precaches the three modules. The taxonomy branch also moves to v10; the
  merge will reconcile.
- **Android:**
  - `android/twa/twa-manifest.json`: Bubblewrap, `fallbackType: customtabs`, never a WebView,
    because WebView would lose isolation.
  - `android/webview-shell/`: a minimal WebView harness shaped like eGov Mobile's mini-service.
  - `site/.well-known/assetlinks.json`: a template with a placeholder fingerprint.
  - `android/README.md`: steps.
  - No signing key was generated. `*.keystore`/`*.jks` are gitignored, and
    `tests/web/test_android_twa.py` checks that none is tracked.

**Evidence.**
- Tests first: `vad.test.mjs` (14), `voice-light.test.mjs` (5) and `voiceMode` cases in
  `asr.test.mjs`. `npm test` 139/139, `pytest` 1346 passed / 2 skipped.
- Browser runs (`docs/research/2026-10/voice_d67/`): real page, desktop Chromium, fake
  microphone, served by `tests_js/tools/static_site.mjs` with isolation **off** (light) or **on**
  (full, the production header set). Second visit, models cached (`--warm`), one sweep, both modes
  under the same machine load. Audio is synthesised (`say`: Milena/Aru, one WAV per line) plus the
  D64 code-switch clips (`cs_mix`).

  | Scene | full, isolated: meter · utterances/lines · speech end → line on screen, median / max | light, not isolated: meter · utterances/lines · segments found | light latency |
  |---|---|---|---|
  | RU scam | 98 · 6/6 · 3.24 / 3.82 s | 98 · 6/6 · 6/6 | 3.03 / 3.39 s |
  | KK scam | 97 · 5/5 · 3.60 / 4.45 s | 97 · 5/5 · 5/5 | 2.91 / 3.62 s |
  | real bank call (legit) | 19 · 4/4 · 3.71 / 4.64 s | 19 · 4/4 · 4/4 | 2.79 / 3.54 s |
  | code-switch `cs_mix` | 92 · 7/7 · 3.52 / 4.86 s | 97 · 7/7 · 7/7 | 2.49 / 3.91 s |

  D64 reported 98 / 96 / 18 / 97 for the same scenes. Every light transcript is GigaAM's
  ("точная расшифровка"); no phrase was dropped.
- **Segmentation:** 22/22 reference lines found, 0 split, 0 merged, 0 spurious segments.
  - Start error median 150–166 ms. That is the 150 ms pre-roll, by design.
  - End error median 10–14 ms. The KK run reads 0.57 / 0.40 s for both, a constant offset that
    comes from the run's timing zero (getUserMedia → first sample), not the VAD.
- **Latency (light):**
  - The VAD closes a phrase 0.62–0.72 s after the speech ends (the 0.6 s hangover).
  - GigaAM decodes in median 1.4–1.8 s in this sweep (1.0–1.1 s in an earlier, quieter sweep);
    scoring takes the rest.
  - The full mode is slower because three Vosk recognisers share the CPU with the decode.
  - From the click to "listening": 2.1–2.3 s (light) vs 3.7–4.1 s (full). GigaAM loads from cache
    in 0.86–0.92 s.
- **Memory:** renderer RSS, read with `ps` over the browser's process tree (dedicated workers live
  in the renderer). This is a different accounting from D64's CDP reading, so compare within this
  table, not with D64's 1.15 GB.
  - Warm: light peak **2.11–2.23 GB**, in-call median 2.09–2.21 GB; full peak **3.13–3.20 GB**,
    in-call median 1.58–3.14 GB (bimodal, depending on GC).
  - Cold RU: light peak 2.50 GB vs full 2.94 GB. A first visit holds download buffers.
  - Two earlier sweeps, not committed, showed the same picture (light peaks 2.06–2.21 GB, one
    1.15 GB; full 1.77–3.25 GB).
  - The light mode saves about 1 GB at peak. It is still ~2.1 GB on desktop: the e5 embedder plus
    GigaAM. That is a real phone risk, and it is unmeasured on a phone.
- **WebView proxy:** the light mode with an Android-WebView user agent (`; wv`) and a 412 px phone
  viewport, `cs_mix`: meter 97, 7/7 lines, latency 1.98 / 2.87 s.
- **No-GigaAM path** (`e2e_voice_unavailable.mjs`, model answered 404): the voice mode turns off
  with "…модель распознавания речи не загрузилась (HTTP 404 …)", the mic radio is disabled, and
  replay still scores 98/100.

**Not done, on record.**
- No emulator run. Installing the needed tooling took too long: the download link measured
  17–46 KB/s. A JDK alone was ~180 MB, plus an Android SDK and a ~1.5 GB arm64 system image, so
  hours. A Homebrew `openjdk@17` install was stopped after 10 min at 4 MB, and `npm i
  @bubblewrap/cli` failed on the network.
- The TWA and WebView shells are configuration and source only, never built. Steps are in
  `android/README.md`.
- Still needs a person:
  - a real Android phone (B9: memory, latency, speakerphone audio);
  - an upload/signing key and its SHA-256 for `assetlinks.json`;
  - the production HTTPS host in `twa-manifest.json`;
  - a decision on whether the D25 phone gate should also cover the light mode once phone numbers
    exist.
- Synthesised, clean audio only. Real speech through a speakerphone, with room noise and
  overlapping talkers, may need silero-vad. The energy VAD gave 0 errors here, so it was not
  justified yet.

### D68 — `court_quotes`: real scammer phrasing from published court verdicts, held out and local-only (2026-10-09)
Every number the project had was synthetic. Court verdicts on telephone fraud report what the
caller said, so they are the one public source of **real** scam phrasing. This ADR records what
could be collected lawfully, how it is stored, and what the shipped model does on it.

**Reachability (checked 2026-10-09 from a Kazakhtelecom IP, identified user agent).**
- **office.sud.kz (KZ judicial acts bank, the only Kazakh-language source):** reachable, no
  robots.txt, but **every search is gated by a reCAPTCHA v2 checkbox**. Automating it would be
  CAPTCHA circumvention, so nothing was harvested. A person can search it by hand.
- **sudrf.ru court sites (RU, official):** reachable, but their robots.txt disallows
  `name=sud_delo` and `name=bsr`, i.e. the case and judgment pages. Not used.
- **sud.gov.kz (KZ Supreme Court press releases):** robots.txt allows news pages (Crawl-delay 10);
  its copyright notice allows re-use with a hyperlink. Plain curl gets an empty reply, a browser
  works. Phone-fraud releases are rare and narrative: 2 passages read, 1 row kept.
- **sudact.ru (RU aggregator of sudrf texts):** robots.txt allows `/regular/doc/<id>/`; its own
  search endpoint (`doc_ajax`) is disallowed, so verdicts were **discovered through a web search
  engine** and only the document pages were fetched, ≥ 12 s apart, 137 pages, no CAPTCHA, no
  login. Its About and personal-data-policy pages contain no terms that forbid this. The earlier
  note in `docs/research/2026-10/datasets.md` (B1) that scraping sudact "violates its ToS" is not
  supported by anything we could find, but it was not contradicted by a written licence either.

**What was built.**
- `src/qorgan/data/court_quotes.py`: from a *criminal* act (title gate) about an impersonation call
  (call + "представился сотрудником/следователем/сыном…" gate), take caller speech: direct quotes
  after a speech verb, and the reported content after «сообщил, что» / «о том, что» / «попросил».
  Spans are verbatim; nothing is rewritten or translated. The rules were written without reading
  `data/lexicon/`.
- PII: every row goes through `scrub_text`; a row that still carries a name with initials, an
  address, a long digit run, an e-mail or a plate-like id is **dropped**, never edited (2 dropped).
  The courts had already masked names (`фио`, `ФИО17 №5`, `«...»`).
- Review: the rules over-collect (victim, relative, courier and witness speech; a construction-fraud
  and a drugs case passed the gate). All 224 candidates were read by the agent and 177 excluded as
  not the caller's words to the victim. `data/manifests/court_quotes.exclusions.json` lists them by
  sha256 prefix only. **A human should spot-check the 47 kept rows.**
- `scripts/court_quotes_build.py` builds `data/raw/court_quotes/quotes.jsonl` (gitignored) and the
  committed `data/manifests/court_quotes.manifest.json` (counts, dates, citations, no text).
- `src/qorgan/eval/court_quotes.py` replays each quote as a fresh one-turn live session
  (`live.session.advance`, exactly the citizen page's first utterance) and reports per-utterance
  risk ≥ threshold, hard-signal cue firing and the meter, with Clopper–Pearson intervals.
- `hf_upload.py` refuses anything that names `court_quotes` (`publish_guard.assert_not_publishing_court_quotes`);
  `.gitignore` names `data/raw/court_quotes/`; both pinned by `tests/data/test_court_quotes.py`.

**Data.** 47 rows (46 sudact.ru, 1 sud.gov.kz) from 7 of 137 sources read, all **Russian**, 0
Kazakh; 42 reported speech, 5 direct quotes; verdict dates 2018-06-17 … 2025-11-27. Clustered: one
grandparent-scam verdict gives 19 rows, so the intervals below (which assume independent rows)
are optimistic. No victim-side legit context was collected: verdicts do not contain legitimate
calls.

**The shipped model on it** (`QORGAN_LINEAR_WEIGHTS=web`, the committed `site/models/weights.json`,
threshold 0.59; `python -m qorgan.eval.court_quotes`):

| Group | N | risk ≥ thr [95% CI] | cue fired [95% CI] | latched after 1 turn | median risk |
|---|---|---|---|---|---|
| all | 47 | 0.617 [0.464, 0.755] (29/47) | 0.021 [0.001, 0.113] (1/47) | 0.021 [0.001, 0.113] (1/47) | 0.690 |
| reported speech | 42 | 0.690 [0.529, 0.824] (29/42) | 0.024 [0.001, 0.126] (1/42) | 0.024 [0.001, 0.126] | 0.778 |
| direct quotes | 5 | 0.000 [0.000, 0.522] (0/5) | 0.000 [0.000, 0.522] (0/5) | 0.000 [0.000, 0.522] | 0.309 |

Pseudo-calls (all kept quotes of one verdict streamed in verdict order — composed, not a real
call): latched 4/7, 0.571 [0.184, 0.901]. The only cue that fired: `otp_request` (1).

**How to read it.**
- **Hard-signal cues almost never fire on real phrasing (1/47).** That is the robust finding.
  Part of it is form (reported speech is third person: «необходимо сообщить код» rather than
  «продиктуйте код»), but most kept rows describe pretexts with no OTP/safe-account wording at
  all (accident, relative, investigator, «декларирование»), the low-arousal gaps
  `RESEARCH_2026-10.md` §1 item 5 predicted.
- **The 0.617 is not a recall comparable to `shift`.** The set has no negatives, so it says nothing
  about FPR, and reported speech carries court-narrative words («якобы», «ДТП», «деньги»,
  «сотрудник полиции») that may lift the embedding risk for reasons a real call would not share.
  The 5 direct quotes, the closest thing to real utterances, all scored below threshold (median
  0.309), but n = 5.
- A single utterance cannot latch the meter unless a cue fires (A6 arms from the 3rd utterance),
  so "latched after 1 turn" equals the cue rate by construction.

**Rules.** Held out: never tune on it, never extend the lexicon from it (D35/D43). It is not in
`data/anchors/inspection_ledger.yaml`, because the ledger only takes ids of committed evaluation
splits (its test enforces that); instead, the whole split is **inspected by construction** (every
row was read at curation), which the manifest's `excluded_by_review` records.

**Legal footing (under `docs/LEGAL_ASSESSMENT.md`; not legal advice).**
- Storage is on the developer's machine in Kazakhstan (PD Law Art. 12(2)). Russian verdicts are
  published by law (ФЗ-262) with names already removed; after scrubbing and the drop rule the rows
  are arguably no longer personal data at all **[I]**. Where they still are, PD Law Art. 7(5)
  allows re-use of lawfully published personal data with a source citation **[S, per
  `DATA_SOURCES.md` §4.4]**: the manifest cites every verdict URL. Court decisions are official
  documents, not copyright objects under ГК РФ ст. 1259(6) **[I, not checked against the KZ
  Copyright Law]**.
- **Unclear, so local-only:** whether redistributing quotes (the Hub is public and cross-border,
  PD Law Arts. 7(3), 7(6), 16) or processing Russian court data in Kazakhstan raises any Russian
  152-ФЗ point **[U]**. The rows never leave `data/raw/`, never reach Gemini (cloud tier off, D49)
  and never the Hub.
- AI Law 230-VIII: evaluation only; no training use, no change to the system's classification.

**Needs the user.**
1. Kazakh: a person searching office.sud.kz by hand (ст. 190 УК РК, «телефон», «қоңырау»), saving
   verdict texts into `data/raw/court_quotes/` for `scripts/court_quotes_build.py` (add a reader for
   their format), so the CAPTCHA is solved by a human, as intended.
2. Volume: sudact's own search would give far more than web-search discovery (6 of 135 fetched pages
   gave a kept row); ask the operator for permission, or for an export.
3. A human spot-check of the 47 rows and a KZ advocate's line on the "local-only" boundary.
