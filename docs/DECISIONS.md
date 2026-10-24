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

**Check 1 result (2026-10-12): FAIL, so stop and keep v1.** Only these 5 rows were read, and
all 5 are in the inspection ledger. Risk is v1 → v2; scoring is whole-transcript on device
embeddings.

| row | gist | v1 → v2 risk | v2 tactic heads over threshold | verdict |
|---|---|---|---|---|
| test `neg_general_conversation_mixed_21` | ex-colleague now at a bank; an unpaid card bonus; sends a link to confirm identity and pick a card for the transfer | 0.538 → 0.949 | none | PASS (bank contact: card, verification, transfer) |
| test `aug_neg_family_money_request_ru_1` | relative "in hospital" asks for 40k tenge by Kaspi today | 0.015 → 0.860 | imp_personal 0.75 | **FAIL** (family call) |
| ood `ood_neg_family_money_request_mixed_0` | acquaintance asks for 50k by Kaspi to cover a stuck supplier payment | 0.058 → 0.805 | payment_redirect 0.63 | **FAIL** (family/friend call) |
| ood `ood_neg_general_conversation_ru_1` | Kaspi support confirms a large transfer to the callee's mother; no asks | 0.073 → 0.814 | none | PASS (bank contact: transfer, security) |
| ood `ood_neg_general_conversation_ru_3` | Kaspi asks to confirm a transaction with an in-app push | 0.508 → 0.798 | none | PASS (bank contact: verification) |

- 3 of 5 rows pass, below the required ≥ 4.
- Two rows are family money requests. Under the check's own wording ("appointment, **family**,
  casual"), those are everyday calls.

Per the decision: checks 2 and 3 were not run, there is no export, and v1 stays shipped. The
role-line fix was conditional on all three checks passing, so it was not made.

Observation (not acted on): the two family false positives are the cost of the new
relative-in-trouble / payment-redirect heads (imp_personal). Legitimate relatives asking for
money by Kaspi now score like the scam. They were in the v1 guardrail class
`family_money_request`, and the generated batch had only 46 family negatives against 35
relative_in_trouble scams.

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

### D70 — Kazakh and Russian only; the live per-word tags survive the endpoint (2026-10-21)
**Context.**
- **Stakeholders (call of 2026-10-21):** Qorğan stays a kk/ru system for Kazakhstan, with no English
  model, to gain accuracy and focus. English is no longer a **call language**. The page's `en`
  **UI locale** stays (judges, foreign residents); it costs the model nothing.
- **User report:** in a live call each word gets a language chip while it is spoken. When the
  phrase ends, the whole line flips to one or two classes and re-labels words that were right.

**Root cause (the bug).** At commit, `asr.js` emitted only the winning recogniser's whole text, so
the live span-vote tags (D64) were thrown away. `endpoint.js` then re-tagged that text from scratch.
- Deployments run the **fallback**: the GigaAM runtime is not hosted yet (PLAN C1). There the text
  is one monolingual Vosk model's output (the RU model cannot write Kazakh letters), so the line
  collapses to one or two runs.
- On the GigaAM path the word LID re-tags GigaAM's text and changes 7.8 % of the words the person
  had already seen.
- D64 intended the replacement. The loss of the live tags was never decided.

**Decisions**
1. **The committed utterance carries the span vote's final words** (`asr.js committedWords`). The
   fallback emits those words and their tags unchanged (`endpoint.js fallbackUtterance`). Their text
   is what the meter scores, because it is the better transcript: WER 0.274 against the vote's
   0.424. Without word-level models nothing changes.
2. **GigaAM's words inherit the live tags** (`site/core/word-align.js`; reference
   `scripts/spikes/asr_wordlevel/spanvote/inherit.py`).
   - Words are aligned by a word edit distance whose substitution cost is 1 − normalised-Levenshtein
     similarity.
   - The word LID's Viterbi then runs with a hint: a paired word (similarity ≥ `min_sim`) costs
     `weight` nats in any language but its live tag.
   - Kazakh-only letters still decide. Unpaired words follow the LID.
   - `min_sim` 0.34 and `weight` 16 were chosen on **DEV**. To make that possible, DEV got GigaAM
     decodes for the first time (`decode_dev_gigaam.py`). TEST was then scored once.
3. **English leaves the voice pipeline.**
   - Gone: the D62 Vosk-EN recogniser, its handicap and its `optional` slot (`config.py`,
     `client_config.py`, `deploy_bootstrap.py`, `.env.example`).
   - The word LID and span vote have two states. A Latin word ("sms", "kaspi") is neutral and
     follows its neighbours.
   - The span-vote calibrators were refit and the vote retuned on DEV rows without English speech
     (`SPANVOTE_LANGS`, default `kk,ru`; `kk,ru,en` reproduces D64, whose
     `spanvote_results.json` is kept). Output: `spanvote_results_kkru.json`.
4. **Word-LID tuning uses the one-standard-error rule.** The spliced synthetic dev set switches
   language every few words, far more often than a call does. So among settings within 1 SE of the
   best, the smoothest is taken. The refit on the current `train` (D66 changed it) otherwise chose
   switch penalty 1.0, with 0.9693 against 0.9668–0.9680 for its neighbours (SE ≈ 0.0026).
   - This was noticed because a unit test («назовите код из смс» tagged «код» kk) failed, not from
     held-out data.
   - «код» occurs 128× in Kazakh and 130× in Russian training text.
   - Chosen: switch 3.0, lexicon weight 2.0.
5. Display:
   - `alignTokens` now lets a hyphenated token take several words and a punctuation token take its
     neighbour's language. Before, one such token removed every chip from the line.
   - English chips are no longer provisional by language, and the page copy says "two" Vosk models.
   - `sw.js` → `qorgan-shell-v12`, adding `core/word-align.js`. **The ui-refresh branch (D69) also
     claims v12; whichever merges second takes v13.**

**Evidence.** The TEST set is `cs_set` (440 code-switched rows, macOS TTS, clean + phone channel),
scored once after DEV tuning, with 1,000 bootstrap resamples.

| Final word tags, TEST | word-language acc | kk / ru | WER |
|---|---|---|---|
| fallback as D64 shipped it (vote text, re-tagged by the LID) | 0.786 [0.766, 0.804] | 0.820 / 0.798 | 0.424 |
| D64 three-model span vote (for reference) | 0.873 [0.859, 0.887] | 0.943 / 0.833 | 0.260 |
| **kk/ru span vote = new fallback** (offline) | **0.875** [0.858, 0.890] | 0.943 / 0.859 | 0.265 |
| kk/ru span vote, live replay with the 0.5 s freeze (what commits) | 0.874 [0.858, 0.889] | 0.931 / 0.873 | 0.274 |

- On kk/ru-only speech (the `no_en` subset) the span vote scores 0.902 [0.888, 0.914], against
  0.803 for the D64 fallback.
- Live flicker: 0.046 tag changes per word with the 0.5 s freeze, against 0.099 in D64 (1.0 s
  freeze, spike REPORT).

| GigaAM path, TEST | word-language acc (kk/ru speech) | live tags changed at the endpoint |
|---|---|---|
| word LID only (D64) | 0.920 [0.910, 0.931] | 7.8 % |
| inherit live tags | 0.931 [0.920, 0.941] | **1.6 %** |

The accuracy intervals overlap, so the accuracy gain is not shown. The result is the 5× drop in
re-labelled words. The DEV `live` words come from calibrators fit on that same DEV, so `weight` 16
may be slightly optimistic.

- **Browser check:** the real page in Chrome with a fake microphone, the same 7 TEST WAVs, no
  GigaAM, so the deployed fallback (`docs/research/2026-10/voice_d70/`).
  - Before: one kk/ru phrase was won by the English model and shown all-`en` («both security team
    says the a committee seems»), and one phrase was committed twice.
  - After: no English, no duplicate, and the lines keep their kk/ru runs.
  - The final meter differs (97.5 vs 82.5, both critical) because the text differs. This is a
    collage, not a labelled call.
  - With the GigaAM runtime installed locally (the hash-pinned D64 build; nothing published), all
    six lines came from GigaAM and kept mixed kk/ru runs (`after-cs_mix_gigaam.json`). That run
    did not measure endpoint latency.
- **Tests:** `npm test` 158/158. pytest 1652 passed, 3 skipped. The new parity fixture is
  `tests_js/fixtures/word_align.json`. `word_lid.json` and `span_vote.json` were regenerated.

**What this does not show.**
- The classifier gates (test / authored / ood / styled / browser) are untouched: they score text,
  not ASR. Decision 1 changes the *text* that reaches the meter in voice mode. It is a better
  transcript, but no labelled-call audio exists to measure the meter on real speech (B1 `real_asr`).
  This is recorded as an exception to "FPR first". It must be re-checked when `real_asr` exists.
- All audio is TTS, and DEV and TEST share voices. English speech now gets kk/ru tags by design.
- Known limits (code review, all low):
  - A line made only of Latin words («kaspi gold») has no neighbour to follow. It ties to `kk`.
  - The span vote keeps the first piece of a hyphenated Vosk word («из-за» → «из»), and that text
    now reaches the meter in the fallback.
  - The empty `handicap` plumbing stays, generic.
  - A deployment that still sets `QORGAN_VOSK_MODEL_EN` has it silently ignored.

**Not decided here.** Whether the classifier's English **training** rows (238 in `train`) stay is
a separate, pre-registered ablation (PLAN_2026-10 §4 A0). D54 saw `shift` improve when English was
added, without isolating the effect, so dropping it untested could cost kk/ru recall.

### D71 — A0: do the English training rows help Kazakh and Russian? Pre-registration (2026-10-22)
**Context.** Since D70 English is not a call language. The current `train` split holds 238 English
dialogues (172 scam, 66 legit) of 1,880, and `val` holds 23 of 135. D54 saw `shift` improve when
English arrived, but three things changed together, so whether these rows help kk/ru recall is
unknown. This entry is committed **before** any run, per PLAN_2026-10 rule 3.

**Arms.** Same code, same corpus build, device embeddings (`QORGAN_EMBED_BACKEND=device`, the
browser bridge, D33), same seeds:
- **B (baseline):** `linear_train` on today's `train` + `val`, all rows.
- **A (ablation):** the same with every `language == "en"` row removed from `train` and `val`.

The shipped v1 `site/models/weights.json` is reported for reference only. It was trained on an older
corpus build, so it is not an arm.

**Step 1: validation only.** Score both arms on the `val` rows in kk / ru / mixed (112 dialogues).
Metrics: FPR and recall at the shipped alert threshold 0.59, ROC-AUC, per language. A passes
step 1 if:
- (a) A's FPR ≤ B's FPR; and
- (b) A's recall ≥ B's recall − 0.03.

**Step 2: one held-out look, only if step 1 passes.**
- Before running it, record the sha256 of both arms' weights in this entry.
- Then score both arms once on the kk + ru rows of `shift_local` (97 calls; English rows
  excluded). A passes step 2 if:
  - (a) A's false alarms ≤ B's false alarms (counts); and
  - (b) A's caught scams ≥ B's − 1.

**Decision, fixed now.**
- **Steps 1 and 2 both pass:** English rows leave the training corpus (corpus config). v2.1 (A2–A6)
  is built without them.
- **Otherwise:** they stay as training-only rows. English remains absent from the product, the
  evaluation headlines and the voice mode.
- **No re-tuning after the look.** Whatever the result, `shift_local` is not used again for this
  question.

**Step 1 result (2026-10-22, `qorgan.eval.run --split val --by-language`, device embeddings).**
Both arms score FPR 0.000 and recall 1.000 on kk (n=30), mixed (38) and ru (44): validation is
saturated (same generator as train), so step 1 passes without telling the arms apart. Per-tactic F1
moves both ways by small amounts (e.g. kk credentials_request 0.000 → 0.333; mixed urgency 0.706 →
0.667).

**Weights frozen before step 2 (sha256):**
- A: `risk_clf.joblib` a1eeee6eac1c53cb4308dfa1bc80a733ead7cbc9d70f6ba5daa06de4b221ab3b ·
  `tactic_clf.joblib` 480b5f589a6b87c2b33f091396a0fd70d9cb62c0e46b2ea6c99f3da0c48a685f ·
  `web/weights.json` 285db215747a0e4d67b8ca8b6252e5b61661f04a081cbe78be0b3d3224e38adc
- B: `risk_clf.joblib` 863cf52559cebe014c47eac7320b6f45635d07c5d1931a8cf551f0259ceae5b2 ·
  `tactic_clf.joblib` 5c63e2426f0d3cf558e014e666354935f805a38e3e275491064691a0bde134e1 ·
  `web/weights.json` aaa1fd4f765c7ec46dda8507ea4df1f3debeb5a230b6872537c999b534ff56a2

**Step 2 result (2026-10-22, one look, `qorgan.eval.run --split shift_local --by-language` on the 97
kk+ru rows, device embeddings, the frozen weights above).**
A first attempt printed no metrics for either arm (the browser bridge died when the Mac slept), so
this run is the only look.

| arm | FPR | recall | kk FPR / recall | ru FPR / recall | PR-AUC |
|---|---|---|---|---|---|
| B (with English rows) | 0.041 [0.005, 0.140] (2/49) | 0.542 [0.392, 0.686] (26/48) | 0.080 / 0.417 | 0.000 / 0.667 | 0.904 |
| A (without) | 0.041 [0.005, 0.140] (2/49) | 0.479 [0.333, 0.628] (23/48) | 0.080 / 0.458 | 0.000 / 0.500 | 0.905 |

- (a) passes: 2 ≤ 2 false alarms.
- (b) **fails**: 23 < 26 − 1 caught scams. The loss is Russian (16 → 12 of 24); Kazakh gains one
  (10 → 11 of 24). The intervals overlap; the rule was fixed in advance, so it decides.

**Decision: the English rows stay in `train` as training-only data.** English stays out of the
product, the voice mode and the evaluation headlines (D70). v2.1 (A2–A6) keeps them. The English
generation slots of A2 still go to Kazakh, which this does not contradict: those are *new* rows.
`shift_local` is not used again for this question.

### D63 v2.1 / A1 — the role line no longer reads a payment channel as the caller's role (2026-10-22)
Display-only; no retrain (`roles.yaml` is not in the hashed lexicons;
`test_role_lexicon_is_not_part_of_the_trained_bundle`). The yaml and its tests landed early in
3cb7af5 by accident (a `git apply --3way` left them staged); this entry is the implementation.

**Rules.**
- An organisation is a claim only when an introduction marker sits within `ROLE_INTRO_WINDOW` = 4
  tokens of it **in the same sentence** (`ROLE_SENTENCE_END` = `.!?` is a wall).
- A bare Kazakh ablative («Kaspi-ден») is no longer enough. "here" and "support" are no longer
  markers. A separate ablative token belongs to the organisation («Kaspi-ден хабарласып»).
- `intro.channel` vetoes an organisation named as a channel or help desk:
  - a preposition right before it («через/на Kaspi», "on Kaspi");
  - "support", «арқылы», «қолдау», «қосымша» or «приложени» right after it;
  - a separate non-ablative case affix («Kaspi-ге», «Kaspi-дің»);
  - a negation between a preceding marker and it («Это не Kaspi»).
- A negation between the organisation and a *following* marker only drops that marker, so
  «служба безопасности Kaspi, не беспокойтесь» still claims.
- Python and `role.js` stay 1:1. The fixture (`tests_js/fixtures/roles.json`) was regenerated with
  88 cases, 13 of them new hand cases.
- «Мам, это я, переведи на Kaspi…» keeps its phase-1 `relative_friend` claim, and no bank claim is
  added. Its test asserts exactly that.

**Measured on the phase-2 batch** (train). The fix was designed on its role-less rows, so this is
**not** independent; re-measure on the next batch.

| p2 batch | role recall | fires on role-less rows | wrong role | no claim |
|---|---|---|---|---|
| before | 749/1136 (0.659) | 47/189 | 63 | 324 |
| after | 738/1136 (0.650) | 33/189 | 54 | 344 |

- All 14 wrong Kaspi / Halyk Invest lines are gone. The 33 left are D63's old 31 cue fires plus
  2 factual OLX buyers.
- **Recall lost:** 14 rows, with 3 gained:
  - 7 bare ablatives with no marker;
  - 3 genitive «X Bank-тің қауіпсіздік қызметі», a cost of the genitive veto kept for «Kaspi-дің өз
    нөміріне…»;
  - 1 sentence wall;
  - 3 channel words.
- The pilot is unchanged: 75/102, 1/18, 4 wrong.
- **Tests:** pytest 1666 passed / 3 skipped; `npm test` 159/159.

### D63 v2.1 / A2 — 300 legitimate dialogues for the v2.1 retrain (run `v21_legit`, 2026-10-22)
**Why.** Two of v2's five new held-out false alarms were ordinary family money requests by Kaspi.
v2.1 therefore adds legitimate family and bank calls. These were chosen after seeing v2's false
alarms on test/ood, so test/ood cannot be the main judge for v2.1; `shift_local_2` is (PLAN §4).

**Grid** (`configs/gen_grid_v21_legit.yaml`, seed 20261022, 300 cells, 15 chunks of 20):
- `scam_share: 0`; registers kk 0.45 / ru 0.30 / mixed 0.25 (no new English rows, D70/D71).
- Classes: family_money_request 120, legit_bank_call 120, and 15 each of delivery code, clinic,
  eGov/ЦОН and utility. 178 are confusable.
- 173 cells are look-alikes: bank_card_phish 36, bank_safe_account 24, mirror_loan 24,
  relative_in_trouble 24, parcel_code 15, and 10 each for clinic, eGov and utility.
- Code (`grid.py`, `spec.py`):
  - `scam_share` may be 0.
  - Per-class `negative_scenarios` (32 English briefs) are spread evenly with a separate RNG, so the
    base grid's manifests and specs are byte-identical.
  - A brief's `lookalike_of` must be the scheme's twin.
  - Tests: `tests/data/gen/test_gen_grid_legit.py` (10).
- No brief asks for a code, PIN or transfer.

**Generation.** 15 writer agents, each seeing only its chunk spec. All passed `check` first time.

**QC.**
- 300/300 schema-valid; **300 accepted**.
- 0 eval-split hash overlaps; 0 near-duplicates (nearest neighbour vs train max 0.940); ROUGE-L max
  0.376.
- A hard-signal cue fires on 16/300 negatives, 12 of them confusables.
- Role-slot recall is 0.51.
- 40 rows are not a scrub fixed point (self-introduced names, 2 rows with numbers read aloud). The
  fold scrubs them (D66).

**Open before A4.**
1. **Style confound.** Fingerprint vs Gemini train: char AUC 1.000, e5 0.984. With this run, the
   Claude block's P(scam) falls to 708/1745 = 0.406, against Gemini train's 0.477.
2. **Reassurance line.** 57 of 120 bank calls carry "we never ask for codes", a line real scammers
   also use, so a head could learn "never-ask-code ⇒ legit".
3. **Persona–scenario coherence** is not enforced (one cell was rewritten by its writer). One
   borderline family row (v21_legit_0100, "new SIM, save my number", then a loan) needs review.

Data is local-only, not folded, not trained.

### D63 v2.1 / A2 (cont.) — the A4 blockers closed: 37 bank calls rewritten, 236 scams added (2026-10-22)
Closes items 1 and 2 of "Open before A4" above. Item 3 (persona coherence, v21_legit_0100) stays open
for the user's review before the fold.

**Reassurance line (item 2).** A caller-side rule (`v21_legit/redo/rule.py`, local) found the line in
57 rows. 20 were kept, and the other 37 were rewritten as chunks 15–16 with a per-cell `constraint`.
The constraint says the call shows it is legitimate only by what it does, never by a disclaimer.
The rows they replace are in `v21_legit/superseded/`. After the redo the rule hits 21 bank calls:
the 20 kept, plus one false match ("если покупка ваша, делать ничего не надо", not a disclaimer).
- On the scam side, `scam_scenarios` gives the line to 28 scam cells. The scammer says it to win
  trust and asks anyway. All 28 carry it, and so do 8 more scams written without the brief.
- **The line is now in 36 scams against 20 legit calls, so it is no longer a legit-only signal.**

**Style confound (item 1).** The scam-only run is `v21_scam` (`configs/gen_grid_v21_scam.yaml`,
seed 20261023, `scam_share: 1.0`, kk 0.45 / ru 0.30 / mixed 0.25).
- It has 236 cells, which puts the Claude block's P(scam) at **944/1981 = 0.4765**. Gemini train
  is 0.4766.
- The nine low-arousal administrative schemes are weighted 1.4× (13 cells each). The other schemes
  get 9–10 each.
- Stages: full 118, hook_escalation 64, extraction_handoff 43, hook_only 11. Phrasing: indirect
  131, direct 105.
- Code (`grid.py`, `spec.py`):
  - `scam_share` may be 1.
  - `ScamScenario` briefs are allocated across schemes by a separate seeded RNG, so grids that use
    none stay byte-identical.
  - A cell `constraint` is a hard per-cell rule.
  - Tests: `tests/data/gen/test_gen_grid_scam.py`.

**Generation.** Seven Opus writer agents wrote two chunks each and read only their chunk specs.
Every chunk passed `check` the first time.

**QC** (`python -m qorgan.data.gen qc`, embedding gates on):

| run | rows accepted | eval overlaps | near-dup (run→train NN max) | ROUGE-L max | negatives with a cue | role recall | fingerprint AUC char / e5 |
|---|---|---|---|---|---|---|---|
| v21_legit (after redo) | 300/300 | 0 | 0 (0.940) | 0.342 | 11 | 0.49 | 1.000 / 0.986 |
| v21_scam | 236/236 | 0 | 0 (0.936) | 0.305 | — | 0.66 | 1.000 / 0.977 |

- v21_scam: the cue-free scam share is 0.775, and no scheme is below its leak target.
  `scheme_rule_agreement` is 236/236.
- Some rows are not a scrub fixed point (legit 32, scam 10), mostly self-introduced names. The fold
  scrubs them (D66).
- The fingerprint AUC stays ≈ 1. Matching the label rate removes the *label* confound. It cannot make
  Claude's text look like Gemini's, and that is why `shift_local_2` is the judge.

**Fresh judge (A3, same day).** `shift_local_2` is 130 calls (kk 80 / ru 50, 65/65). It uses the
same briefs, model and sampling as `shift_local`, with seed 20261022. All 130 slots were accepted by
automated checks only, and it is disjoint from every split, `shift_local` included. Provenance:
`data/provenance/shift_local_2.json`. **No model has been scored on it.**

Data is local-only, not folded, not trained. Next: A4 (pre-registered retrain, device embeddings).

### D63 v2.1 / A4 — PRE-REGISTRATION of the v2.1 retrain and its one-shot held-out run (2026-10-23)
_Committed before any v2.1 model was trained or any held-out split was scored. Everything below is
chosen by the user (2026-10-23) or copied from v2 phase 2 (f038cde) and PLAN_2026-10 §4._

**User decisions (2026-10-23).**
1. `v21_legit_0100` is dropped. It is a legit neighbour loan that opens with "new SIM, save my
   number", which is the new-number relative scam's opening. The drop is in the committed
   `configs/gen_review_rejects.yaml`, which `build_corpus --gen-run` applies after QC
   (`fold.load_review_rejects`; tests in `test_gen_fold.py`).
2. The operating-point rule below is confirmed.

**Corpus.**
- Command: `build_corpus --gen-run pilot --gen-run p2 --gen-run v21_legit --gen-run v21_scam
  --label-space v2 --processed-dir data/cache/processed_v21`. Content hash **2e8b3d5f77df…88e93d**.
- `pilot` and `p2` had been deleted from the main checkout. They were restored from an agent
  worktree; the two surviving copies are byte-identical (120 + 1,325 QC-accepted rows).
- The v1 corpus rebuilds to its recorded hash a183b0e5…, so the inputs are current.
- Train: Claude 3,374 rows (incl. ASR copies), P(scam) 0.4766; Gemini 1,880, P(scam) 0.4766.
- Val: Claude 293 (0.478), Gemini 135 (0.430); 230 legit and 198 scam in total.
- `data/processed` (v1) is untouched.

**Trainer: v2's settings, unchanged.**
- `QORGAN_LABEL_SPACE=v2 QORGAN_EMBED_BACKEND=device python -m qorgan.classifier.linear_train
  --processed-dir data/cache/processed_v21 --out-dir models/linear_v21_device`.
- Device embeddings come from the browser bridge (D33).
- Label masking is on. `V2_MIN_OOF_POSITIVES` = 8. `_RISK_C` = 4.0. Per-tactic thresholds are
  re-fit by D30.
- No setting is changed after seeing a result. If training fails for a technical reason, the fix
  and the rerun are recorded here.

**Operating-point rule (validation only, device embeddings).**
- **enter** is the lowest threshold on a 0.01 grid at which v2.1's validation false positives are
  ≤ the shipped v1's false positives on the **same v2.1 validation set** at 0.59.
- **exit** = enter − 0.10. The single-shot `risk_threshold` = enter.
- The hard-signal floors come from `meter.hard_signal_floors(enter)`.
- Shipped v1 stays at 0.59 / 0.49.

**Freeze.** Before any held-out split is scored, this entry gets the sha256 of v2.1's
`risk_clf.joblib`, `tactic_clf.joblib` and `web/weights.json`, the chosen enter/exit, and the
validation table (v1 vs v2.1, FPR and recall with 95 % CP, by generator and by language). That
commit closes A4.

**A5: one look, device embeddings, both models in the same harness, each at its own point.**
- Splits: `shift_local_2` (the judge; kk / ru), plus test / authored_heldout / ood as **clean**
  subsets (`qorgan.eval.run`'s `(clean)` rows exclude every inspection-ledger id), and the ASR-styled
  copies.
- Reported, but not a ship criterion: `shift_local` (v2 and A0 looked at it), and the
  validation stream false-latch rate (`eval.stream`).

**A6: ship criteria (PLAN_2026-10 §4), all required.**
- (a) FPR does not rise on the clean test / authored / ood subsets or on the styled copies.
- (b) On `shift_local_2`, v2.1's false alarms ≤ v1's + 1, and v2.1's recall > v1's.
- (c) Every new false alarm is an institutional look-alike, never a family or everyday call. Only
  those rows are read, and they go into the inspection ledger.
- After export: the browser gate passes, and the demo scenes are right (the RU and KK scams alert,
  and the real bank call stays LOW).
- **On a fail, v1 stays.** Nothing is re-tuned on what A5 shows.

#### D63 v2.1 / A4 — FREEZE: trained, operating point by rule (2026-10-23)
_Committed before any held-out split is scored. Validation only, device embeddings
(transformers.js 3.8.1 / ORT-web wasm bridge). Numbers pasted from
`scripts/taxonomy_v2_dryrun_eval.py` (`models/linear_v21_device/val_report.json`)._

**Training ran as pre-registered:** no rerun and no setting changed.
- 23 heads, none dropped. The thinnest are task_advance_fee 94, cash_courier_handoff 105 and
  loan_inducement 108 OOF positives.
- 1,880 Gemini train rows are masked for the new tactics.

**Frozen model** (`models/linear_v21_device`, sha256):
- `risk_clf.joblib` 16c6e450b812070ed33a2df641d4e90d969433a9d16bef43d8502f56209b27f7
- `tactic_clf.joblib` 67996f602399be3e47629948d5487dcc0504e44adee91bfb3c7c3fe4c397104c
- `web/weights.json` cd7b47bba7ab2bc4aff0666d316133b1ae4efa0cfea502de58f6ab58affe5e40

**Operating point (the rule):**
- Shipped v1 makes 5/230 false positives on the v2.1 validation set at 0.59.
- **enter = 0.82** is the lowest 0.01 step at which v2.1 makes ≤ 5. **exit = 0.72**; the
  single-shot `risk_threshold` is also 0.82.
- The hard-signal floors from `live.meter.hard_signal_floors` are **84 / 84**.

| validation (device) | ROC-AUC / PR-AUC | FPR [95 % CP] | recall [95 % CP] |
|---|---|---|---|
| shipped v1 @ 0.59 | 0.867 / 0.868 | 5/230 = 0.022 [0.007, 0.050] | 91/198 = 0.460 [0.389, 0.532] |
| v2.1 @ 0.59 (for reference) | 0.958 / 0.962 | 18/230 = 0.078 [0.047, 0.121] | 177/198 = 0.894 [0.843, 0.933] |
| **v2.1 @ 0.82 (rule)** | | **5/230 = 0.022 [0.007, 0.050]** | **147/198 = 0.742 [0.676, 0.802]** |
| v1, Claude rows | 0.783 / 0.766 | 4/153 | 33/140 |
| v2.1 @ 0.82, Claude rows | 0.930 / 0.939 | 4/153 | 90/140 |
| v1, Gemini rows | 1.000 / 1.000 | 1/77 | 58/58 |
| v2.1 @ 0.82, Gemini rows | 1.000 / 0.999 | 1/77 | 57/58 |

v2.1 @ 0.82 by language (Claude rows), each as v1 → v2.1:

| language | FPR | recall |
|---|---|---|
| kk | 1/52 → 1/52 | 10/54 → 34/54 |
| ru | 3/58 → 3/58 | 15/42 → 33/42 |
| mixed | 0/27 → 0/27 | 4/30 → 12/30 |

**The five validation false positives at 0.82** (generated or augment rows, not held-out):
- aug_neg_delivery_notification_ru_0 (Gemini, ru, 0.826)
- p2_0864 (kk, 0.868)
- p2_1105 (ru, 0.863)
- p2_1248 (ru, 0.906)
- p2_1761 (ru, 0.908)

**Compared with v2 phase 2** (different validation sets, so indicative only): v2 needed 0.78 to
match v1's FPs, and v2.1 needs 0.82. On Claude rows v2.1 ranks better than v2 (ROC-AUC 0.930 vs
0.919). **Mixed-language recall at the rule point is the weak spot (12/30).**

A4 is closed. Next is A5: one look, per the pre-registration above.

#### D63 v2.1 / A5–A6 — the one held-out look: A6 (a)–(c) pass (2026-10-23)
_One run on device embeddings, the frozen weights of 799bbe4. Both arms ran in the same harness,
each at its own point: v1 at 0.59/0.49, v2.1 at 0.82/0.72. Outputs:
`models/linear_v21_device/a5/{v1,v21}.{run,styled}.txt`, from `qorgan.eval.run` and
`qorgan.eval.asr_realism`. Nothing was re-run or re-tuned._

**The judge: `shift_local_2`** (130 calls, never seen before this run).

| shift_local_2 | FPR [95 % CP] | recall [95 % CP] | PR-AUC |
|---|---|---|---|
| v1 @ 0.59 | 2/65 = 0.031 [0.004, 0.107] | 28/65 = 0.431 [0.308, 0.560] | 0.871 |
| **v2.1 @ 0.82** | **0/65 = 0.000 [0.000, 0.055]** | **38/65 = 0.585 [0.456, 0.706]** | **0.942** |
| v1, kk / ru | 2/40 / 0/25 | 16/40 / 12/25 | |
| v2.1, kk / ru | 0/40 / 0/25 | 21/40 / 17/25 | |

**Other held-out splits** (FPR, then recall; each as v1 → v2.1):

| split | FPR | recall |
|---|---|---|
| test (clean) | 0/60 → 0/60 | 83/83 → 80/83 |
| authored_heldout (clean) | 0/19 → 0/19 | 16/18 → 17/18 |
| ood (clean) | 0/69 → 0/69 | 41/44 → 40/44 |
| test, ASR-styled | 0/60 → 0/60 | 82/83 → 82/83 |
| authored, ASR-styled | 0/24 → 0/24 | 17/18 → 17/18 |
| ood, ASR-styled | 1/74 → 1/74 | 41/44 → 37/44 |
| `shift_local` (reported only; seen before) | 3/64 → 3/64 | 34/62 → 42/62 |

**A6 verdicts.**
- **(a) passes.** FPR does not rise on any clean subset or styled copy.
- **(b) passes.** On `shift_local_2`, false alarms fall 2 → 0 (the limit was ≤ 3) and recall rises
  28 → 38.
- **(c) passes.** v2.1 adds two false alarms that v1 does not make. Only these two rows were read:
  - `neg_general_conversation_mixed_21` (test, inspected subset, 0.942): already in the ledger since
    2026-10-12. A former colleague now at a bank offers a card bonus with an identity-confirmation
    link. The institutional pretext matches.
  - `ood_neg_legit_gov_service_mixed_2` (ood, styled copy only, 0.828; clean 0.53): added to the
    ledger. A clinic registrar re-attaches the patient to a GP after a system update and asks for
    the IIN "for verification". The institutional pretext matches.
- v2.1 also drops four of v1's false alarms: three in `shift_local_2` (kk) and
  `ood_neg_legit_bank_call_mixed_3` (styled).

**Costs, recorded, not criteria.**
- v2.1 misses 3 `test` scams that v1 caught.
- On ood styled copies it catches 37/44 against v1's 41/44.
- The validation stream false-latch rate has not been measured yet.

**Side finding.** The IIN in `ood_neg_legit_gov_service_mixed_2` is read with a filler in the middle
("980315... эээ... 350021"). The D66 scrubber does not mask it. The data is fabricated, but it is a
scrubber gap to fix separately (it is not part of v2.1).

**Next (A6, after export):** the browser gate on device embeddings, the demo scenes, and pytest + npm.

#### D63 v2.1 / A6 — verdict: the browser gate fails by 6e-6; v1 stays; v2.1 is a frozen candidate (2026-10-23)
The export was run locally (weights cd7b47bb…, config 0.82 / 0.72, floors 84/84,
`active_label_space: v2`, sw.js v13). The existing int8 browser gate (`browser_gate.test.mjs`,
assertions unchanged) then read, against v1 on the same gate:

| browser gate (200 cases, ONNX-Python vs WASM-browser) | decision flips | p95 Δrisk | tag-set Jaccard |
|---|---|---|---|
| limit | ≤ 6 | < 0.15 | ≥ 0.9 |
| shipped v1 | 4 | 0.091 | 0.952 |
| v2.1 @ 0.82 | 1 | 0.129 | **0.899994 (fail)** |

The decision and risk gates pass. The tag-set gate misses by 6e-6, most likely because v2.1 has 23
tactic heads against v1's 15. **The pre-registration said "the gate passes, do not loosen it; on a
fail, v1 stays."** The user chose to follow the rule (2026-10-23):
- The export is reverted. v1 stays shipped (weights f0659e7c…, 0.59 / 0.49).
- The gate is not loosened and not switched to device mode after the fact.
- `models/linear_v21_device` stays as a frozen candidate (sha256 above). `shift_local_2` has now
  been looked at once, so it cannot judge a later model.

**Found during the export, for any future v2-space ship:**
- `.env` / `.env.example` pin `QORGAN_RISK_THRESHOLD*`, which overrides `config.py`'s defaults.
  A threshold change must edit both env files, or `client_config` exports the old point.
- At enter 0.82 the meter needs a fourth risk-0.9 utterance to latch: the EMA reaches 78.75 after
  three.
- The flip to `active_label_space: v2` breaks 12 Python tests. Most are expected updates; the anchor
  dialogues and the `shift_local` briefs still carry v1 tactic ids, which is real migration work.
- The demo scenes were not run.

**Next: v2.2** (PLAN_2026-10 §4, track A, re-opened).
- About 250 new dialogues, aimed at v2.1's weak cells: code-switched (validation recall 12/30) and
  Kazakh (`shift_local_2` 21/40).
- Split between Claude and Gemini writers, so the new rows do not deepen the Claude style gap.
- The scam share stays matched to train's.
- New judge `shift_local_3`: the same local model and briefs, a new seed. It is judge-only and is
  never a training writer.
- The same cycle: pre-register → train → freeze → one look → A6.

### D63 v2.2 / A7 — PRE-REGISTRATION: targeted kk + code-switched data from two writers (2026-10-23)
_Committed before any v2.2 model was trained or any held-out split was scored. The procedure is
v2.1's (A4 pre-registration above) with the changes listed here._

**Why.** v2.1 beat v1 on its judge but failed the browser gate's tag-set Jaccard by 6e-6. Its weak
cells were code-switched calls (validation recall 12/30 at the rule point) and Kazakh
(`shift_local_2` 21/40). The user chose to keep v1 and generate targeted data (2026-10-23).

**New data** (both runs pass QC; local-only):
- `v22_claude` (Claude agents, spec only) and `v22_gemini` (Gemini 2.5 Pro via
  `qorgan.data.gen.gemini_writer`, the same chunk specs).
  - Each is 126 cells: kk 57 / mixed 69, scam 60 / legit 66, 53 look-alike negatives. The grids are
    `configs/gen_grid_v22_{claude,gemini}.yaml`, which is the base grid with a new seed and
    registers.
- QC: 126/126 accepted in each run, 0 eval overlaps, 0 near-duplicates.

  | QC | v22_claude | v22_gemini |
  |---|---|---|
  | ROUGE-L max | — | 0.340 |
  | fingerprint AUC vs Gemini train, char / e5 | 0.971 / 0.939 | 0.984 / 0.914 |
  | negatives with a cue | 6 | 5 |
  | role recall | 0.69 | 0.48 |

- **Writer difference, kept, not filtered.** Gemini's `mixed` rows are more Kazakh-dominant:
  58/69 have fewer than two Kazakh-free turns, against 34/69 for Claude. The legacy Gemini corpus
  shows the same tilt. No heuristic filter is applied.
- **Provenance.** `configs/gen_run_writers.yaml` marks `v22_gemini` as `gemini_grid`. Its rows carry
  v2 labels and are not masked. The v2.1 corpus rebuilds byte-identically under this code.

**Corpus.**
- Command: `build_corpus --gen-run pilot --gen-run p2 --gen-run v21_legit --gen-run v21_scam --gen-run
  v22_claude --gen-run v22_gemini --label-space v2 --processed-dir data/cache/processed_v22`.
  Content hash **c1adcb4268d0…f6c04**. The `v21_legit_0100` reject is applied.
- Train P(scam): claude 1714/3592 = 0.477; gemini (legacy) 896/1880 = 0.477;
  gemini_grid 102/212 = 0.481.
- Val: 465 rows (kk 150, mixed 118, ru 144, en 53).

**Unchanged from v2.1.**
- The trainer: v2's settings on device embeddings, into `models/linear_v22_device`.
- The operating-point rule: enter = the lowest 0.01 step at which v2.2's validation FPs ≤ v1's on
  the v2.2 validation set at 0.59; exit = enter − 0.10.
- The freeze step: sha256s and the validation table are committed before the look.
- A6 (a)–(c), and the browser gate with its assertions unchanged, including tag-set Jaccard ≥ 0.9.
  v2.1's 0.899994 is a known risk for v2.2 too.

**The judge is `shift_local_3`.** It is 130 kk/ru calls (seed 20261024), never looked at, and
disjoint from `shift_local` and `shift_local_2`. A6 (b) applies on it: v2.2's false alarms ≤ v1's
+ 1, and v2.2's recall > v1's.
- `shift_local_2` and `shift_local` are reported only; both were seen.
- The frozen v2.1 is scored alongside v2.2 for reference. It is not a criterion.

#### D63 v2.2 — FREEZE: trained, operating point by rule (2026-10-23)
_Committed before any held-out split is scored. Validation only, device embeddings, from
`scripts/taxonomy_v2_dryrun_eval.py` (`models/linear_v22_device/val_report.json`). Training ran as
pre-registered: no rerun, 23 heads, none dropped (the thinnest are task_advance_fee 105,
cash_courier_handoff 116 and refund_request 119), 1,880 legacy rows masked._

**Frozen model** (`models/linear_v22_device`, sha256):
- `risk_clf.joblib` aefc76c09fd7e2059336f888e7a812d70594cf78d1d8b592d647a317b72e5e8a
- `tactic_clf.joblib` 0e3c6cbfd1f38670c1fe54a7cff6414b05b6399942eb9f8ec8ce140478362ede
- `web/weights.json` e08bcba041795e79881b46d87b3c2a1c1f55068898e1d33f5fe37d9404394aa6

**Operating point (the rule).** v1 makes 5/251 validation FPs at 0.59, so **enter = 0.82,
exit = 0.72**, with floors 84 / 84 (the same point as v2.1).

| validation (device) | ROC-AUC / PR-AUC | FPR [95 % CP] | recall [95 % CP] |
|---|---|---|---|
| shipped v1 @ 0.59 | 0.867 / 0.867 | 5/251 = 0.020 [0.007, 0.046] | 96/214 = 0.449 [0.381, 0.518] |
| v2.2 @ 0.59 (reference) | 0.957 / 0.961 | 18/251 = 0.072 [0.043, 0.111] | 187/214 = 0.874 [0.822, 0.915] |
| **v2.2 @ 0.82 (rule)** | | **5/251 = 0.020 [0.007, 0.046]** | **158/214 = 0.738 [0.674, 0.796]** |

By generator and language at the rule point, as v1 → v2.2 (FPs; caught scams):

| rows | FPs | caught scams |
|---|---|---|
| Claude kk | 1/58 → 1/58 | 10/56 → 34/56 |
| Claude mixed | 0/31 → 0/31 | 4/35 → 17/35 |
| Claude ru | 3/58 → 4/58 | 15/42 → 33/42 |
| Claude en | 0/16 → 0/16 | 4/14 → 11/14 |
| gemini_grid | 0/11 → 0/11 | 5/9 → 6/9 |
| legacy Gemini | 1/77 → 0/77 | 58/58 → 57/58 |

The five validation FPs at 0.82 are p2_0864 (kk), p2_0955, p2_1105, p2_1248 and p2_1761 (ru). They
are generated rows, not held-out.

#### D63 v2.2 — the one look: (b) passes, (a) and (c) FAIL; v1 stays (2026-10-23)
_One run on device embeddings, with the frozen weights of 782460a. Each arm ran at its own point: v1
0.59 / 0.49, v2.2 and the frozen v2.1 0.82 / 0.72. Outputs are in
`models/linear_v22_device/a5/{v1,v22,v21}.{run,styled}.txt` and `*.neg_rows.json` (ids and risks,
no text). v2.1 is reference only._

**The judge: `shift_local_3`** (129 calls, first look).

| | FPR [95 % CP] | recall [95 % CP] | PR-AUC | kk FPR / recall | ru FPR / recall |
|---|---|---|---|---|---|
| v1 @ 0.59 | 6/65 = 0.092 [0.035, 0.190] | 32/64 = 0.500 [0.372, 0.628] | 0.878 | 0.125 / 0.450 | 0.040 / 0.583 |
| **v2.2 @ 0.82** | **2/65 = 0.031 [0.004, 0.107]** | **43/64 = 0.672 [0.543, 0.784]** | 0.955 | 0.025 / 0.600 | 0.040 / 0.792 |
| v2.1 @ 0.82 (ref.) | 2/65 = 0.031 [0.004, 0.107] | 42/64 = 0.656 [0.527, 0.771] | 0.950 | 0.025 / 0.575 | 0.040 / 0.792 |

**Other held-out FPR** (clean subsets and styled copies):

| split | v1 | v2.2 | v2.1 (ref.) |
|---|---|---|---|
| test (clean) | 0 | 0 | 0 |
| authored (clean) | 0 | 0 | 0 |
| ood (clean) | 0 | **1** | 0 |
| test, styled | 0/60 | 0/60 | 0/60 |
| authored, styled | 0/24 | 0/24 | 0/24 |
| ood, styled | 1/74 | **2/74** | 1/74 |

**A6 verdicts.**
- **(a) fails.** ood (clean) FPR rises 0 → 1 (`ood_neg_legit_gov_service_kk_0`, 0.830), and ood
  styled rises 1 → 2 (the same row, 0.829). v1 scores that row 0.057.
- **(b) passes.** On `shift_local_3`, false alarms fall 6 → 2 and recall rises 32 → 43.
- **(c) fails.** v2.2's new false alarms against v1:

  | row | v2.2 risk | in the ledger? | what it is |
  |---|---|---|---|
  | `aug_neg_family_money_request_ru_1` (test, inspected) | 0.878 | since 2026-10-12 | a relative in hospital asks for money by Kaspi: **a family call** |
  | `neg_general_conversation_mixed_21` (test) | 0.954 | yes | institutional |
  | `ood_neg_general_conversation_ru_1` (ood) | 0.840 | yes | Kaspi support confirms a transfer: institutional |
  | `ood_neg_legit_gov_service_mixed_2` (styled) | 0.838 | yes | institutional |
  | `ood_neg_legit_gov_service_kk_0` | 0.830 | no | not read |
  | `sl3_ru_legit_05` (judge) | 0.918 | no | not read |

  The family row decides (c). The two unread rows are left unread: the verdict does not depend on
  them, so the held-out sets stay unread.

**Decision (rule): v1 stays shipped.** v2.2 was not exported, and the browser gate was not run.
`models/linear_v22_device` is kept frozen. `shift_local_3` has now been looked at once.

**What the three models show (indicative, not a decision).**
- At the same 0.82 threshold, v2.2 adds held-out false alarms that v2.1 does not make:
  - the family request (v2.1: 0.807);
  - an eGov call (0.773);
  - a Kaspi confirmation (0.818).
- Both models make the same 2 errors on the new judge, and recall is 43 vs 42.
- So the extra 250 rows did not buy measurable recall on the independent judge. They moved some
  legitimate kk/ru look-alikes over the line.
- **v2.1 now has two independent judges in its favour.**
  - `shift_local_2` (its pre-registered look): 0/65 false alarms vs v1's 2, recall 38 vs 28.
  - `shift_local_3` (reference, not pre-registered for it): 2/65 vs 6/65, 42 vs 32.
  - Its only failed gate is the browser tag-set Jaccard, 0.899994 < 0.9.

### D72 — Scrubber v3: a dictated number may continue over a hesitation (2026-10-24)
**Context.** The D63 v2.1 one look found an IIN in `ood_neg_legit_gov_service_mixed_2` read as two
six-digit groups split by an ellipsis and a filler («…... эээ... …»). D66 joins number tokens only
over ≤ 3 characters of spaces, commas and dashes, so neither half reached a PII shape and both
were stored verbatim. People dictate exactly like this, and so does any transcriber that writes
pauses as "...".

**Decision.** A number run may continue over a *hesitation* (`scrub_numbers._is_hesitation`,
1:1 in `site/core/scrub-numbers.js`): at most two filler sounds (`э+`, `э+м+`, `ә+`, `аа+`, `мм+`,
`хм+`, `uh+`, `um+`, `hm+`; never a lone «а», which is a conjunction) between separator pieces of
≤ 5 characters drawn from the gap characters plus `.` and `…`. The gap must hold a filler or an
ellipsis (`..` / `…`). A full stop, a third filler or any real word still ends the run. The
masking thresholds are unchanged: a 6-digit code said with a hesitation stays evidence, and digit
groups are masked only at a PII shape (16 / 12 / 10–11). `SCRUBBER_VERSION` 2 → 3; `sw.js` →
`qorgan-shell-v13`.

**Evidence.**
- Tests first: 8 red, then green. `tests/data/test_scrub_spoken.py` gains 7 masked and 6
  untouched cases plus idempotency, and `scripts/export_scrub_fixtures.py` gains the same 12 cases
  for the JS port. pytest 1738 passed, 2 xfailed (the two designed "rebuild pending" guards);
  npm 159/159.
- Effect on stored text (counted, every changed span printed only as an abstract shape: digits →
  `9`, number words → `N`, other words → `w`): 3 rows change, all true PII. These are a phone in
  `train` dictated in groups over ellipses, and two IINs in `ood`, one of them the row above.
  `shift_local` shows 15 other differences, all `[NAME]` masks from D66. That split predates the
  name rule; it is local-only and never published, so it is left as built.

**Consequences.** The corpus on disk was scrubbed with v2, so `hf_upload.py` refuses to publish it
until the next corpus rebuild, which is the guard working as designed (D66 precedent). The shipped
v1 model is not retrained for this: 1 of 1,880 training rows changes, and only inside a masked
number.

### D73 — Taxonomy v3: five more tactics, and the v3 retrain. PRE-REGISTRATION (2026-10-24)
_Committed before any v3 model was trained and before `shift_local_4` was scored or read. The
procedure is v2.1's (D63 v2.1 A4 above) with the changes listed here._

**User decisions (2026-10-24).**
1. "Let's improve. Also, let's widen the range of tactics we cover." v1 stays shipped and v2.1
   stays frozen; neither is exported by this entry.
2. Five new tactics, all chosen by the owner: `minor_targeting`, `blackmail_extortion`,
   `survey_pretext`, `romance_trust`, `charity_appeal`. Evidence for the gap: keyword counts over
   the 6,327 reports of `fraudlens-ru-v1` (CC-BY; 97 % Russia-grounded, so mechanism only):
   children/teenagers 378 (phone 208), blackmail 191 (66), survey/poll 99 (46), romance 88 (41),
   charity 38 (16). Voice-cloned relatives, SIM-contract expiry and marketplace "safe deal" were
   already covered by schemes.
3. The owner reviews every new RU / KK / EN name, example and advice line before any v3 export.
   Until then they are marked DRAFT in `tactics.yaml`, `schemes.yaml`, `roles.yaml` and
   `advice_{ru,kk,en}.yaml`.
4. Gemini's prepaid credits are exhausted (402 RESOURCE_EXHAUSTED), so the planned Gemini run was
   replaced by a second Claude run ("Claude only, double it").

**Taxonomy v3** (6bb1965, 3e6f27c, 3a982de).
- Label space `v3` = the 23 v2 ids + the five. Each new tactic has a legitimate twin
  (`legit_child_call`, `legit_debt_notice`, `legit_survey`, `legit_partner_call`,
  `legit_charity_call`), a scheme (`minor_targeted`, `blackmail`, `survey_data_harvest`,
  `romance_scam`, `charity_fraud`) and advice. Two display-only roles: `pollster`, `charity_fund`.
  The hard-signal cue block stays 5 columns and no lexicon is edited.
- Masking: provenance records the label space of every row (legacy Gemini v1, every earlier grid
  run v2, the two new runs v3). A row is unlabelled for every tactic a later space introduced.
  On the v2.2 corpus this mask equals the D63 writer mask exactly; v2-era grids rebuild their
  manifests identically (`v21_legit` up to its 37 recorded constraints).
- A grid declares its label space (default v2) and may pin a class's callee (`class_personas`);
  the minors scheme and its twin always get a child.

**New data** (local-only; both QC pass).

| QC | v3_claude | v3_claude_b |
|---|---|---|
| cells / accepted | 450 / 450 | 450 / 450 |
| eval overlaps / near-dups / ROUGE-L flags | 0 / 0 / 0 | 0 / 0 / 0 |
| scheme-rule agreement | 216/216 | 216/216 |
| rows not a scrub fixed point (all `[NAME]`, scrubbed on fold) | 35 | 35 |
| fingerprint AUC vs Gemini train, char / e5 | 0.9999 / 0.985 | 0.9998 / 0.984 |
| role-slot recall | 0.53 | 0.51 |

- Grids `configs/gen_grid_v3_claude{,_b}.yaml`: kk 0.40 / ru 0.30 / mixed 0.30, scam share 0.48,
  the five v3 schemes at weight 8 (26–27 scams each per run), `mule_recruitment` 2.
- 7 `v3_claude` cells pair "college student" with age 10–13 (band and occupation were drawn
  apart); their writers wrote the callee as a young child. Kept. `v3_claude_b` pins school pupils.
- Largest cross-run token Jaccard 0.304 (no pair ≥ 0.5).
- **One writer family.** All v3 rows are Claude's. The five new heads see only v3 rows (every
  earlier row is masked), so their positives and negatives share one style; the risk head keeps
  the D63 balance (below).

**Corpus.**
- `build_corpus --gen-run pilot --gen-run p2 --gen-run v21_legit --gen-run v21_scam --gen-run
  v3_claude --gen-run v3_claude_b --label-space v3 --processed-dir data/cache/processed_v3`.
  Content hash **4fbc336362a7…f2e126b1**, scrubber v3.
- **The v2.2 runs are left out.** They added no recall on `shift_local_3` (43 vs v2.1's 42) and
  v2.2 failed A6 (a) and (c). That choice uses what `shift_local_3`, `ood` and A6 showed, which is
  why the judge below is a new split.
- Train 6,740 rows (kk 2,304 / ru 2,048 / mixed 1,774 / en 614); P(scam) Claude 2,342/4,860 =
  0.482, Gemini 896/1,880 = 0.477. Val 585 (322 legit, 263 scam).
- Calls carrying each v3 tactic (train incl. ASR copies / val): minor_targeting 96 / 6,
  blackmail_extortion 92 / 6, survey_pretext 88 / 8, romance_trust 86 / 9, charity_appeal 94 / 5.

**Trainer, operating point and freeze: v2.1's, unchanged.**
- `QORGAN_LABEL_SPACE=v3 QORGAN_EMBED_BACKEND=device python -m qorgan.classifier.linear_train
  --processed-dir data/cache/processed_v3 --out-dir models/linear_v3_device` on the device bridge.
  `V2_MIN_OOF_POSITIVES` = 8 applies to every head; `_RISK_C` = 4.0; D30 thresholds.
- enter = the lowest 0.01 step at which v3's validation FPs ≤ v1's on the v3 validation set at
  0.59; exit = enter − 0.10.
- The freeze commit records the sha256s, enter/exit and the validation table before the look.

**The judge is `shift_local_4`.** 160 kk/ru calls (kk 96 / ru 64) by the local Qwen3.5-Kazakh
model, plan seed 20261029, from the old briefs plus `configs/shift_local_briefs_v3.yaml` (15 scam
briefs, three per v3 tactic, and 10 legitimate twins), disjoint from every earlier batch, never
published. Nobody has read or scored it.

**A5 / A6, all required; v1 stays on any fail.**
- (a) FPR does not rise on the clean test / authored / ood subsets or on the styled copies.
- (b) On `shift_local_4`, v3's false alarms ≤ v1's + 1, and v3's recall > v1's.
- (c) Every new false alarm is an institutional look-alike, never a family or everyday call (the
  child, partner and charity-of-a-parents'-committee twins count as family/everyday). Only those
  rows are read, and they go into the inspection ledger.
- After export: the browser gate with its assertions unchanged (tag-set Jaccard ≥ 0.9; v2.1's
  0.899994 is a known risk, five more heads make it harder), the demo scenes, and the owner's
  sign-off on the draft texts.
- Reported, not criteria: v2.1 scored alongside; on `shift_local_4`, recall on the scams of the
  v3 briefs for v1 / v2.1 / v3; per-tactic tag recall of the five new heads on those calls.
  Nothing is re-tuned on what A5 shows.

#### D73 — FREEZE: trained, operating point by rule (2026-10-24)
_Committed before any held-out split is scored. Validation only, device embeddings, numbers from
`scripts/taxonomy_v2_dryrun_eval.py` (`models/linear_v3_device/val_report.json`). Training ran as
pre-registered: no rerun, 28 heads, none dropped, 5,254 train rows masked for at least one head.
Out-of-fold positives of the new heads: minor_targeting 102, blackmail_extortion 98,
survey_pretext 96, romance_trust 95, charity_appeal 99._

**Frozen model** (`models/linear_v3_device`, sha256):
- `risk_clf.joblib` 2892f7fff23d7174d419980c622b6b7cf9389145fd53f2b6b1be11793abb6506
- `tactic_clf.joblib` 4cf905db60a16b813237a8a60b6a92987da325fd85dd7ee729e55271ad7631f5
- `web/weights.json` b3bfef297bc91da616bb5725abb1b35172f9f34d185cbd8b6d3f9ed22f2a6666

**Operating point (the rule).** v1 makes 7/322 validation FPs at 0.59, so **enter = 0.83,
exit = 0.73**.

| validation (device) | ROC-AUC / PR-AUC | FPR [95 % CP] | recall [95 % CP] |
|---|---|---|---|
| shipped v1 @ 0.59 | 0.837 / 0.831 | 7/322 = 0.022 [0.009, 0.044] | 101/263 = 0.384 [0.325, 0.446] |
| v3 @ 0.59 (reference) | 0.955 / 0.957 | 25/322 = 0.078 [0.051, 0.113] | 233/263 = 0.886 [0.841, 0.922] |
| **v3 @ 0.83 (rule)** | | **7/322 = 0.022 [0.009, 0.044]** | **187/263 = 0.711 [0.652, 0.765]** |

At the rule point, as v1 → v3 (FPs; caught scams):

| rows | FPs | caught scams |
|---|---|---|
| Claude kk | 1/89 → 0/89 | 11/82 → 48/82 |
| Claude mixed | 0/54 → 1/54 | 10/56 → 29/56 |
| Claude ru | 5/86 → 4/86 | 18/53 → 43/53 |
| Claude en | 0/16 → 0/16 | 4/14 → 10/14 |
| legacy Gemini | 1/77 → 2/77 | 58/58 → 57/58 |

Validation per-tactic F1 of the new heads (labelled v3 rows only, support 5–9, so wide):
minor_targeting 0.714, blackmail_extortion 0.667, survey_pretext 0.824, romance_trust 0.762,
charity_appeal 0.462. mule_recruitment 0.718 (v2.1: 0.50).

#### D73 — the one look: (b) passes, (a) and (c) FAIL; v1 stays (2026-10-24)
_One run on device embeddings with the frozen weights of 96a1f64, each arm at its own point: v1
0.59 / 0.49, v3 0.83 / 0.73, the frozen v2.1 0.82 / 0.72 (reference). Outputs in
`models/linear_v3_device/a5/{v1,v3,v21}.{run,styled}.txt`, `*.neg_rows.json` and `*.v3_scams.json`
(ids, risks and tag ids, no text). The first attempt stopped in a report-only script
(`ScoreResult.tags`, not `tactic_tags`); the whole look was rerun unchanged and is deterministic._

**The judge: `shift_local_4`** (159 calls, first look).

| | FPR [95 % CP] | recall [95 % CP] | PR-AUC |
|---|---|---|---|
| v1 @ 0.59 | 0.063 [0.021, 0.142] | 0.412 [0.304, 0.528] | 0.825 |
| **v3 @ 0.83** | **0.063 [0.021, 0.142]** | **0.688 [0.574, 0.787]** | 0.914 |
| v2.1 @ 0.82 (ref.) | 0.063 [0.021, 0.142] | 0.562 [0.447, 0.673] | 0.901 |

(b) passes: the same false alarms as v1, and recall 0.688 > 0.412.

**(a) fails.**
- test (clean): FPR 0.000 → 0.017, one new false alarm (`neg_family_money_request_en_8`).
- ood styled copies: 1 → 2 false alarms (`ood_neg_legit_gov_service_kk_0`, the row that also failed v2.2).
- authored and ood (clean) are unchanged at 0.

**(c) fails.** v3's new false alarms against v1 include family and everyday calls, judged by
their class ids: `neg_family_money_request_en_8` (test, 0.847), `aug_neg_family_money_request_ru_1`
(test, inspected; the row that failed v2.2's (c), 0.899) and two general-conversation rows
(`neg_general_conversation_mixed_21`, inspected, 0.966; `aug_neg_general_conversation_mixed_1`,
styled, 0.845). The verdict does not depend on their text, so the three unread rows stay unread.

**Report only: the v3 briefs' 30 scams on `shift_local_4`.**

| caught | blackmail | charity | romance | survey | minors | total |
|---|---|---|---|---|---|---|
| v1 | 1/7 | 0/7 | 1/5 | 2/5 | 5/6 | 9/30 |
| v2.1 | 4/7 | 1/7 | 0/5 | 1/5 | 6/6 | 12/30 |
| v3 | 5/7 | 4/7 | 2/5 | 2/5 | 6/6 | 19/30 |

v3's new tags on those calls: minor_targeting 4/6, romance_trust 3/5, charity_appeal 3/7,
blackmail_extortion 1/7, survey_pretext 1/5.

**Verdict.** v1 stays shipped; v3 stays frozen in `models/linear_v3_device`, not exported. Nothing
is re-tuned on this look. `shift_local_4` has now been looked at once and cannot judge a retrain
built after it.

**What it shows.** v3 widens coverage (19/30 v3-brief scams vs v1's 9) and recall on the judge,
but for the second time in a row (v2.2, v3) adding scam data pushes family money requests and
everyday chat over the alert line at the rule point. The rule matches v1's validation FPs, and
validation is mostly writer-matched rows, so it does not protect these classes. v2.1 remains the
only retrain without such a false alarm.

### D74 — v3.1: family and everyday hard negatives for the v3 retrain. PRE-REGISTRATION (2026-10-24)
_Committed before any v3.1 model was trained and before `shift_local_5` was scored or read. The
procedure is D73's with the changes listed here._

**User decision (2026-10-24).** "Continue" after the D73 verdict, on the recommended option: keep
the v3 data and add hard negatives for family money requests and everyday chat, with a fresh
judge. v1 stays shipped; v2.1 and v3 stay frozen, not exported.

**Why these classes.** v2.2 and v3 both failed A6 (c) on family money requests and general
conversation (D73 one look). The class choice uses what A6 showed on test / ood — class ids only;
no row text was read beyond the two rows already in the inspection ledger. So (a) and (c) on
test / ood are no longer blind for these two classes; `shift_local_5` is.

**New data: run `v31_claude`** (local-only; QC pass), `configs/gen_grid_v31_claude.yaml`, seed
20261031, 800 cells, kk 320 / ru 240 / mixed 240, label space v3.
- 436 legitimate: family_money_request 166, general_conversation 165, legit_child_call 35,
  legit_partner_call 35, legit_charity_call 35. Every family and everyday cell carries one of 24
  new briefs (12 per class), written after the judge briefs and kept apart from them and from
  `v21_legit`'s; none was built from an evaluation row. 41 cells are look-alikes of
  `relative_in_trouble`.
- 364 scams, only the personal schemes those calls can be mistaken for: relative_in_trouble 101,
  romance_scam 51, minor_targeted 51, charity_fraud 51, fake_boss 34, marketplace_seller_phish 34,
  investment_broker 25, blackmail 17. Every other scheme has weight 0 in this run.
- Scam share 0.455 keeps the Claude train block at the Gemini rate (D63 phase 2):
  (1171 + 0.7 s) / (2430 + 0.7 · 800) = 0.4766 → s ≈ 357.
- 16 Claude writer agents, two 25-cell chunks each, the D73 writer brief unchanged; every chunk
  passed `gen check` on its first run.

| QC (`data/synthetic/gen/v31_claude/qc_report.json`) | v31_claude |
|---|---|
| cells / accepted | 800 / 800 |
| eval overlaps / near-dups / ROUGE-L flags | 0 / 0 / 0 |
| scheme-rule agreement | 364/364 |
| rows not a scrub fixed point (scrubbed on fold) | 36 |
| cue-free scam share | 0.907 |
| negatives containing a cue | 22 |
| fingerprint AUC vs Gemini train, char / e5 | 0.9999 / 0.995 |
| role-slot recall | 0.16 (most callers are private persons, no institutional role) |

- Writers' notes: some scam rows annotate extra tactics the dialogue shows (allowed by the spec);
  in 5 legitimate cells (`v31_claude_0442`, `_0539`, `_0570`, `_0634`, `_0635`) the writer changed
  the relationship a brief names (e.g. "father") to fit the drawn callee's age and kept the
  situation. Two writers produced their JSONL through a scratch script from hand-written text.
  Kept.

**Corpus.** `build_corpus --gen-run pilot --gen-run p2 --gen-run v21_legit --gen-run v21_scam
--gen-run v3_claude --gen-run v3_claude_b --gen-run v31_claude --label-space v3 --processed-dir
data/cache/processed_v31`. Content hash **8a10f6fea6e8…ae7** (`8a10f6fea6e801998237cbf720f11de185cd83fb50881d14fed825cc24afdae7`),
scrubber v3.
- Train 8,100 (3,868 scam / 4,232 legit; kk 2,846 / ru 2,456 / mixed 2,184 / en 614). P(scam)
  Claude 2,972/6,220 = 0.4778, Gemini 896/1,880 = 0.4766.
- Val 705 (312 scam / 393 legit). The new run adds 120 val rows: 26 family money requests,
  26 general conversation, 19 child / partner / charity twins, 49 scams. Test and authored are
  unchanged (143 / 42).

**Trainer, operating point and freeze: D73's, unchanged.**
- `QORGAN_LABEL_SPACE=v3 QORGAN_EMBED_BACKEND=device python -m qorgan.classifier.linear_train
  --processed-dir data/cache/processed_v31 --out-dir models/linear_v31_device` on the device bridge.
- enter = the lowest 0.01 step at which v3.1's validation FPs ≤ v1's on the v3.1 validation set at
  0.59; exit = enter − 0.10.
- The freeze commit records the sha256s, enter/exit and the validation table before the look.

**The judge is `shift_local_5`.** 160 kk/ru calls (kk 96 / ru 64) by the local Qwen3.5-Kazakh
model, plan seed 20261101, prefix sl5, from the `shift_local_4` briefs plus
`configs/shift_local_briefs_v31.yaml` (12 legitimate family and everyday briefs), disjoint from
every earlier batch, never published (`NEVER_PUBLISHED`, gitignored). Its build started before
this entry; nobody has read or scored it.

**A5 / A6, all required; v1 stays on any fail.** D73's, on the new judge:
- (a) FPR does not rise on the clean test / authored / ood subsets or on the styled copies.
- (b) On `shift_local_5`, v3.1's false alarms ≤ v1's + 1, and v3.1's recall > v1's.
- (c) Every new false alarm is an institutional look-alike, never a family or everyday call (the
  child, partner and charity-of-a-parents'-committee twins and every `shift_local_briefs_v31`
  brief count as family/everyday). Only those rows are read, and they go into the inspection ledger.
- After export: the browser gate with its assertions unchanged, the demo scenes, and the owner's
  sign-off on the D73 draft texts.
- Reported, not criteria: the frozen v3 (0.83 / 0.73) and v2.1 (0.82 / 0.72) scored alongside;
  on `shift_local_5`, recall on the v3-brief scams for v1 / v2.1 / v3 / v3.1 and false alarms on
  the `shift_local_briefs_v31` calls. Nothing is re-tuned on what A5 shows.

#### D74 — FREEZE: trained, operating point by rule (2026-10-24)
_Committed before any held-out split is scored and before `shift_local_5` is finished. Validation
only, device embeddings, numbers from `scripts/taxonomy_v2_dryrun_eval.py`
(`models/linear_v31_device/val_report.json`, `val_eval.log`). Training ran as pre-registered: no
rerun, 28 heads, none dropped. Out-of-fold positives of the v3 heads: minor_targeting 197,
blackmail_extortion 128, survey_pretext 96, romance_trust 192, charity_appeal 193;
mule_recruitment 189._

**Frozen model** (`models/linear_v31_device`, sha256):
- `risk_clf.joblib` a56f371b12001b6a242303806187a13ed5a76c9d4dd29849b1df5b0e083b8644
- `tactic_clf.joblib` b3a9e32a54b9cfb0960b2d3d4d6022ee8de2937f659f0d26e6e6d3eafa8fed90
- `web/weights.json` 5d971fe427d3513ffc55988cec34c21adc40e6e61165a5994835af68981c94d5

**Operating point (the rule).** v1 makes 10/393 validation FPs at 0.59, so **enter = 0.84,
exit = 0.74**.

| validation (device) | ROC-AUC / PR-AUC | FPR [95 % CP] | recall [95 % CP] |
|---|---|---|---|
| shipped v1 @ 0.59 | 0.828 / 0.808 | 10/393 = 0.025 [0.012, 0.046] | 106/312 = 0.340 [0.287, 0.395] |
| v3.1 @ 0.59 (reference) | 0.957 / 0.957 | 33/393 = 0.084 [0.059, 0.116] | 276/312 = 0.885 [0.844, 0.918] |
| **v3.1 @ 0.84 (rule)** | | **10/393 = 0.025 [0.012, 0.046]** | **225/312 = 0.721 [0.668, 0.770]** |

At the rule point, v1 → v3.1 (FPs; caught scams):

| rows | FPs | caught scams |
|---|---|---|
| Claude kk | 2/118 → 2/118 | 14/102 → 64/102 |
| Claude mixed | 0/73 → 2/73 | 11/72 → 43/72 |
| Claude ru | 7/109 → 4/109 | 19/66 → 52/66 |
| Claude en | 0/16 → 0/16 | 4/14 → 9/14 |
| legacy Gemini | 1/77 → 2/77 | 58/58 → 57/58 |

Validation per-tactic F1 of the v3 heads (labelled v3 rows only, support 8–14):
minor_targeting 0.560, blackmail_extortion 0.818, survey_pretext 0.889, romance_trust 0.897,
charity_appeal 0.710; mule_recruitment 0.714.

#### D74 — the one look: (b) passes, (a) and (c) FAIL again; v1 stays (2026-10-24)
_One run on device embeddings with the frozen weights of dc16b7e, each arm at its own point: v1
0.59 / 0.49, v3.1 0.84 / 0.74, and for reference the frozen v3 0.83 / 0.73 and v2.1 0.82 / 0.72.
Outputs are in `models/linear_v31_device/a5/{v1,v31,v3,v21}.{run,styled}.txt`, `*.neg_rows.json`
and `*.v3_scams.json` (ids, brief ids, risks and tag ids, no text). No row text was read for this
look, so the inspection ledger is unchanged; the verdict rests on class ids and brief ids alone._

**The judge: `shift_local_5`** (156 calls, 76 scam / 80 legit, first look).

| | FPR [95 % CP] | recall [95 % CP] | PR-AUC |
|---|---|---|---|
| v1 @ 0.59 | 0.062 [0.021, 0.140] | 0.447 [0.333, 0.566] | 0.846 |
| **v3.1 @ 0.84** | **0.050 [0.014, 0.123]** | **0.618 [0.500, 0.728]** | 0.927 |
| v3 @ 0.83 (ref.) | 0.050 [0.014, 0.123] | 0.592 [0.473, 0.704] | 0.916 |
| v2.1 @ 0.82 (ref.) | 0.050 [0.014, 0.123] | 0.539 [0.421, 0.655] | 0.911 |

(b) passes: 4 false alarms against v1's 5, and recall 0.618 > 0.447.

**(a) fails, and is worse than v3.**

| v1 → v3.1 FPR | clean | styled |
|---|---|---|
| test | 0.000 → 0.050 (clean subset 0.000 → 0.017) | 0.000 → 0.033 |
| authored_heldout | 0.000 → 0.000 | 0.000 → 0.000 |
| ood | 0.000 → 0.027 (both rows already inspected) | 0.014 → 0.054 |

**(c) fails.** v3.1's new false alarms against v1, by class or brief id:
- **Family and everyday:**
  - test: `neg_family_money_request_en_8` (0.879)
  - test, inspected: `aug_neg_family_money_request_ru_1` (0.889) and `neg_general_conversation_mixed_21` (0.969)
  - ood, inspected: `ood_neg_family_money_request_mixed_0` (0.848) and `ood_neg_general_conversation_ru_1` (0.957)
  - ood, styled: `ood_neg_general_conversation_mixed_0` (0.840)
  - test, styled: `aug_neg_general_conversation_mixed_1` (0.943)
  - `shift_local_5`: `sl5_kk_legit_46`, from brief `l31_chat_neighbour_police` (0.963 clean), and `sl5_kk_legit_44`, from brief `l31_family_kaspi_red` (0.951 styled)
- **Institutional:** `sl5_ru_legit_25` (`l_parcel_delivery`), plus the styled copies of `aug_neg_legit_gov_service_mixed_3`, `ood_neg_legit_gov_service_{kk_0,mixed_2}` and `ood_neg_delivery_notification_mixed_3`.

**Report only.**
- Of the 12 `shift_local_briefs_v31` brief calls (sl5), the arms alert on (clean or styled):
  - v1: `l31_chat_scam_story`, `l31_chat_phone_help`
  - v3.1, v3 and v2.1: the same two plus `l31_chat_neighbour_police` and `l31_family_kaspi_red`
- The v3 briefs' 26 scams on `shift_local_5`:

| caught | blackmail | charity | romance | survey | minors | total |
|---|---|---|---|---|---|---|
| v1 | 2/6 | 1/6 | 0/4 | 2/5 | 4/5 | 9/26 |
| v2.1 | 3/6 | 1/6 | 1/4 | 2/5 | 5/5 | 12/26 |
| v3 | 3/6 | 2/6 | 2/4 | 2/5 | 5/5 | 14/26 |
| v3.1 | 4/6 | 3/6 | 2/4 | 2/5 | 5/5 | 16/26 |

**Verdict.** v1 stays shipped. v3.1 stays frozen in `models/linear_v31_device` and is not exported.
Nothing is re-tuned on this look, and `shift_local_5` cannot judge any retrain built after it.

**What it shows.**
- The family and everyday hard negatives did not work. 800 cells, 331 of them family or everyday,
  left v3.1 with more family/everyday false alarms on test/ood than v3, not fewer. Recall on the
  judge rose again, by 0.03.
- The val-matched rule point is not the cause on its own: validation FPs were equal to v1's
  (10/393), yet held-out family/everyday rows alert. More writer-matched hard negatives make
  validation easier without moving the held-out family boundary.
- Every retrain (v2.1, v3, v3.1) alerts on the same two `l31_*` everyday briefs as each other.
- The next step needs a different lever, not more data of the same kind: for example, a gate
  on caller role or an institutional cue for scores between the old and new points, or keeping
  v1's risk head and adding only the new tactic heads. That needs its own ADR and a new judge.

### D75 — a v1 + v3.1 risk ensemble. PRE-REGISTRATION (2026-10-24)
_Committed before the ensemble was scored on any split (validation included) and before
`shift_local_6` was generated._

**User decision (2026-10-24).** After the D74 verdict, the user chose the recommended option
"Ensemble v1 + v3.1" over "v1 risk + new tactic tags" and "park track A". v1 stays shipped until
the ensemble passes; v2.1, v3 and v3.1 stay frozen, not exported.

**Why.** Three retrains (v2.2, v3, v3.1) raised recall on the independent judges and each time
made family money requests and everyday chat alert. v1 does not alert on those calls (D73, D74),
and v3.1 catches the most scams. The two heads err on different calls, so their mean may keep
most of v3.1's recall without its family and everyday false alarms.

**The model: no new training.**
- Risk = 0.5 · v1 + 0.5 · v3.1, both calibrated probabilities. The weight is fixed here, not
  tuned. Both exported heads are "the mean over 3 calibrated CV members" and read the same
  features (e5-base device embedding, cue matcher v2, hybrid-v1, identical embedded lexicons),
  so the mix is the 6 members in one head.
  - Built by `python -m qorgan.classifier.web_ensemble --base site/models/weights.json
    --partner models/linear_v31_device/web/weights.json --out models/linear_v31e/web/weights.json`.
    The module refuses heads whose inputs or member counts differ (`tests/classifier/test_web_ensemble.py`).
  - `WebScorer` and `site/core/head.js` score that file unchanged, so the browser needs no code
    change and parity holds by construction.
  - Inputs, sha256: v1 `site/models/weights.json` f0659e7c1872e88c4104c1a7956ed540cf9e3fd126e82ba20809f73908a40bc5;
    v3.1 `models/linear_v31_device/web/weights.json` 5d971fe427d3513ffc55988cec34c21adc40e6e61165a5994835af68981c94d5 (D74 freeze).
  - Output: `models/linear_v31e/web/weights.json`, sha256 af82696cafbdbed9b89d976e9aaadb851f0f9e965604160cd1654e4672a739b9.
- The tactic heads, their thresholds and the label space (v3, 28 heads) are v3.1's, unchanged.
- It is served from the JSON (`QORGAN_LINEAR_WEIGHTS=web`, `QORGAN_WEB_WEIGHTS_PATH=` that file).

**Operating point and freeze: D73's rule.**
- enter = the lowest 0.01 step at which the ensemble's validation FPs ≤ v1's, both scored on
  the v3.1 validation set (`data/cache/processed_v31`, 705 rows); exit = enter − 0.10.
- Scored by `scripts/taxonomy_v2_dryrun_eval.py --model-dir models/linear_v31e/web/weights.json`;
  the script now accepts a weights JSON.
- The freeze commit records enter/exit and the validation table before the look.

**What is NOT blind.**
- The design was chosen after D74's look. Its per-row risks for v1 and v3.1 on test, authored,
  ood and `shift_local_5` are on disk (`models/linear_v31_device/a5/*.neg_rows.json`), and the
  ensemble's risk on those rows is their mean. So (a) and (c) on test / authored / ood are safety
  checks here, not evidence. Nothing is computed from those files before the freeze.
- `shift_local_6` is the only blind judge.

**The judge is `shift_local_6`.** 160 kk/ru calls (kk 96 / ru 64) by the local Qwen3.5-Kazakh
model, plan seed 20261103, prefix sl6. The fifth batch's briefs (21 slots from
`shift_local_briefs_v31.yaml`, 80 scam slots). Disjoint from every earlier batch, never published.

**A5 / A6, all required; v1 stays on any fail.** D74's, on the new judge:
- (a) FPR does not rise on the clean test / authored / ood subsets or on the styled copies.
- (b) On `shift_local_6`: the ensemble's false alarms ≤ v1's + 1, and its recall > v1's.
- (c) Every new false alarm is an institutional look-alike, never a family or everyday call (the
  twins and every `shift_local_briefs_v31` brief count as family/everyday). Only those rows are
  read, and they go into the inspection ledger.
- After export: the browser gate with its assertions unchanged, the demo scenes, and the owner's
  sign-off on the D73 draft texts.
- Reported, not criteria: v3.1 alone (0.84 / 0.74) and v1 alone on `shift_local_6`, recall on its
  v3-brief scams, and the false alarms on its `shift_local_briefs_v31` calls. Nothing is re-tuned
  on what A5 shows.

#### D75 — FREEZE: operating point by rule (2026-10-24)
_Validation only. No held-out split has been scored with the ensemble; `shift_local_6` is
being generated._

`QORGAN_LABEL_SPACE=v3 QORGAN_EMBED_BACKEND=device python scripts/taxonomy_v2_dryrun_eval.py
--processed-dir data/cache/processed_v31 --model-dir models/linear_v31e/web/weights.json
--baseline site/models/weights.json --out models/linear_v31e/val_report.json`, on cached device
embeddings.

**Operating point (the rule).** v1 makes 10/393 validation FPs at 0.59, so **enter = 0.66,
exit = 0.56**. The fused file was then rebuilt with that point written into its `thresholds`
(`web_ensemble --enter 0.66 --exit 0.56`). The risk head is unchanged by this.

**Frozen model** `models/linear_v31e/web/weights.json`, sha256
be546b6bda871a2b9d34eaaaa8a1917046d2d0750ae5a560203e353136c6a90c. It has 6 risk members (v1's 3
and v3.1's 3, sources recorded in `risk_ensemble`), v3.1's 28 tactic heads and thresholds 0.66 / 0.66 / 0.56.

| validation (device) | ROC-AUC / PR-AUC | FPR [95 % CP] | recall [95 % CP] |
|---|---|---|---|
| shipped v1 @ 0.59 | 0.828 / 0.808 | 10/393 = 0.025 [0.012, 0.046] | 106/312 = 0.340 [0.287, 0.395] |
| ensemble @ 0.59 (reference) | 0.935 / 0.918 | 16/393 = 0.041 [0.023, 0.065] | 148/312 = 0.474 [0.418, 0.531] |
| **ensemble @ 0.66 (rule)** | | **10/393 = 0.025 [0.012, 0.046]** | **127/312 = 0.407 [0.352, 0.464]** |
| v3.1 alone @ 0.84 (D74) | 0.957 / 0.957 | 10/393 = 0.025 | 225/312 = 0.721 |

At the rule point, v1 → ensemble (FPs; caught scams):

| rows | FPs | caught scams |
|---|---|---|
| Claude kk | 2/118 → 2/118 | 14/102 → 21/102 |
| Claude mixed | 0/73 → 2/73 | 11/72 → 12/72 |
| Claude ru | 7/109 → 6/109 | 19/66 → 29/66 |
| Claude en | 0/16 → 0/16 | 4/14 → 7/14 |
| legacy Gemini | 1/77 → 0/77 | 58/58 → 58/58 |

On validation the ensemble keeps much less of v3.1's recall than hoped: +21 scams over v1, against
v3.1's +119. The tactic F1s are v3.1's (same head): minor_targeting 0.560, blackmail_extortion 0.818,
survey_pretext 0.889, romance_trust 0.897, charity_appeal 0.710, mule_recruitment 0.714.

#### D75 — the one look: (a) and (b) pass, (c) FAILS on two everyday calls; v1 stays (2026-10-24)
_One run on device embeddings (bridge with 2 pages) with the frozen files of 15b510c: v1 0.59 / 0.49,
the ensemble 0.66 / 0.56, and v3.1 alone at 0.84 / 0.74 for reference. Outputs are in
`models/linear_v31e/a5/{v1,v31e,v31}.{run,styled}.txt`, `*.neg_rows.json` and `*.v3_scams.json`
(ids, brief ids, risks and tag ids, no text). No row text was read, so the inspection ledger is unchanged._

**The judge: `shift_local_6`** (157 calls, 78 scam / 79 legit, first look).

| | FPR [95 % CP] | recall [95 % CP] | PR-AUC |
|---|---|---|---|
| v1 @ 0.59 | 0.025 [0.003, 0.088] | 0.436 [0.324, 0.553] | 0.833 |
| **ensemble @ 0.66** | **0.013 [0.000, 0.069]** | **0.474 [0.360, 0.591]** | 0.910 |
| v3.1 alone @ 0.84 (ref.) | 0.013 [0.000, 0.069] | 0.628 [0.511, 0.735] | 0.943 |

(b) passes: 1 false alarm against v1's 2, and recall 0.474 > 0.436 (3 more scams caught).

**(a) passes** for the first time since v2.1.

| v1 → ensemble FPR | clean subset | styled copies |
|---|---|---|
| test | 0.000 → 0.000 | 0.000 → 0.000 |
| authored_heldout | 0.000 → 0.000 | 0.000 → 0.000 |
| ood | 0.000 → 0.000 | 0.014 → 0.000 |

**(c) fails.** The ensemble's two new false alarms against v1 are both everyday calls:
- `neg_general_conversation_mixed_21`: an inspected test row, so not in the clean subset; 0.754 clean. v3 and v3.1 also alerted on it.
- `sl6_kk_legit_31`, from brief `l31_chat_phone_help`: its styled copy only (0.687; clean 0.585).

Both arms alert on `sl6_ru_legit_27` (`l31_chat_scam_story`, two friends retelling a scam call),
so it is not a new alarm. v1's own 0.59 alert on `sl6_kk_legit_08` (`l_bank_fraud_alert`) clears.

**Report only.**
- The v3 briefs' 31 scams on `shift_local_6`: v1 9/31, ensemble 11/31, v3.1 alone 19/31.
- v3.1 alone has 12 new false alarms against v1 on test / ood / judge, the same family/everyday
  pattern as D74.

**Verdict.** v1 stays shipped. The ensemble is frozen in `models/linear_v31e`, not exported.
Nothing is re-tuned on this look, and `shift_local_6` cannot judge anything built after it.

**What it shows.**
- Averaging v1 into v3.1 removes almost all of v3.1's family and everyday false alarms, from
  12 new ones to 2, and for the first time a retrain passes (a). But it also gives back most of
  v3.1's recall gain: +3 scams on the judge, against v3.1's +15.
- The fixed 50/50 weight sits near the point where both effects cancel. A different weight would
  have to be chosen on validation alone, and validation did not predict either D74 or this
  result, so this look gives no basis for one.
- The remaining everyday alarms are calls about scams or bank safety ("my neighbour was nearly
  fooled", "never read SMS codes to anyone"). Every arm, v1 included, scores them high. They talk
  about the scam vocabulary the risk head learns, which a different lever (a speaker-role or
  narrative feature) would have to separate. That needs its own ADR.
