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
backend has no current numbers — its key is not on this machine), and training data from
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
