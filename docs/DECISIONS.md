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
