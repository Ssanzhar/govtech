# Qorğan — technical reference (English)

_Moved from the README on 2026-11-10. The Russian technical documentation is
[`docs/TECHNICAL.md`](TECHNICAL.md); the numbers live in [`docs/eval_report.md`](eval_report.md)._

## What it does — three pages, one pipeline

The product is the static PWA under `site/`. (The Streamlit app in `app/` is a local dev
harness only — ADR D4 — it scores in the server process and its mic mode uploads audio. Do
not read it as the product.)

| Page | Persona | What happens |
|---|---|---|
| **Landing** (`index.html`) | citizen | What the tool does and does not do, in KK / RU / EN, with the AI disclosure and the download-size notice before the first model fetch. |
| **Live call** (`live.html`) | citizen | The call is analysed **turn by turn, in the browser**: utterances (replayed script, or the microphone: Vosk KK+RU show live words and GigaAM re-transcribes each phrase) → rolling window → int8 `multilingual-e5-base` in a web worker → **0–100 suspicion meter** (hysteresis, hard-signal floors) → verbatim trigger phrases, tactic tags, a templated reason and tactic-specific advice in the chosen language → post-call summary → **reviewed, editable, consent-gated report** with a redaction preview. Nothing leaves the device until the citizen sends it. |
| **Analyst console** (`admin.html`) | gov analyst | Fed **only** by consented reports. KPI row, priority queue of scam organizations named by dominant tactics, novel-scheme flags, drill-down. Per-person keys, analyst vs investigator roles, a stated purpose code and an hourly budget to open a full transcript, and an HMAC-chained audit log (ADR D46). |

## Quick start

```bash
# 1. Environment (Python 3.11+)
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"         # runtime + tests; ".[all]" adds the cloud tier, Streamlit harness, research paths
cp .env.example .env             # set the number-HMAC, analyst and audit-chain secrets (see the file); GEMINI_API_KEY only for the cloud tier / data-gen

# 2. Models + demo data in one go (idempotent; ~690 MB download on first run):
#    the corpus splits from Hugging Face, the int8 ONNX embedder the browser ships (self-hosted
#    under site/models/), the Vosk and GigaAM speech models, and the Level-2 demo seeds. The heads
#    served are the committed site/models/weights.json.
python scripts/deploy_bootstrap.py
#    ...and, once the bootstrap has written the dialogue pool, retrain if you need to (seconds, CPU):
python -m qorgan.data.build_corpus && python -m qorgan.classifier.linear_train

# 3. Run the site (landing + live call + analyst dashboard) -- analysis runs ON THE DEVICE
python -m qorgan.api             # http://localhost:8000
#    or the Streamlit dev harness (needs `pip install -e ".[harness]"`; not the product):
streamlit run app/streamlit_app.py
```

**On-device by design.** The pages under `site/` load the same `multilingual-e5-base`
int8 ONNX graph and the exported head weights (`site/models/`) and score transcripts in
the browser (`site/core/`, a 1:1 port of the Python classifier — `npm test` proves parity
on golden fixtures). The server embeds with the *same* int8 graph
(`QORGAN_EMBED_BACKEND=onnx`) on its native ONNX Runtime, a documented proxy of the browser (cosine
about 0.98, ADR D33), so a verdict can differ near the threshold. No route accepts audio; call
content leaves the device only on an explicit report. After the first visit the replay works with no
network at all (ADR D86).

**No model, no key?** The app still runs — it degrades to a deterministic `mock` backend
so the demo scripts work out of the box.

**Harness microphone (optional):** `pip install -e ".[harness,live]"` (vosk, streamlit-webrtc,
sounddevice). First use downloads two small Vosk models (~100 MB) to `~/.cache/vosk`.
Put the call on speakerphone near the device. Without the extra, the Live tab's replay
mode still works and the mic modes show an install hint.

## The demo storyline (3 replay scenes + the analyst flow)
All three ship in `site/core/scenarios.json` and were verified in real Chrome.
1. **Russian scam call** (`live_scam_bank_ru`) — the meter climbs to Critical, evidence and
   advice appear mid-call, the post-call summary offers a report.
2. **Kazakh scam call** (`live_scam_bank_kk`) — the same, in Kazakh, with `secrecy`,
   `otp_request` and `safe_account` firing on verbatim cue hits.
3. **Hard negative** (`live_hard_negative_bank_ru`) — a *real* bank call does **not**
   trigger. False-positive discipline is the product's core metric.

Then the **analyst flow**: submit the report from scene 1 (use a number from a seeded org,
e.g. `+7 700 101 20 30`), open `admin.html` with an analyst key and click **ingest reports**
(«принять сообщения в анализ») — it lands inside that organization. The queue also holds a new
scheme, flagged NEW: a small operation the seeds always include (ADR D86).

## Evaluate

```bash
# FPR-first tables, per language. The shift_local* batches are the independent judges (local
# only, never published); the other splits share a generator with training and read high.
# The published numbers use the browser's own embeddings and the committed weights:
#   npm run device:serve -- --pages 4 &    (the embedding bridge, ADR D33)
QORGAN_CLASSIFIER_BACKEND=linear QORGAN_EMBED_BACKEND=device QORGAN_LINEAR_WEIGHTS=web python -m qorgan.eval.run \
    --split test --split authored_heldout --split ood --split shift --split shift_local --by-language
QORGAN_CLASSIFIER_BACKEND=linear QORGAN_EMBED_BACKEND=device QORGAN_LINEAR_WEIGHTS=web python scripts/pooled_judges.py   # the headline: all six judges, kk+ru

# Streaming eval: false-latch rate (live FPR analog), time-to-alert
QORGAN_CLASSIFIER_BACKEND=linear python -m qorgan.eval.stream \
    --split test --split authored_heldout --backend linear

# Live-meter parameter sweep on cached per-turn traces (seconds, after a ~2 min trace build)
QORGAN_CLASSIFIER_BACKEND=linear python -m qorgan.eval.meter_sweep --min-turns 1,2,3 --damping 1,2,3

# Level-2 cluster quality under number-rotation stress, with stability intervals
python -m qorgan.eval.cluster --resamples 50

# Adversarial paraphrases: cue-free (A9) and legit-sounding (A9b), paired recall vs the sources
QORGAN_CLASSIFIER_BACKEND=linear python -m qorgan.eval.adversarial
QORGAN_CLASSIFIER_BACKEND=linear python -m qorgan.eval.adversarial --split adversarial_legit

# The recogniser's register: every eval dialogue clean vs ASR-styled (lowercase, no
# punctuation, numerals as words), paired; --drop-latin is the worst case for «SMS»/«CVV»
QORGAN_CLASSIFIER_BACKEND=linear python -m qorgan.eval.asr_realism [--drop-latin]

pytest -q        # offline; tests that need the bootstrapped corpus or local-only judges skip on a fresh clone
npm test         # JS core parity with Python + the DEVICE gate (browser embeddings, seconds)
npm run gate:browser   # re-capture the browser's embeddings of the gate set (headless Chromium;
                       # `npx playwright install chromium` once) after a model/runtime change

# Retrain / evaluate on the DEVICE's embeddings (ADR D33): start the bridge, then any command
# above with QORGAN_EMBED_BACKEND=device (vectors are cached; the first pass takes minutes)
npm run device:serve -- --pages 4 &
QORGAN_EMBED_BACKEND=device python -m qorgan.classifier.linear_train
```

**Numbers live in one place.** The current, measured state — FPR-first, with Clopper–Pearson
intervals, per split and per language — is the table in the README's «Качество модели» section; the full methodology and
every ablation is [`docs/eval_report.md`](eval_report.md). They are deliberately not
repeated here: this section used to carry its own copy, it drifted out of date, and a reviewer
caught the contradiction.

Three points that belong with the numbers rather than in the table:

- **They describe the device.** Heads are trained and evaluated on the **browser's own
  embeddings** (ADR D33), so the figures describe what the citizen's device decides, not a
  server approximation. The server's native runtime is a cosine-0.98 proxy and disagrees on
  4/200 borderline calls at the shipped weights — reported, not hidden.
- **The independent judges are the number to lead with.** The six `shift_local*` batches (828
  kk/ru calls) were written offline by a local model from another family that never saw the
  corpus, the prompts or the lexicons, all after v1's weights were frozen:
  `python scripts/pooled_judges.py` gives FPR 22/417 = 0.053 [0.033, 0.079] and recall
  187/411 = 0.455 [0.406, 0.505] (Kazakh 0.383, Russian 0.564; ADR D76). Every other split shares
  its generator with training data. `shift` was written by Claude, which now writes training data,
  so it is validation-grade (D65). Until real calls exist (A3), the recall claim is "another
  generator's scams".
- **The cloud tier is the accuracy tier, and it is off by default.** On the 66 `shift` calls the
  `llm` backend (Gemini 2.5 Pro) scores 33/33 recall at 0/33 FPR (ADR D38). It sends text
  abroad, so it requires `QORGAN_CLOUD_TIER=on` plus the citizen's per-request consent, is
  never cached and is refused outright on analyst routes (ADR D49). The gap between the device
  model and that tier is the honest cost of running on-device.

## Microphone mode (on-device speech recognition)

On desktop browsers the live page can listen to a call directly: Kazakh and Russian Vosk
models run in the browser (Vosklet/WASM, one instance each) and each word gets a live kk/ru chip;
when a phrase ends, GigaAM re-transcribes it, and that text feeds the same on-device classifier
and meter as replay (D64, D70, on the demo since D79) — **no audio or text leaves the device** (ADR
D26). Phones and Android WebView get the light mode: GigaAM only, no live words (D67). It has never
run on a real phone. English left the voice pipeline in ADR D70 (calls are kk/ru only). Requirements the server already meets:
`/live.html`, `/core/*` and `/vendor/*` are served cross-origin isolated (COOP/COEP) because the
recogniser needs SharedArrayBuffer; `python scripts/deploy_bootstrap.py` installs the
hash-pinned Vosklet and transformers.js/onnxruntime-web runtimes under `site/vendor/` (ADR D61)
and packages the model tarballs (downloaded once by the browser) — nothing is
loaded live from a CDN, which Safari would block under COEP.

## Real calls (when they arrive)

`docs/DATA_INTAKE.md` is the policy and protocol: encrypted/on-prem delivery, a
`batch.yaml` + `calls.csv` batch format, `python scripts/ingest_partner_calls.py <batch>`
(scrub → hashed number linkage → first-come allocation into a hash-locked
`real_heldout_v2` of 60 legit / 40 scam, the rest to `real_train`). The locked set is
scored, never read, and only at release points. `data/real/` never enters git.

## Reports API (consented ingress) & privacy

`POST /api/reports` is the only way call content enters the analyst layer, and only on an
explicit user action. The server stores the transcript PII-scrubbed, the caller number as
a salted HMAC digest + prefix (`+7 700 ***`), returns exactly what it kept plus a receipt,
and `DELETE /api/reports/{receipt}` forgets it everywhere (`python -m qorgan.reports.purge`
applies the retention window). Set `QORGAN_NUMBER_HMAC_KEY` (see `.env.example`); without
it the server refuses reports that carry a number. No route accepts audio; these
invariants are enforced by `tests/test_architecture.py` (ADRs D12–D14).

## Analyst dashboard exposure

`admin.html` asks for an **analyst key** (`QORGAN_ANALYST_KEYS="id:secret:role,..."`; unset =
the console is closed) and then shows organization aggregates and excerpts; the identity in
every log line comes from the key. Reading a whole call needs the **investigator** role and a
stated purpose (`pattern_review` / `citizen_request` / `partner_request`); the server writes
`analyst · case.open · incident:<id> · purpose` to `data/processed/audit_log.jsonl` before it
answers (ADR D20). That log is a keyed hash chain (`QORGAN_AUDIT_CHAIN_KEY`):
`python -m qorgan.audit verify` names the first edited, deleted or reordered entry. Analysts
can **confirm / dismiss / merge** an organization; the verdict is stored as an append-only
event keyed by the operation's numbers, so it survives re-clustering (a dismissed operation
drops to 20 % priority; ADR D24). Per-person keys are a stand-in for SSO in a deployment.

**Open demo access (on in the hosted demo).** `QORGAN_ADMIN_OPEN_ACCESS=analyst|investigator`
(default `off`) admits a visitor *without* a key as the reserved identity `public-demo` in that
role, so a jury can open the console without credentials. Only who may enter changes: every
action is still audited, under `public-demo`, and the audit-chain key is still required. Roles
and the stated purpose still apply. A presented key keeps its own identity, and a wrong key is
still refused. Use it only on a server holding fabricated data: anyone with the URL can read
what the role allows (ADR D59).

## Partner API (`/api/v1`) — consented reports in, aggregates out

A bank fraud desk, telecom or hotline can feed confirmed cases into the analyst layer and
read back the organization-level picture — without ever sending call content it does not
have to. It is a second *consented* ingress, not a bulk feed (ADR D19):

- **Auth**: `X-API-Key` per partner; keys live in the server env
  `QORGAN_PARTNER_API_KEYS="bank_a:<secret ≥16 chars>[:daily_quota],telecom_b:<secret>"`
  (unset = the API is closed; `QORGAN_PARTNER_DAILY_QUOTA` is the default budget).
- **One report per request**, preferably **structured tactic hits**; a transcript is accepted
  only if the partner already PII-scrubbed it (the server checks and refuses otherwise,
  without echoing it). `consent_basis` (a code from the data-sharing agreement) is required.
  The caller number is hashed on receipt like a citizen report. `partner_reference` makes
  retries idempotent.
- **Limits**: 60 requests/min and a rolling 24 h quota per partner (`X-Quota-Limit` /
  `X-Quota-Remaining` on every response); a content-free audit line for every action
  (`data/processed/audit_log.jsonl`).
- **Export**: `GET /api/v1/organizations` returns aggregates only — no numbers, no digests,
  no transcripts. Partners can `DELETE` only their own receipts.

```bash
export QORGAN_PARTNER_API_KEYS="bank_a:replace-with-a-32-char-secret-00000000"
python -m qorgan.api    # OpenAPI at http://localhost:8000/docs (scheme: PartnerApiKey)

# 1. a confirmed case as structured signals (preferred shape)
curl -s -X POST http://localhost:8000/api/v1/reports \
  -H "X-API-Key: replace-with-a-32-char-secret-00000000" -H "Content-Type: application/json" \
  -d '{"consent_basis":"customer_consent","tactic_ids":["otp_request","safe_account"],
       "phone_number":"+7 700 555 66 77","partner_reference":"CASE-2026-0912"}'
# -> 201 {"receipt_id": "...", "number_prefix": "+7 700 ***", "quota": {"limit":200,"used":1,...}}
# 2. the same case again -> 200 "duplicate", nothing stored, quota untouched
# 3. the organization-level picture (aggregates only)
curl -s http://localhost:8000/api/v1/organizations?locale=ru -H "X-API-Key: replace-with-a-32-char-secret-00000000"
# 4. withdraw a report
curl -s -X DELETE http://localhost:8000/api/v1/reports/<receipt_id> -H "X-API-Key: replace-with-a-32-char-secret-00000000"
```

Signals-only reports (no transcript) are stored, receipted, deletable and counted, and are
placed into organizations through the number graph alone (ADR D23).

## Publish model/data updates (maintainers)

After a retrain: `hf auth login` (write token) then `python scripts/hf_upload.py` — it
uploads `models/linear` **together with** `data/lexicon` (the bundle hash-validates the
lexicons) and the scrubbed dataset splits. Never widen the dataset patterns: the other
`data/processed/` files (incidents, citizen reports) and raw `data/synthetic/` must stay
off the Hub.

## Layout
`src/qorgan/` (server: API, config, audit, classifier, explain, live, analytics, reports, privacy,
data, asr, eval, web) · `site/` (the product: landing, live call, analyst console, `core/` on-device
modules, service worker) · `android/` (TWA, WebView shell, Kotlin core) · `app/` (Streamlit dev
harness, not the product) · `data/` (taxonomy, lexicons, provenance) · `scripts/` · `tests/`,
`tests_js/` · `docs/`.

## Status & limits
**Level 1 analysis already runs on the device** — `site/` is a static PWA that loads the int8
ONNX embedder and the exported heads and scores in the browser; no route accepts audio. What
remains unproven is **mobile**: the light voice mode has never run on a real phone (PLAN C2).
Operator-level interception is off the roadmap for good (ADR D12).

The live-mic path is real (Vosk KK+RU live words, a GigaAM pass per phrase) but
speakerphone-quality ASR — especially Kazakh — is the accuracy bottleneck; the meter's
confidence weighting absorbs some of it. Transient mid-call latching on legitimate calls is
measured by `eval.stream` rather than estimated: currently **3/60 on `test`** and **1/19 on the
clean authored negatives**, with the inspected anchors reported separately. The weakest call
language on the independent judges is Kazakh (recall 0.383 vs Russian 0.564) — see the README's «Ограничения» section. English is a UI locale only since ADR D70. Real call recordings do not exist yet (A3);
every number is synthetic or author-written.
