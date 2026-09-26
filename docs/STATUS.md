# Project Status & Handoff — Qorğan

_Last updated: **2026-09-25**. Current-state doc for anyone picking the project up. Read this,
then `docs/PLAN_2026-09.md` (the post-verdict plan and what is open), `docs/DECISIONS.md`
(ADRs D11–D49), `docs/eval_report.md` (numbers, with intervals). The July sprint log below is
kept as history._

## TL;DR (September 2026)
- **What it is:** scam-pattern decision support for KK/RU phone calls. Level 1 (citizen) runs
  **on the device**: `site/` pages load `multilingual-e5-base` int8 ONNX + exported head
  weights and score in the browser (`site/core/`, a 1:1 port of the Python classifier with
  golden parity tests). Level 2 (analyst) is fed **only** by consented reports.
- **Privacy by architecture, enforced by tests** (`tests/test_architecture.py`): no route
  accepts audio; `/api/analyze` persists nothing; the analyst store has a single ingress
  (`POST /api/reports`) plus the consented partner ingress (`POST /api/v1/reports`,
  API-key, quota, audit, aggregates-only export — ADR D19); numbers stored only as salted HMAC digests + `+7 700 ***`
  prefixes; transcripts PII-scrubbed; every report has a receipt, `DELETE` forgets it
  everywhere, `python -m qorgan.reports.purge` applies retention.
- **Model:** e5-base int8 embeddings (same graph server + device, one text per run) ⊕ 6
  interpretable features → calibrated LR; **threshold 0.59**. Retrained 2026-09-18 with 20
  legit-style scam paraphrases + 45 institutional-register legit negatives (ADR D27; uncommitted). With the A6 meter (ADR D29) every stated gate held on the server's ORT 1.27; **restated on the device's own embeddings 2026-09-21 (ADRs D32/D33)**: test 0.000 / 0.953 · authored 0.000 / 0.889 · ood 0.000 / 0.886 (corpus repaired, ADR D34), styled FPR 0 everywhere, streaming authored 3/24 (one transient latch over the A6 gate, an inspected anchor) & 15/18, browser gate 0/200. Eval
  @0.59: test FPR 0.000 [0, 0.068] / recall **0.953** · authored_heldout 0.000 [0, 0.142] /
  1.000 (clean subset n=19 negatives → [0, 0.176]) · ood 0.000 [0, 0.048] / 0.844 ·
  adversarial cue-free 0.927 · adversarial_legit 0.826 (was 0.651).
  **`authored_heldout` is hand-written, not real calls** — the locked real-call set (PLAN A3)
  does not exist yet; that is the biggest open item.
- **The honest headline (2026-09-21, ADR D35):** on `shift` — 66 calls written by a *second
  generator* (Claude, no sight of corpus / prompts / lexicons; `python -m qorgan.data.shift_set`)
  — the same model has **recall 0.242 [0.111, 0.423] (8 / 33), FPR 0.030**; ru 0.455 · kk 0.182 ·
  mixed 0.091. Every other split shares its generator with train. The cue lexicon fires on 6 / 33,
  the embedding head puts textbook prize / customs / relative-in-trouble scams at ≤ 0.10; the
  reassurance feature and the FPR story hold. Do not extend the lexicon from this set. The
  utterance-level heads were a measured no-go (D36); the e5-large server tier is D37.
- **Register diversity (2026-09-24, ADR D42):** the cross-generator gap was half house style.
  45 register-varied scams + 74 negatives (31 carrying the reassurance counter-signal) folded
  into train → `shift` recall **0.242 → 0.364**, test 0.953 → 0.984, ood 0.886 → 0.932,
  adversarial 0.917 → 0.945, legit-sounding 0.807 → 0.835, FPR 0.000 on test/authored/ood.
  Cost, on record: `shift` FPR 0.030 → 0.061 (one pushy bank sales call at 0.623) — both shift
  false positives are now ledger-inspected rows (ADR D43), so the split reports
  `shift (clean)` 0.000 [0.000, 0.112] / 0.364 (n=64) alongside the full row. Threshold
  stays 0.59 — 0.65 would remove that call but loses 3 true positives AND would stop a single
  hard signal latching the meter (floor 61 < 65).
- **STT Tier B (2026-09-24, ADRs D40/D41):** language locking — after 3 voted utterances keep
  only the winning recogniser, reopen both on a confidence drop — costs 0 cue detections and
  1.3 % of ru+kk utterances, and saves ~39 % of decodes on a five-minute call (18 % on the
  short demo dialogues); 85 % of calls never reopen. Implemented in the reducer as narrowing
  `state.languages`, **shipped OFF** (`QORGAN_ASR_LOCK_AFTER=0`) because the one risk that
  matters, code-switching, is the one thing synthesised audio cannot produce. Grammar-
  constrained decoding measured and rejected (worse in ru, +2/30 false fires in kk, a third
  decode).
- **Cue matching survives the recogniser (2026-09-23, ADR D39):** cues were matched by exact
  substring, so one mis-recognised character killed the hard-signal floor. Measured on 272 real
  recogniser outputs (`data/asr_capture/`), then fixed two ways — a bounded-edit matcher over
  de-spaced text (`classifier/cue_match.py` + `site/core/cue-match.js`, 1:1, 3,200-case fixture)
  and three behaviour-based Kazakh `remote_access` cues. Cue recovery **67 % → 76 %** (lexicon
  fixes Kazakh 76→86 %, matcher fixes Russian 74→84 %), false fires 0/160 throughout, every
  gate held after the retrain. `cue_matcher_version` is now part of the bundle contract.
- **The cloud second opinion generalises (2026-09-22, ADR D38):** the `llm` backend (Gemini 2.5 Pro,
  prompt now grounded in the 15 taxonomy ids) scores `shift` **33 / 33 · 0 / 33**, `authored_heldout`
  18 / 18 · 0 / 24, `test` 1.000 recall with 4 / 51 FPs. Device = privacy tier, cloud = accuracy
  tier, offered on the citizen's explicit request (D11). Client hardened: one client per process,
  transport timeout, 180 s wall-clock deadline + retries (the eval used to hang on idle sockets).
- **Tests:** `pytest -q` → 1038 offline (+2 skipped until a real held-out set exists) · `npm test` → 28 (JS core parity, the ASR vote/alignment reducer, the int8
  runtime gate on 200 transcripts — which needs the self-hosted model files). Branch `sanzh-ts`.
- **Verified in Chromium (2026-09-24, Playwright), after the D39/D42 model changes:** all
  **three** demo scenes now ship in the replay path (the Kazakh one existed only in the ASR
  bench until today) — RU scam **100/100 CRITICAL**, **KK scam 97/100 CRITICAL** with
  secrecy / otp_request / safe_account at weight 1.00 (verbatim cue hits), real bank call
  **26/100 LOW, no tags**. Kazakh advice renders from `advice_kk.yaml` on a fresh run.
  **Network: zero POSTs and zero requests carrying call content** — only static GETs for the
  model, config, tokenizer and WASM runtime. Known wart: switching the ru/kk locale mid-call
  does not re-render advice already on screen (a fresh run is correct).
- **Verified in Chromium (2026-09-14, Playwright):** scene 1 on-device → 90/100 CRITICAL with
  cue-grounded tags/advice/summary; scene 2 (real bank call) → 5/100 LOW; report submit →
  receipt, digest + `+7 700 ***` on disk, raw number absent; delete → gone. Only network
  calls with call content: `POST /api/reports`, `DELETE /api/reports/{receipt}`. Model load
  ~3 s from localhost, ~0.6 s first inference (WASM/WebGPU).

## 2026-09-25 — fresh-clone recheck + first improvement loop (branch `dev/loop`, uncommitted)
- **Fresh clone was not self-deployable:** `deploy_bootstrap.py` probed/retrained the model
  before downloading the int8 embedder it needs → ONNX `NO_SUCHFILE`, and the Docker build
  runs the same script. Fixed: `ensure_embedder` runs first, head weights are copied after the
  bundle (`ensure_web_weights`); `ensure_corpus` tops up missing files (now incl. `shift`)
  without overwriting local splits. `tests/test_deploy_bootstrap.py`.
- **Lean test env:** the suite no longer needs torch / transformers / streamlit / hdbscan to
  collect — xlmr, Streamlit and hdbscan-overlay tests skip without them. Runtime set used:
  pydantic, dotenv, pyyaml, pandas, numpy, scikit-learn, networkx, fastapi, uvicorn,
  huggingface_hub, onnxruntime 1.21, tokenizers, num2words (+ pytest, httpx).
- **Reproduced (server `onnx` proxy, threshold 0.59):** test 0.020 / 0.969 · authored 0.000 /
  0.944 · ood 0.014 / 0.932 · **shift 0.030 / 0.424** (clean 0.000 / 0.424) — consistent with
  the device headline within the documented proxy gap (D33).
- **D44** report review (editable, consent-gated, redaction preview; Chrome e2e
  `tests_js/tools/e2e_report_review.mjs`, `QORGAN_E2E_CHANNEL=chrome`). **D45** Cyrillic-glued
  card/IIN redaction + pre-publish PII gate; the Hub's `ood.jsonl` still carries one
  unscrubbed IIN until a maintainer re-runs `scripts/hf_upload.py`.
- Tests: `pytest` 1132 passed / 10 skipped (lean env) · `npm test` 49 / 49.

### 2026-09-25 (second pass) — Level-2 access control, a Kazakh/Russian citizen page, a lean runtime
- **D46 analyst access:** `/api/admin` now needs a per-person key (`QORGAN_ANALYST_KEYS`) and
  fails closed; a full transcript needs the `investigator` role, a purpose code and fits in 30
  opens/hour; the excerpt view no longer leaks ~75 % of a call through its trigger phrases; the
  audit log is an HMAC chain (`QORGAN_AUDIT_CHAIN_KEY`, `python -m qorgan.audit verify`). The
  partner API is closed without the audit key too.
- **D47 citizen page:** one ҚАЗ / РУС / ENG control drives all text; switching mid-call
  re-renders advice, summary and review in place (the known wart is fixed); plain-language
  copy with the "a human decides, the AI can be wrong" disclosure; a size notice before the
  ~300 MB first download. **Kazakh strings awaiting native review** (all in `site/i18n.js`):
  «динамик» for speakerphone (`hero.lede`, `mic.copy`); «Талдаушы кабинеті», «талдаушы
  кезегінде», `report.intro`; «түбіртек нөмірі»; «құсбелгіні алып тастаңыз»;
  `report.phone_help`; `report.consent` (also legal wording); «100-ден {score}», «Қоңырау ·
  фразалар саны: {n}»; `band.*_desc`; the "how it works" steps; «Алғаш іске қосқанда», «Wi-Fi
  арқылы жүктеген дұрыс»; `mic.reason_*`, the tagline, «Сапаны бағалау».
- **D48 lean runtime:** torch / transformers / Streamlit / Gemini moved to extras; the Docker
  image no longer installs torch; a test blocks the extras and scores a call without them.
- **Manifest fixed:** `counts.*.positives` meant "not a hard negative", so `authored_heldout`
  claimed 27 positives for 18 scams. Counts now carry `scam` / `legit` / `hard_negatives`
  (`positives` kept, = `scam`); the 0.5 label cut is one constant (`schema.SCAM_RISK_THRESHOLD`)
  instead of six copies. Local manifests recomputed from the unchanged split files; the Hub copy
  updates on the next `scripts/hf_upload.py`.
- **`docs/LEGAL_ASSESSMENT.md`** (new; not legal advice): data-flow inventory against KZ law
  (PD Law 94-V as amended in 2025–26, AI Law 230-VIII, the 2026 Constitution, CPC, NBK
  Resolution 54), with primary sources. It found paths the product story says do not exist —
  `backend=llm` is caller-selectable on `/api/analyze`, `/api/live/session` and the admin
  analysis (text to Gemini abroad, and the LLM cache writes verbatim phrases to disk);
  `/api/live/session/*` holds transcripts in server memory and its `/report` has no consent
  step; retention runs on the client-supplied timestamp and nothing schedules the purge;
  spoken-word numbers (the on-device recogniser's output) escape the scrubber. These are the
  next loop's work.
- Repo hygiene: the in-repo `CLAUDE.md` (July sprint brief) was rewritten to the current state.
- Tests: `pytest` 1204 passed / 10 skipped · `npm test` 60 / 60 · Chrome e2e: admin auth,
  live i18n, report review all pass (re-run independently on a scratch copy of `data/`).

### 2026-09-26 — closing the undeclared data paths (D49) + review fixes
- **D49:** the cloud tier is off by default and needs per-request consent when on (never
  cached, never for analysts); the server-side live-session API (a consent-free second ingress
  holding transcripts in memory) is retired, and every write route is now named in the
  architecture test; retention runs on the server's `received_at`, the client timestamp is
  bounded, and the server purges at startup + daily; deleting a report also clears its traces
  from analyst feedback; each report stores the version of the consent wording it was sent
  under. New settings: `QORGAN_CLOUD_TIER`, `QORGAN_REPORT_PURGE_INTERVAL_HOURS`.
- **Independent review** of D44–D48: no critical/high; both mediums fixed (partner API audits
  before it stores/deletes; unchained audit lines are sealed only by an explicit
  `python -m qorgan.audit seal`), one low fixed, two documented.
- **HF cards drafted:** `docs/hf/MODEL_CARD.md`, `docs/hf/DATASET_CARD.md` (licence field
  "undecided" until the maintainers choose one); not uploaded.
- **Open, top of the list:** spoken-number redaction for microphone-mode reports (numbers as
  words escape `scrub_text`); a licence; native Kazakh review of `site/i18n.js`; the legal
  owner for `docs/LEGAL_ASSESSMENT.md` §6.
- Tests: `pytest` 1214 passed / 10 skipped · `npm test` 60 / 60 · Chrome e2e: report review,
  live i18n, admin auth pass on a scratch copy of `data/`.

## How to run (September)
```bash
pip install -e . && cp .env.example .env     # set QORGAN_NUMBER_HMAC_KEY (see file)
python scripts/deploy_bootstrap.py           # corpus + heads from HF, int8 ONNX -> site/models/, L2 seeds
python -m qorgan.api                         # http://localhost:8000  (landing · live call · admin)
pytest -q && npm test                        # Python suite · JS parity suite
# retrain (needed after ANY data/lexicon change; also re-exports web/weights.json):
python -m qorgan.data.build_corpus && python -m qorgan.classifier.linear_train
python -m qorgan.data.shift_set                # the second-generator eval split (A12) -> processed/shift.jsonl
python -m qorgan.web.client_config && python scripts/export_parity_fixtures.py   # refresh client config + golden fixtures
QORGAN_CLASSIFIER_BACKEND=linear python -m qorgan.eval.run --split test --split authored_heldout --split ood
```
Key env: `QORGAN_EMBED_BACKEND=onnx` (default; `sentence-transformers` = fp32 PyTorch, needs a
bundle trained that way), `QORGAN_RISK_THRESHOLD=0.59`, `QORGAN_NUMBER_HMAC_KEY` (required to
accept numbered reports), `QORGAN_REPORT_RETENTION_DAYS=180`.

## Repository map (September additions)
| Path | What |
|---|---|
| `site/core/` | On-device core (lexicon, head, attribution, score, explain, recommend, meter, session, summary, device, embed-worker) + `qorgan-config.json`, `scenarios.json` (generated by `qorgan.web.client_config`). |
| `site/models/` | `weights.json` (exported heads, committed) · `Xenova/multilingual-e5-base/` (278 MB int8 ONNX, downloaded by `deploy_bootstrap`, gitignored). |
| `site/sw.js`, `manifest.webmanifest`, `icons/` | PWA shell: offline after first load; `/api/` never cached. |
| `tests_js/` | Node built-in test runner: golden parity (`fixtures/parity.json`, generated) + integration gate. |
| `src/qorgan/privacy/`, `src/qorgan/reports/`, `api_reports.py`, `api_ratelimit.py` | Number hashing, minimised report storage, receipts/deletion/purge, the consented ingress. |
| `site/core/asr.js`, `asr/web_models.py`, `api_limits.CrossOriginIsolationMiddleware` | On-device dual-language speech recognition (Vosklet), model tarball packaging, the COOP/COEP headers the live page needs (ADR D26). |
| `eval/meter_sweep.py` | Offline live-meter sweep: per-turn traces cached once, variants replayed through the production meter (the A6 methodology, ADR D29). |
| `eval/asr_realism.py` | Paired clean-vs-ASR-styled evaluation (FPR first, cue/reassurance survival, `--drop-latin` worst case) — the A10 gate (ADR D31). |
| `classifier/cue_match.py`, `site/core/cue-match.js` | Cue matching that survives the recogniser (ADR D39): verbatim first, then bounded-edit over de-spaced text; pigeonhole prefilter; versioned as part of the bundle's feature contract. |
| `scripts/spikes/asr_cue_survival/` (`capture_dual.py`, `lock_sweep.py`, `grammar_probe.py`), `data/asr_capture/dual.jsonl` | Tier B evidence (D40/D41): both recognisers per utterance in dialogue order; offline replay of locking policies; the grammar-decoding probe. |
| `scripts/spikes/asr_cue_survival/`, `data/asr_capture/` | The capture + measurement harness behind D39: corpus utterances spoken by `say`, decoded by the shipped Vosk models; recovery / false-fire / clean-drift tables. |
| `data/shift_set.py`, `data/authored/shift/` | The generator-shift split (A12, ADR D35): 66 calls by a second generator, grounded from verbatim phrases, scrubbed, refused if they overlap any split; `python -m qorgan.data.shift_set` → `processed/shift.jsonl` + manifest. |
| `data/clean.py` | Generation-artefact repair run by `build_corpus` (and by hand on `ood.jsonl`): unwrap/split jammed turns, delete backspaces, decode escapes; rows with lost Kazakh letters are dropped and listed in the manifest (ADR D34). |
| `classifier/device_embed.py` | The `device` embed backend: the browser's own vectors through the headless-Chromium bridge (`npm run device:serve`), sqlite-cached; a warm cache works with the bridge down (ADR D33). |
| `tests_js/tools/device_runtime.mjs` + `device_embed_server.mjs` + `browser_gate_embed.mjs` | Playwright tooling: the site's real worker as an embedding service, and the browser-gate fixture capture (ADRs D32/D33). |
| `data/asr_style.py` | Deterministic ASR-register styler (`num2words` ru/kz) + the seeded train augmentation `build_corpus` folds in (ADR D31). |
| `analytics/feedback.py` | Analyst confirm / dismiss / merge: append-only events keyed by the operation's numbers, applied at read time (ADR D24). |
| `api_partner.py`, `api_partner_export.py`, `partners.py`, `audit.py`, `reports/partner.py`, `api_limits.py` | Partner API (`/api/v1`): key registry, one-report-per-request ingress, receipt-time quota, content-free audit log, aggregates-only export (ADR D19); app-wide body cap + 422s that never echo input. |
| `data/real_intake.py`, `data/real_allocation.py`, `scripts/ingest_partner_calls.py`, `docs/DATA_INTAKE.md` | Real-call intake (A8): batch validation, scrub + hashed linkage, first-come allocation, hash-locked held-out set. `data/real/` is gitignored. |
| `src/qorgan/classifier/web_bundle.py`, `embed.py::OnnxEmbedder` | JSON export of the heads + reference scorer; the int8 ONNX embedder. |
| `src/qorgan/eval/intervals.py`, `src/qorgan/data/ledger.py`, `data/anchors/inspection_ledger.yaml` | Clopper–Pearson / bootstrap intervals; the inspection ledger behind the `(clean)/(inspected)` rows. |
| `tests/test_architecture.py` | The privacy invariants as tests. |

## Open threads (see PLAN_2026-09 for owners/estimates)
- **Cross-generator recall (A12, ADR D35):** 8 / 33 on `shift`. Levers that are honest: real calls (A3), a third generator for *training* data, the `llm` cloud second opinion (Gemini 2.5 Pro, taxonomy-grounded prompt since 2026-09-21: 33 / 33 recall and 0 / 33 FPR on `shift` — the calls the device model misses are exactly the ones the cloud tier catches), the e5-large tier (D37). Not honest: lexicon entries mined from `shift`.
- **Real data (A3):** no real calls yet; everything is synthetic or author-written. Stakeholders were asked for both scam and legit recordings. The intake protocol + tooling exist (A8: `docs/DATA_INTAKE.md`, `scripts/ingest_partner_calls.py`, hash-locked `real_heldout_v2`, `tests/data/test_heldout_lock.py` skips until a set exists); the legal owner has not reviewed the protocol yet.
- **On-device ASR (B9, ADR D26):** microphone mode is live on **desktop** browsers — `site/core/asr.js` runs the small KK + RU Vosk models in Vosklet (one WASM instance each), votes per utterance, and feeds the same session/meter as replay; the live page and `/core/*` are served cross-origin isolated (COOP/COEP, `no-cache`). The three demo scenes pass through the recogniser (RU scam 81, bank call 14, KK scam 61). **Phones stay disabled until the Android bench passes** (`scripts/spikes/vosklet_bench/`). `deploy_bootstrap` packages the tarballs (~106 MB, gitignored). Whisper is a no-go (D25); server-side audio is gone for good (D12).
- **B8 static quantisation — closed, no-go (2026-09-17, ADR D18):** int8 graphs drift
  ~0.6 % cosine across ONNX Runtime versions; a statically calibrated graph cost 5 pts of
  fidelity and did not move the cross-runtime floor. Decision-level parity (0 flips / 28) is
  the guarantee. The browser loads the graph named by the server's `QORGAN_EMBED_ONNX_DIR`
  (`qorgan-config.json::embedder`).
- **Admin auth:** `/api/admin` is unauthenticated in the demo (pre-existing). Since C4 (ADR D20) reading a full transcript is an audited `open` action naming `X-Analyst-Id` — a real deployment must put SSO in front of the admin routes so that name is trustworthy.
- **Novelty needs support (C10, ADR D21):** a number-less single report can no longer create a "novel scheme" callout (29 false flags under number rotation → 1); two such reports, or one with a number, still can. `python -m qorgan.eval.cluster` is the regression check.
- **Adversarial (A9/A9b, ADRs D22/D27):** cue-free paraphrases cost nothing (0.927); the legit-sounding adversary dropped recall to 0.651 → addressed with paired train data (20 legit-style scams + 45 institutional-register legit negatives): 0.826 with every FPR gate held — the one hairline streaming latch it introduced is removed by the A6 meter (see eval report). Price: the two RU legit anchors moved from ~0.31 to ~0.45 (still clear). Only real calls (A3) can confirm the knee.
- **Cross-runtime parity (ADR D28):** measured on 200 transcripts, the int8 runtime residual is ~1 % decision flips (shipped model 1.5 %), |Δrisk| p95 0.10 / max 0.27; the gate now asserts ≥ 99 % same decision. The old "0 flips / 28" was the small set.
- **A6 meter (ADR D29, uncommitted):** the latch arms from the third utterance unless a hard signal fired (`QORGAN_METER_MIN_TURNS_TO_ARM=3`; damping knob `QORGAN_METER_SHORT_WINDOW_TURNS`, off). With the retrained heads: authored false-latch 2/24, alert-hit 16/18, median turns-to-alert 3 (+1). The demo scam scene latches on its third line.
- **Per-tactic thresholds (ADR D30, uncommitted):** the tactic head over-predicted (2.86 tags per test dialogue vs 1.91 true); each tactic now has its own cut, tuned by max-F1 over 0.50–0.70 on out-of-fold train + val at training time (`multilabel.out_of_fold_proba`, `calibrate.tune_tactic_thresholds`; < 8 tuning positives keeps 0.5). End-to-end: test micro-F1 0.733 → 0.770 (P 0.61 → 0.76), tags/dialogue 1.98; ood 0.396 → 0.496; authored within noise; `urgency` / `verification_ploy` lose test recall (their 208 / 160 tuning positives disagree with test's 43 / 29 — on record, not tuned away). Risk head byte-identical, so no FPR gate moves; JS decodes 1:1 (fixtures regenerated).
- **ASR realism (A10, ADR D31, uncommitted):** `python -m qorgan.eval.asr_realism [--drop-latin]` scores every eval dialogue clean and ASR-styled (`data/asr_style.py`: lowercase, no punctuation, numerals → words via `num2words`), paired, FPR first. Under the device-faithful runtime the register alone cost test FPR 5/52 on clean-trained heads → `build_corpus` adds an ASR-styled copy of every train row (`QORGAN_ASR_STYLE_TRAIN_FRACTION=1.0`; train 711 → 1,422; generated at build) and the two hyphenated KK cues have ASR forms. Same runtime before → after: styled test FPR 5/52 → 0, streaming test false-latch 5/52 → 3/52, adversarial-legit 0.872; costs: one server-side ood FP (`ood_neg_legit_bank_call_mixed_3`, 0.67 server / 0.15 browser) and a third transient authored latch (`real_neg_bank_fraud_alert_ru`, inspected anchor). Rollbacks: `models/linear_d30` (needs the pre-D31 lexicon — it hash-checks — and was trained on ORT 1.27).
- **Runtime truth (ADR D32, uncommitted):** the browser's **WebGPU** path was broken for the int8 graph (cosine 0.78 to the server, slower than WASM) → `embed-worker.js` is WASM-only. The D17/D18/D28 "runtime residual" was an **ONNX Runtime version gap** (Python 1.27 vs onnxruntime-node 1.21): `onnxruntime==1.21.*` is pinned (`tests/test_runtime_pin.py`), Node is now bit-identical, and `npm test` drops from ~30 min to minutes. The browser (onnxruntime-web 1.22-dev, WASM) is a third build at cosine 0.982 / min 0.949: the runtime gate is `tests_js/integration/browser_gate.test.mjs` on `tests_js/fixtures/runtime_gate_browser.f32`, captured by `npm run gate:browser` (Playwright dev dependency, `npx playwright install chromium` once), asserted at the measured level (≥ 97 %, p95 < 0.15, Jaccard ≥ 0.9; 5/200 flips on both the old and the new heads). **Headline under the pinned runtime:** test 0.019 [0.000, 0.103] / 0.953 · authored 0.000 [0.000, 0.142] / 0.944 · ood 0.013 [0.000, 0.072] / 0.867 — the earlier "0.000 everywhere" was the server's 1.27 kernels. Follow-up **B10**: train/eval on device embeddings.
- **Device embeddings (B10, ADR D33, uncommitted):** `QORGAN_EMBED_BACKEND=device` (`classifier/device_embed.py`) posts texts to `npm run device:serve` — the site's real `embed-worker.js` in headless Chromium (Playwright; `--pages 4` ≈ 0.1 s/text; sqlite cache `data/cache/device_embeddings.sqlite` keyed by model + browser build + text). The shipped heads are trained and every eval table computed with it; `metadata.json::embed_backend = device`; the server's `onnx` backend loads them as the documented proxy (`linear_train._PROXY_BACKENDS`). **Headline, on the device (corpus repaired, ADR D34):** test 0.000 [0.000, 0.070] / 0.953 (n=115) · authored 0.000 [0.000, 0.142] / 0.889 (16/18: `customs_ru` 0.515, `prize_phone_kk` 0.514) · ood 0.000 [0.000, 0.049] / 0.886 (n=118) · styled FPR 0 on every split · cue-free adversarial 0.917, legit-sounding 0.815 · streaming test 3/51 & 62/64, authored 3/24 & 15/18 — with the inspection ledger applied (`eval.stream` reports `(clean)`/`(inspected)` rows now): clean 1/19 [0.001, 0.260], inspected 2/5 · **browser gate 0/200, |Δrisk| 0.000, Jaccard 1.000**. The previously shipped heads decide the same on the device (their 1/52 and 1/75 FPs were the server runtime's). Server proxy vs device: 6/200, p95 0.114 — reported, not gated. Retrain needs the bridge running; the cache makes reruns free. Rollbacks (gitignored): `models/linear_d32` (native-1.21-trained), `linear_d30` (1.27, pre-D31 lexicon).
- **Streamlit `app/`:** kept on purpose (D4, decided 2026-09-21) as the easily-run local demo / dev harness — `streamlit run app/streamlit_app.py`, degrades to the `mock` backend without a model. Not a gate, not the deployed product: it scores in the server process and its mic mode sends audio to wherever Streamlit runs.
- `models/linear_fp32/` (the fp32-trained heads) and `models/linear_embed_only/` are the local rollback bundles (gitignored); `models/xlmr/` and the e5-small experiment were deleted on 2026-09-14.

---

# July 2026 sprint log (history)

## What landed 2026-07-14 → 15 (the sprint log)

**1 · Live citizen pipeline** (from the real-time design spec / `task.md`): committed
utterances → head+tail rolling window → `score()` → **0–100 suspicion meter** (asymmetric EMA,
hard-signal floors 61/81, 55/45 hysteresis latch; `live/meter.py`, `live/session.py`) →
tactic→advice engine (`explain/recommend.py` + `advice_{ru,kk}.yaml`) → post-call summary +
**consent-gated editable report** (`live/summary.py` → `data/processed/citizen_reports.jsonl`).
Meter knobs: `QORGAN_METER_ALPHA_UP/_DOWN`. UI: `app/live_view.py`.

**2 · Live microphone**: Input radio in the Live tab — *Replay script* (default, zero-setup) ·
*Microphone — browser* (streamlit-webrtc) · *Microphone — local* (sounddevice). ASR = **dual
Vosk KK+RU streaming recognizers, per-utterance word-confidence voting** (`asr/vosk_stream.py`;
small models auto-download to `~/.cache/vosk`, config `QORGAN_VOSK_MODEL_KK/_RU`). Plumbing:
`asr/capture.py` (drop-oldest queue); wiring: `app/mic_live.py` (worker threads → event queue →
`st.fragment` drain; threads never touch Streamlit). Optional extra `pip install -e ".[live]"`,
graceful degrade without it. Verified end-to-end with synthesized speech (macOS `say`, RU+KK)
through real Vosk + the linear model: scam scene → 84/Critical/latched; hard negative stays Low.
That e2e also fixed `predict._merge_cue_evidence`: a verbatim cue hit now upgrades the tactic
tag to weight 1.0 (was suppressed by the head's lower weight → hard-signal floor never fired).

**3 · Model-improvement sprint (all FPR-gated, one attempt rolled back)**:
- `authored_heldout` widened 27→**42** anchors → immediately exposed a real KK/mixed FP (legit
  tariff notice @ 0.809).
- New **streaming eval harness** `python -m qorgan.eval.stream`: false-latch rate (live FPR
  analog), alert-hit rate, time-to-alert.
- **KK boundary stabilized**: 15 hand-written KK legit hard negatives
  (`data/augment/kk_legit_negatives.jsonl`, train-only, leakage-tested vs anchors) + KK
  reassurance terms (`қажеті жоқ` …) + payment sensitive-terms. A lexicon-only attempt FAILED
  gates (the joint-LR refit moved unrelated KK negatives; test FPR 0→0.019) and was rolled
  back — **pair lexicon edits with training data**, that combination passed.
- **Cue-lexicon ASR variants** ("безопасной счёт", "код из сообщения", KK bare imperatives) —
  the live hard-signal floor now fires on real Vosk output.
- Full ablation + honest caveats: `docs/eval_report.md` **2026-07-15 addendum**.

**4 · Analyst dashboard overhaul + citizen-report loop**: L2 extracted to `app/analyst_view.py`
— KPI row, novel-scheme callout, priority queue with **tactic-derived display names**
(`analytics/presentation.py`; never raw cluster ids), drill-down with tactic/activity charts
and span-highlighted representative script. **Report loop closed**: Live-tab reports →
`analytics/intake.py` (idempotent content-addressed ids; embeds ONLY new transcripts via the
`incident_embeddings.npz` cache `analytics/pipeline.py` now writes) → **"Ingest into
analysis"** button → number-graph placement (known number joins that org; unknown = novelty
candidate). Verified live: a real report joined the seeded `bank_security` org in ~9 s;
re-ingest is a no-op. Intake normalizes tz-aware report timestamps to the store's naive
convention (mixing crashed priority ranking).

## Repository map (what to read)
| Path | What |
|---|---|
| `CLAUDE.md` | Master brief, locked decisions, conventions. Auto-loaded by Claude agents. |
| `docs/SCOPE.md` · `ARCHITECTURE.md` · `DECISIONS.md` | Scope cut, system design, decision log. |
| `docs/eval_report.md` | **Model eval + FPR ablations** (incl. 2026-07-15 addendum). Source of truth for numbers. |
| `data/README.md` | Data + ASR-model provenance, PII scrubbing, augmentation (graded, ТЗ §9). |
| `src/qorgan/classifier/` | The classifier: predict interface, features, lexicon matchers, training. |
| `src/qorgan/live/` | Suspicion meter, live session, post-call summary/report. |
| `src/qorgan/asr/` | Batch whisper wrapper · replay/pseudo-stream · Vosk dual-stream · capture queues. |
| `src/qorgan/explain/` | Explainer + templates + the tactic→advice recommendation engine. |
| `src/qorgan/analytics/` | Level-2: clustering, novelty, ranking, presentation helpers, report intake. |
| `src/qorgan/eval/` | FPR-first tables (`run`), threshold tuner, **streaming eval** (`stream`). |
| `app/` | `streamlit_app.py` (shell + L1) · `live_view.py` · `mic_live.py` · `analyst_view.py` · `ui_shared.py`. |
| `scripts/` | `demo_seed.py` (L2 incidents) · `augment_reassurance_negatives.py` (Gemini) · `hf_upload.py`. |

## Classifier backends — `classifier/predict.py :: score(transcript, backend=...)`
One interface, backend chosen by `QORGAN_CLASSIFIER_BACKEND` (default `linear`):
- **`linear`** — the SHIPPED offline model. Hybrid risk head; embedding-only per-tactic head
  with **per-tactic decision thresholds** (ADR D30: tuned on out-of-fold train + val at
  training time, `metadata.json::tactic_thresholds`; verbatim cue hits upgrade tags to weight 1.0). Weights in `models/linear/` (gitignored);
  **hash-validates** `data/lexicon/*.yaml` — any lexicon edit forces a retrain, and per the
  sprint lesson should be **paired with training data** (see sprint log §3).
- **`mock`** — deterministic keyword safety net, zero deps/keys; the app auto-degrades to it.
- **`llm`** — Gemini structured classifier; needs `GEMINI_API_KEY`.
- **`xlmr`** — **ABANDONED** (fine-tune collapsed). `models/xlmr/` (~1.1 GB) is deletable.

### The hybrid feature (why the model looks the way it does)
The embedding-only model false-positived on realistic legit RU calls (`authored_heldout` FPR
0.143). Root cause = a **train/reality gap**: synthetic legit calls never _reassure_ ("we'll
never ask for your code"). Fix = **data + a reassurance feature together** (neither works
alone): `data/augment/reassurance_negatives.jsonl` (train-only) + `classifier/reassurance.py`
+ the request-cue features (`classifier/cue_lexicon.py`). The 2026-07-15 sprint repeated the
same pattern for **Kazakh** (KK legit negatives + KK reassurance terms). Rollback bundle:
`models/linear_embed_only/`.

## Data pipeline (`src/qorgan/data/`)
`generate.py` (Gemini synthetic corpus) → `build_corpus.py` (**scrub PII** → dedup →
deterministic split; folds `data/augment/*.jsonl` — reassurance + KK-legit negatives — into
**train only**; `authored_heldout` = 42 curated anchors, kept separate) →
`data/processed/{split}.jsonl` + `manifest.json`.
- **PII / publishing:** `data/processed/` splits and `data/augment/` are scrubbed →
  publishable. `data/synthetic/` (raw) is **NOT**. `data/processed/{incidents,organizations,
  citizen_reports}.jsonl` contain fabricated numbers / live report intake — **never publish**
  (the `hf_upload.py` allow-patterns already exclude them; don't widen them).

## Hugging Face repos
- **Models:** `sanzh-ts/govtech` — root = shipped hybrid bundle; `embed_only/` = baseline;
  `lexicon/` = the cue/reassurance yamls the bundle validates. **Model and lexicons must be
  uploaded together** (hash guard).
- **Dataset:** `sanzh-ts/govtech_ds` — scrubbed splits + `augment/`.
- **Publish** (write token): `hf auth login` then `python scripts/hf_upload.py`.
- **Pull instead of retraining:** `snapshot_download("sanzh-ts/govtech", local_dir="models/linear_hf")`
  then `export QORGAN_LINEAR_MODEL_DIR=models/linear_hf` (lexicons are committed in-repo).

## How to run
```bash
pip install -e .                      # Python 3.11+; add ".[live]" for microphone modes
streamlit run app/streamlit_app.py    # 3 tabs; mock backend if no model/key

# eval — full-transcript FPR tables + streaming false-latch/time-to-alert
QORGAN_CLASSIFIER_BACKEND=linear python -m qorgan.eval.run --split test --split authored_heldout --split ood --by-language
QORGAN_CLASSIFIER_BACKEND=linear python -m qorgan.eval.stream --split test --split authored_heldout --backend linear

# retrain from committed corpus (seconds, CPU; needed after ANY data/lexicon change)
python -m qorgan.data.build_corpus && python -m qorgan.classifier.linear_train

# Level 2: seed demo incidents -> organizations + embedding cache (needed for report ingest)
python scripts/demo_seed.py && python -m qorgan.analytics.pipeline

pytest -q                             # 697 tests, all offline
```
Alert threshold `QORGAN_RISK_THRESHOLD` = **0.55** (tuner now recommends 0.59–0.62 with equal
recall — headroom, deliberately not taken). Demo storyline (3 scenes) is in `README.md`.

## Known issues / open threads (for your own planning)
- **Transient streaming false-latch ~0.167**: ~1 in 6 legit calls latches the live meter
  mid-call, then recovers (full-transcript FPR is 0). Structural: early short windows are
  noisy. Candidates: minimum-turns before the latch arms, short-window damping. Now
  measurable via `eval.stream` — tune against it.
- **Plan Phase 3 (tail-tactic data top-up) superseded by ADR D30**: the weak per-tactic rows
  were over-prediction, not thin data — per-tactic thresholds fixed the operating point
  (test micro-F1 0.733 → 0.770). A top-up remains an option if real calls show a real gap.
- **Mild eval circularity, twice**: both the reassurance fix (2026-07-13) and the KK fix
  (2026-07-15) were motivated by inspected heldout FPs. Mitigated (fresh un-inspected sets,
  leakage tests) but keep honest in the pitch.
- **No `legit_telecom` negative category in the taxonomy** — covered at the data level
  (anchors + KK augment), a dedicated category would still be cleaner.
- **Vosk KK small-model accuracy** on far-field/synthetic speech is the live-mode bottleneck;
  meter confidence-weighting absorbs some of it.
- **Browser-mic mode** cannot run under AppTest (webrtc needs the real runtime) — one manual
  click-through before demoing.
- `models/xlmr/` (~1.1 GB) and `error_log.txt` are dead weight — safe to delete.
- Persistent working memory for agents lives in
  `~/.claude/projects/-Users-sanzhars-Documents-proga--govtech/memory/` (indexed by `MEMORY.md`).
