# Eval report — classifier backends

> **2026-09-13 note (PLAN_2026-09 A1–A2).** The split formerly called `real_heldout` is now
> **`authored_heldout`**: it is hand-written by the team (18 scam / 24 legit), not real calls,
> and five of its negatives were *read* during feature engineering (see
> `data/anchors/inspection_ledger.yaml`; the harness now reports `(clean)` / `(inspected)`
> rows). The tables below are July point estimates without intervals: `FPR 0.000` on 24
> negatives has a 95 % Clopper–Pearson interval of `[0.000, 0.142]`. The harness prints
> intervals now; this report is regenerated with them in A7.

> **2026-09-20/21 (ADRs D32, D33).** Everything above the 2026-09-20 addendum was computed on
> the server's ONNX Runtime 1.27; the device runs another build and single calls near the
> threshold move with it. The heads are now trained and the tables computed on the browser's
> own embeddings (`QORGAN_EMBED_BACKEND=device`) — the final addendum holds the numbers that
> describe what citizens run, with the browser gate at 0 / 200 by construction.

FPR is the **primary** metric (a false alarm on a real bank call destroys trust). Numbers
are reported on `test` **and** `authored_heldout` **separately**; `authored_heldout` (hand-written,
different source) is the honest generalization signal. Regenerate with:

```bash
python -m qorgan.eval.run --split test --split authored_heldout --backend <linear|llm|mock> --by-language
```

The CLI also prints a **per-tactic F1** table and a **recommended alert threshold** — the
cutoff maximising recall subject to `fpr <= 0.05` on `authored_heldout` (`eval/threshold.py`).

## Corpus
**833 dialogues** (scams + **adversarial hard negatives**: legitimate calls in the *same
domains* as scams — bank, gov, telecom, delivery — that superficially resemble a scam but
contain none of the tactics), balanced across RU/KK/mixed. Splits: **train 604 / val 113 /
test 116** (`data/processed/manifest.json`), plus a separate **27**-dialogue hand-written
`authored_heldout` anchor set (different source — the honest generalization signal).

## `linear` backend — **SHIPPING offline model (hybrid)** ✅
Frozen `multilingual-e5-base` embeddings **⊕ 6 interpretable features** → class-weighted,
**calibrated** Logistic Regression risk head + per-tactic LR (embedding-only) tactic head.
The risk head's input is `[768 embedding | 5 hard-signal request-cues | 1 reassurance]`
(`classifier/features.py`). Trains in seconds on CPU; the export is ~60 KB.

### The FPR fix — data + a reassurance feature (the two are inseparable)
The embedding-only model had a stubborn **authored_heldout FPR 0.143**, entirely 2 legitimate RU
calls (a bank fraud-alert, a telecom offer). Diagnosis (`docs/DECISIONS`-style, reproduced in
git history): these are a **train/reality gap** — the synthetic training negatives never do
what real institutions do, **proactively reassure** ("we will never ask for your code";
0/307 train negatives had this pattern). Two levers, neither sufficient alone:
- **Request-cue features** (`otp_request`/`credentials_request`/`safe_account`/`remote_access`/
  `secrecy`) are *absent* on the FPs → don't move FPR; but they recover recall by re-finding
  real scams, and give grounded "this call *asked* for your OTP" highlights.
- **27 reassurance hard-negatives** (`scripts/augment_reassurance_negatives.py`, Gemini,
  committed under `data/augment/`, added to **train only**) + a **reassurance feature**
  (`classifier/reassurance.py`: a sensitive term within a window of a negation-of-need phrase,
  never scam-secrecy). Data alone leaves FPR unmoved and hurts recall; the feature alone (no
  data) backfires (0 negative coverage → wrong sign). **Together they crush the FP scores
  (~0.99 → ~0.34)**, which opens headroom to operate at the tuned threshold.

### Ablation (fixed eval sets; alert threshold in parentheses)
| Model | test FPR/rec | **authored_heldout FPR/rec** | ood FPR/rec |
|---|---|---|---|
| embedding-only (baseline, @0.70) | 0.000 / 0.969 | **0.143 / 0.923** | 0.000 / 0.911 |
| hybrid (cues + reassurance + data, @0.70) | 0.000 / 0.953 | **0.000 / 0.923** | 0.000 / 0.778 |
| **hybrid @ tuned 0.55 (shipped)** | 0.000 / 0.953 | **0.000 / 1.000** | 0.000 / 0.867 |

0.55 is the `eval/threshold.py` cutoff (max recall s.t. FPR ≤ 0.05 on authored_heldout). On the
honest hand-written set the hybrid model is a **strict Pareto win** — false positives
eliminated *and* recall to 1.000 — for a small recall cost on the synthetic in-distribution
(test −1.6 pts) and disfluent-stress (ood −4.4 pts) sets, FPR 0 everywhere.

**Honest caveats.** (1) The reassurance matcher was first motivated by inspecting the 2 FPs,
so authored_heldout's FPR=0 is *mildly* optimistic; mitigating evidence — it fires on 59/72
independently-generated reassurance negatives and 0/13 authored_heldout scams, and on a **fresh,
never-inspected** 24-call honest set the hybrid model scores **FPR 0 (max 0.271** vs baseline
0.416), i.e. it rates unseen legit calls *lower*, so the mechanism generalizes rather than
memorizes. (2) It is not magic: a legit call compressed into one scam-vocabulary-dense line
can still fire — the fix holds for *realistic* multi-turn calls (the held-out signal). (3) The
bundle is versioned (`metadata.json`: `hard_signal_enabled` + cue/reassurance content hashes);
loading against drifted lexicons raises rather than silently mis-scoring. Rollback is the
frozen embedding-only bundle `models/linear_embed_only/` (flag off).

## `xlmr` backend (fine-tuned XLM-R) — STALLED, not shipping
Three end-to-end fine-tunes of `xlm-roberta-base` (CLS→mean pooling, clamped class weights,
LR warmup + grad clip, up to lr 5e-5 × 12 epochs on MPS) all **collapsed to the class prior**
(~0.64 for every input). Root cause: end-to-end fine-tuning a 270M-param transformer on a few
hundred examples is the wrong tool — anisotropic embeddings barely move without lots of data.
The *code path* is complete + tested; only the trained weights are unusable. Fix taken:
**reuse a pretrained multilingual embedder (which already understands context) + a light head**
— the `linear` backend above. Full XLM-R fine-tuning is a Phase-1 option once the corpus is
in the thousands.

## `mock` backend (deterministic keyword safety-net, no API/model)
Zero-setup fallback for the app. FPR 0 on both splits (never flags a hard negative), but
0 recall on `test` (can't match LLM-generated wording). A safety net, not a classifier.

## `llm` backend (Gemini structured classifier)
Prompt-based baseline (`gemini-2.5-pro`). Requires `GEMINI_API_KEY`; regenerate with
`--backend llm` (billed, ~13 calls/split). Ships as the always-available cloud option
alongside the offline `linear` model — the plan's "two classifiers behind one interface."

## Level 2 -- scam organization clustering

500 synthetic incidents seeded from 6 scam **script families** (5 established + 1 injected
novel "crypto giveaway" scheme), each family operating from its own reused phone number.

**Finding (why number > text here):** scam scripts across families are semantically
near-identical to `multilingual-e5` -- family embedding centroids sit at cosine **0.97-0.999**
of each other (e.g. bank-security vs police-finpol = 0.999). So HDBSCAN text clustering
collapses every family into one blob; text cannot separate scam *sub-types*. The
**phone-number co-occurrence graph is therefore the primary organization link** -- which is
exactly how investigators link real operations (reused numbers = one operation). Text
embeddings are reserved for the task where their signal *is* reliable: novelty.

**Clustering quality vs ground-truth script families** (`analytics/pipeline.cluster_quality`):

| Metric | Value |
|---|---|
| Purity | **1.00** |
| Adjusted Rand Index | **1.00** |
| Organizations recovered | 6 |

**Novelty:** the injected `crypto_giveaway_new` scheme -- distinct enough in text
(~0.11 cosine-distance from established families vs ~0.03 among them) and small -- is
correctly **flagged as a new scheme**; the 5 established families are not. Organizations are
ranked for the analyst queue by `priority = f(size, recency, recent-growth)`
(`analytics/rank.py`).

### Cluster quality under number-availability stress (2026-09-17, PLAN C8)

`python -m qorgan.eval.cluster --resamples 50` — the same 500 seeded incidents, with the
caller number removed from a seeded random share of calls (SIM rotation is the realistic
case) and, optionally, the HDBSCAN text overlay enabled. Intervals are 2.5–97.5 percentiles
over 80 % subsamples × 50 (clustering *stability*, so a full-set point can sit outside them).

| Condition | Purity [95% CI] | ARI [95% CI] | Orgs | Multi-member share | Novel flagged |
|---|---|---|---|---|---|
| numbers as seeded (shipped) | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 6 | 1.00 | 1 |
| numbers rotated for 50 % of calls | 1.000 [1.000, 1.000] | 0.323 [0.297, 0.350] | 256 | 0.50 | 29 |
| numbers rotated for 50 % + text overlay | 0.494 [0.502, 0.625] | 0.085 [0.066, 0.282] | 77 | 0.85 | 25 |
| text only (no numbers) | 0.724 [0.694, 0.777] | 0.047 [0.039, 0.112] | 171 | 0.69 | 44 |

**Caveat:** the seeds are synthetic and every family reuses one number by construction — the
first row is an upper bound, not a field result. **What the stress rows say:** (1) the number
graph is exact but *only as good as number reuse* — with half the numbers rotated, half the
incidents become singletons (ARI 0.32) while purity stays 1.0 only because singletons are
trivially pure; (2) the text overlay does not rescue it — it *lowers* purity (0.49) because
scam scripts across families are near-identical to the embedder, confirming the July finding
with a number; (3) **novelty over-fires once the graph thins**: 29 number-less singletons are
flagged as "new schemes" at the shipped 0.06 cosine threshold. The analyst's novel-scheme
callout was therefore only trustworthy while numbers were reused.

**Fix (same day, C10 / ADR D21):** novelty now requires *support* — a linkable number or
≥ 2 incidents — on top of the distance rule. Measured first: the truly novel family sits
0.103–0.108 from the nearest large organization while false candidates reach 0.090 (p95
0.065), so a distance margin alone would separate them by ~0.013 — too thin; the structural
fact is that every false candidate was a number-less singleton. After the rule:

| Condition | Novel flagged (before → after) |
|---|---|
| numbers as seeded (shipped) | 1 → **1** (the injected scheme, it has a number) |
| numbers rotated for 50 % of calls | 29 → **1** |
| numbers rotated for 50 % + text overlay | 25 → **1** |
| text only (no numbers) | 44 → **1** (a 2-call text cluster; the injected singleton is unsupported and correctly *not* flagged) |

Purity / ARI rows are unchanged (the rule touches flags, not clustering).

## Addendum (2026-07-15) — KK legit-boundary stabilization + reassurance widening

The `authored_heldout` anchor set was widened 27 -> **42 dialogues** (Phase 1A: +10 negatives
spanning telecom/delivery/other legit domains, +5 positives giving previously-starved tail
tactics — `mule_recruitment`/`secrecy`/`remote_access`/`investment_scam`/`prize_lottery` —
measurable recall). On the retrained (unmodified-lexicon) model this widening alone
surfaced a **new authored_heldout FPR of 0.042** (1/24 negatives): `real_neg_telecom_tariff_notice_mixed`
scored **0.809**, well above the 0.55 alert threshold, with `real_neg_bank_card_delivery_kk`
sitting at a **0.541 near-miss** just under it.

Root cause: `linear_train` refits one L2-regularized LogisticRegression over the whole
`[768-dim embedding | 5 cues | 1 reassurance]` vector, so the KK legit-call boundary is
**data-starved** — a handful of borderline KK negatives were carried by embedding geometry
alone, with no reassurance-feature support, and previously showed up as false positives after
a lexicon-only change reshuffled the boundary. This cycle fixes it with **data first, then a
minimal, previously-validated lexicon change**:

1. **`data/augment/kk_legit_negatives.jsonl`** — 15 hand-written, train-only KK/mixed hard
   negatives (5 bank-service, 5 gov/e-gov-service, 5 telecom-notice calls: card pickup,
   service confirmation, appointment/document-ready, tariff/SIM/maintenance notices), several
   reassuring in Kazakh ("қажеті жоқ", "талап етпейміз" style). All new text, verified to have
   zero verbatim overlap with the `authored_heldout` anchors (`tests/data/test_kk_legit_negatives_augment.py`).
   Train grew 631 -> **646** (+15, no dedup collisions); `train_augment_count` 27 -> 42.
2. **`data/lexicon/reassurance_patterns.yaml`** — added sensitive terms `төлем`, `оплата`,
   `оплату`, `платить` and KK reassurance terms `қажеті жоқ`, `қажет жоқ`, `талап етпейді`,
   `талап етпейміз` (deliberately **not** `керек емес`, which was verified to fire on a real
   scam-corpus line — inversion risk). RED->GREEN on `tests/classifier/test_reassurance.py`
   (4 new tests: tariff-phrase fire, new-term fires, inversion guards).

**Result — all gates cleared** (`QORGAN_CLASSIFIER_BACKEND=linear`, threshold unchanged at 0.55):

| Split | FPR before / after | Recall before / after |
|---|---|---|
| test | 0.000 / **0.000** | 0.953 / 0.953 |
| authored_heldout | 0.042 / **0.000** | 1.000 / 1.000 |
| ood | 0.000 / **0.000** | 0.867 / **0.889** |

| Split | False-latch before / after |
|---|---|
| authored_heldout | 0.167 / 0.167 |
| test | 0.135 / **0.115** |

**Tracked anchors (direct score, before -> after):**

| Anchor | Before | After |
|---|---|---|
| `real_neg_telecom_tariff_notice_mixed` (authored_heldout) | 0.809 | **0.281** |
| `real_neg_bank_card_delivery_kk` (authored_heldout) | 0.541 | **0.474** |
| `neg_legit_gov_service_kk_19` (test) | 0.448 | 0.484 |

`eval/threshold.py`'s max-recall-s.t.-FPR<=0.05 tuner now recommends **0.620** (fpr=0.000,
recall=1.000) on the widened set — up from 0.550 pre-fix — but the shipped default (0.55,
`config.py`) already clears the FPR bar at equal recall, so it is left unchanged.

**Honest caveat** (mirrors the caveat style above): this fix, like the original reassurance
mechanism, was **motivated by inspecting** the two heldout false/near-positives the widened
anchor set exposed (`real_neg_telecom_tariff_notice_mixed`, `real_neg_bank_card_delivery_kk`),
so the authored_heldout FPR=0.000 here is *mildly* optimistic in the same sense as the original
fix. Mitigating evidence: the new augmentation data covers 3 legit-call domains with 15
independently-written dialogues (not just the 2 inspected anchors), the reassurance-lexicon
change was validated feature-by-feature (RED->GREEN, including inversion guards) before
retraining, and `ood` FPR stayed 0 with recall *improving* (0.867 -> 0.889) — a held-out set
untouched by this change, suggesting the fix generalizes rather than overfits to the two
inspected cases.

**2026-07-15 (later same day) — cue-lexicon ASR-morphology variants.** With the KK
legit-boundary now stabilized above, added 6 hand-curated ASR-normalized variants to
`data/lexicon/hard_signal_cues.yaml` (no other lexicon/config/code changes), confirmed
against real Vosk ASR output that previously matched no cue: `safe_account` +4
("безопасной счёт", "безопасному счёту", "безопасный счет" [е-vs-ё], "қауіпсіз шотқа
аудар" [KK bare-imperative]), `otp_request` +1 ("код из сообщения"), `secrecy` +1
("ешкімге айтпа" [KK bare-imperative]). No bare-mention cues added (invariant held). 7
new RED->GREEN tests in `tests/classifier/test_features.py` (3 safe_account inflections,
1 otp ASR form + grounded-span check, 2 KK bare-imperative variants, 1 explicit
case-insensitivity check) plus 2 pre-existing-invariant guard tests (bare mentions,
reassurance-style RU/KK sentences) confirmed still all-zero. `match_cues`/
`hard_signal_features` were confirmed already case-insensitive before editing, so no
redundant lowercase variants were added.

Retrained `linear` on the widened lexicon; full suite 670/670 green (663 baseline + 7
new). All eval gates held: test FPR 0.000 (recall 0.953, unchanged), authored_heldout FPR
0.000 (recall 1.000, unchanged), ood FPR 0.000 (recall 0.889, unchanged) — tuner still
recommends 0.590 on authored_heldout (fpr=0.000, recall=1.000), threshold left at the shipped
0.55. Streaming false-latch unchanged: authored_heldout 0.167, test 0.115. Direct-scored the
motivating ASR failure string ("...переведите деньги на безопасной счёт...") — `safe_account`
now appears in tags at weight 1.0 (verbatim cue-merge). Sentinel negatives held:
`real_neg_telecom_tariff_notice_mixed` 0.284, `real_neg_bank_card_delivery_kk` 0.478
(both < 0.55, materially unchanged from the prior addendum's post-fix values). Net effect:
the hard-signal floor now fires correctly on the two documented ASR-normalized failure
modes without moving any headline metric.

## Addendum (2026-09-14) — intervals, honest names, int8-trained heads, threshold 0.59

Regenerated by the harness (`QORGAN_CLASSIFIER_BACKEND=linear python -m qorgan.eval.run
--split test --split authored_heldout --split ood`), pasted verbatim. Every rate now carries
its exact 95 % Clopper–Pearson interval; PR-AUC a seeded bootstrap. `real_heldout` is now
`authored_heldout` (hand-written, not real calls); the five negatives that were read during
feature engineering are reported as the `(inspected)` subset
(`data/anchors/inspection_ledger.yaml`).

**Shipped bundle:** `multilingual-e5-base` **int8 ONNX** embeddings (one text per run) ⊕ 6
interpretable features → calibrated LR, **threshold 0.59** (enter 0.59 / exit 0.49).
Server and browser run the same graph (ADR D17).

| Split | FPR [95% CI] | Precision | Recall [95% CI] | F1 | PR-AUC [95% CI] | N |
|---|---|---|---|---|---|---|
| test | 0.000 [0.000, 0.068] | 1.000 | 0.938 [0.848, 0.983] | 0.968 | 0.998 [0.994, 1.000] | 116 |
| authored_heldout | 0.000 [0.000, 0.142] | 1.000 | 1.000 [0.815, 1.000] | 1.000 | 1.000 [1.000, 1.000] | 42 |
| authored_heldout (clean) | 0.000 [0.000, 0.176] | 1.000 | 1.000 [0.815, 1.000] | 1.000 | 1.000 [1.000, 1.000] | 37 |
| authored_heldout (inspected) | 0.000 [0.000, 0.522] | 0.000 | 0.000 [-] | 0.000 | 0.000 [-] | 5 |

`Recommended alert threshold (fpr<=0.05 on authored_heldout): 0.660  -> fpr=0.000 recall=1.000`

### What changed vs the July numbers (fp32 heads @ 0.55: test 0.000/0.953 · authored 0.000/1.000 · ood 0.000/0.889)
- FPR is 0 on all splits at 0.59; the single ood negative that crossed 0.55 with int8 heads sat at 0.551.
- Recall cost of int8-trained heads: test −1.5 pts (0.938), ood −4.5 pts (0.844); authored unchanged (1.000).
- `e5-small` was measured and rejected (authored recall 0.778 at 0.55; margins overlap) — ADR D17.

### Inspection ledger (what was looked at, when, why)
| id | inspected | why |
|---|---|---|
| `real_neg_bank_fraud_alert_ru` | 2026-07-13 | FP @0.990 on the embedding-only model; motivated the reassurance feature + data |
| `real_neg_telecom_tariff_ru` | 2026-07-13 | FP @0.942, same investigation |
| `real_neg_bank_card_ready_ru` | 2026-07-13 | near-miss @0.647 examined alongside |
| `real_neg_telecom_tariff_notice_mixed` | 2026-07-15 | KK/mixed FP @0.809 after widening; retrain-gate sentinel |
| `real_neg_bank_card_delivery_kk` | 2026-07-15 | borderline @0.541 during the KK work; retrain-gate sentinel |

The clean authored subset has **19 negatives** → an FPR of 0.000 there bounds the true rate
only at **0.176** (95 %). Bounding ≤ 5 % needs ≥ 59 clean negatives; the locked real-call
set (PLAN_2026-09 A3) is the only way to tighten this.

### On-device parity (`npm test`) — runtime numbers superseded by the 2026-09-20 addendum (ADR D32)
The JS port (`site/core/`) reproduces Python on 28 golden cases: risk to < 1e-6 with the same
embeddings, identical tags, spans, RU/KK explanations and live-meter trajectories. Across
ONNX Runtime implementations the int8 graph itself drifts (cosine ~0.994 mean / ~0.98 min,
Node vs Python); with int8-trained heads that is decision-safe: **0 flips / 28, |Δrisk| ≤
0.055, tag-set Jaccard 0.977** (`tests_js/integration/embedding.test.mjs`).

Static (calibrated) int8 quantisation was measured as the candidate fix and rejected
(2026-09-17, ADR D18; `scripts/quantize_embedder.py`, 88 transcripts):

| graph | cosine vs fp32 | Node ORT 1.21 vs Python ORT 1.27 |
|---|---|---|
| dynamic int8 (shipped) | 0.992 mean / 0.981 min | 0.985 mean / 0.970 min |
| static int8 (MinMax-MA, 128 calib.) | 0.944 mean / 0.924 min | 0.992 mean / 0.971 min |

It loses 5 pts of fidelity and leaves the cross-runtime floor where it was, so the residual
is runtime kernel differences, not activation scales. The decision-level gate above is the
guarantee; it is not bit-level parity and this report does not claim it.

### Adversarial paraphrases — "assume the adversary has the lexicon" (2026-09-17, PLAN A9, ADR D22)

`python scripts/paraphrase_adversarial.py` rewrote every scam in `test` + `ood` (109) with
`gemini-2.5-flash` under the constraint that none of the 35 hard-signal cue phrases
survives; compliance was verified locally with the classifier's own matcher (0 / 109 cue
hits; the sources had hits on 15 / 109) and a Kazakh-letter check kept every paraphrase in
its source language (0 switches after retries; 0 failures). Same turn count on average
(7.2). `QORGAN_CLASSIFIER_BACKEND=linear python -m qorgan.eval.adversarial` (paired, @0.59):

| Set | N | Recall [95% CI] |
|---|---|---|
| source scams (test + ood) | 109 | 0.899 [0.827, 0.949] |
| adversarial (lexicon-free paraphrases) | 109 | 0.917 [0.849, 0.962] |
| &nbsp;&nbsp;kk · source | 34 | 1.000 [0.897, 1.000] |
| &nbsp;&nbsp;kk · adversarial | 34 | 0.971 [0.847, 0.999] |
| &nbsp;&nbsp;mixed · source | 38 | 0.816 [0.657, 0.923] |
| &nbsp;&nbsp;mixed · adversarial | 38 | 0.868 [0.719, 0.956] |
| &nbsp;&nbsp;ru · source | 37 | 0.892 [0.746, 0.970] |
| &nbsp;&nbsp;ru · adversarial | 37 | 0.919 [0.781, 0.983] |

Recall drop: **−1.8 points** (3 scams flip to clear, 5 flip to scam) — within the plan's
15-point gate, so A10 stays a "could". **Reading:** recall does not rest on the lexicon;
the embedding head carries it, and the cue features are an explainability / precision
instrument (grounded highlights, the live meter's hard-signal floors). An attacker who
scripts around the published cue list gains nothing on the single-shot verdict.
**Caveats, stated:** (1) the paraphrases are LLM-written by the same model family that
wrote the training corpus — a shared "synthetic style" may keep them easy; (2) the
constraint targets the cue *phrases*, not the embedding's notion of a scam — an adversary
who rewrites the call to *sound legitimate* (e.g. mimicking institutional reassurance) is
a different, harder attack (follow-up **A9b**); (3) the live meter's hard-signal floors
cannot fire on these calls — measured with `eval.stream --split adversarial`: alert-hit
**0.972 [0.922, 0.994]** (test: 0.969), median turns-to-alert 2 (same), P90 4 vs 3 — the
meter still latches from the embedding signal, with a one-turn delay in the tail.

### The legit-sounding adversary (2026-09-18, PLAN A9b) — the gate FAILS

`python scripts/paraphrase_adversarial.py --style legit_sounding` rewrote the same 109 test +
ood scams to avoid every cue *and* to sound like the institution — calm procedural register,
the reassurances a real bank gives («we will never ask for your password», «a routine
security procedure») — while still pursuing the scam goal (109 / 109 produced, 0 cue hits, 0
language switches; the reassurance feature fires on **36 / 109** of them).
`python -m qorgan.eval.adversarial --split adversarial_legit` (paired, @0.59):

| Set | N | Recall [95% CI] |
|---|---|---|
| source scams (test + ood) | 109 | 0.899 [0.827, 0.949] |
| adversarial_legit (lexicon-free + legit-sounding register) | 109 | **0.651 [0.554, 0.740]** |
| &nbsp;&nbsp;kk · source / adversarial_legit | 34 | 1.000 → 0.735 |
| &nbsp;&nbsp;mixed · source / adversarial_legit | 38 | 0.816 → **0.500** |
| &nbsp;&nbsp;ru · source / adversarial_legit | 37 | 0.892 → 0.730 |

Recall drop: **24.8 points** (33 scams flip to clear, 6 flip to scam) — **fails the 15-point
gate.** Reading: the model's recall does not depend on the cue lexicon (A9), but it *does*
depend on register — a scam that talks like a bank, including the reassurance sentence that
the July fix taught the model to trust, is missed one time in three. The reassurance feature
is a documented anti-fraud practice and stays; the response is data, not a feature tweak
(the sprint lesson): legit-style paraphrases of **train** scams
(`scripts/augment_legit_style_scams.py`, 268 / 268 produced, 0 failures, train-only), then an
**augmentation-dose sweep** against every FPR gate (`models/linear_knee_*`, seeded subsets):

| Train augmentation | test FPR / recall | authored FPR / recall | ood FPR / recall | adversarial_legit | cue-free | streaming (authored false-latch / alert-hit) |
|---|---|---|---|---|---|---|
| none (shipped 2026-09-14) | 0.000 / 0.938 | 0.000 / 1.000 | 0.000 / 0.844 | 0.651 | 0.917 | 2/24 / 16/18 |
| 40 legit-style scams | 0.000 / 0.953 | **0.083** / 1.000 | 0.000 / 0.844 | 0.972 | 0.917 | – |
| 268 legit-style scams (all) | 0.000 / 0.938 | **0.208** / 0.833 | 0.000 / 0.844 | 0.991 | 0.908 | – |
| 40 scams + 45 institutional-register legit negatives | 0.000 / 0.953 | 0.000 / 1.000 | 0.000 / 0.844 | 0.890 | 0.927 | **3/24** / **15/18** |
| **20 scams + 45 institutional-register legit negatives (adopted)** | 0.000 / **0.953** | **0.000 / 1.000** | 0.000 / 0.844 | **0.826** | **0.927** | **3/24** / 16/18 |

Scam-side data alone teaches the model that the institutional register can be a scam and
brings the July false positives back (the two RU bank/telecom anchors at 0.650 / 0.619
with 40 examples; five of 24 negatives fire with all 268). Pairing them with the 45 unused
institutional-register **legit** negatives from the July generation
(`data/synthetic/reassurance_negatives.jsonl`, now all 72 committed to
`data/augment/reassurance_negatives.jsonl`) restores every single-shot FPR gate — the model
must then rely on the *request*, not the register. **The streaming gate is missed by one
hairline call at every dose:** `real_neg_bank_fraud_alert_ru` latches transiently at turn 2
(meter 60.3 vs enter 59; the shipped model already reads its first two windows at risk
0.862 / 0.713, meter 57.2, and only drops to 0.31 when the reassurance sentence arrives at
turn 3). That is the early-window noise PLAN A6 exists for, not a change in what the model
believes about the call; false-latch 3/24 [0.027, 0.324] vs 2/24 [0.010, 0.270] are
statistically indistinguishable, but the gate says ≤ 2/24 and it is stated as missed.
**Adopted in the working tree (2026-09-18): 20 scams + 45 negatives — retrained
`models/linear`, rollback `models/linear_prev` (`QORGAN_LINEAR_MODEL_DIR=models/linear_prev`
puts the shipped heads back):**

| Set | before | after |
|---|---|---|
| test FPR / recall | 0.000 [0, 0.068] / 0.938 | 0.000 [0, 0.068] / **0.953** [0.869, 0.990] |
| authored_heldout FPR / recall | 0.000 [0, 0.142] / 1.000 | 0.000 [0, 0.142] / 1.000 |
| ood FPR / recall | 0.000 [0, 0.048] / 0.844 | 0.000 [0, 0.048] / 0.844 |
| adversarial (cue-free) recall | 0.917 | **0.927** [0.860, 0.968] |
| adversarial_legit recall | 0.651 | **0.826** [0.741, 0.892] (drop vs source 8.3 pts) |
| streaming authored false-latch / alert-hit | 2/24 / 16/18 | **3/24** / 16/18 (one hairline call, above) |
| cross-runtime gate (200 transcripts) | 3 flips (1.5 %) | 2 flips (1.0 %) |

The inspection-ledger caveat applies twice over: the RU anchors were read in July *and* the
knee was chosen against `authored_heldout`; only the locked real set (A3) can confirm it.
Caveat as for A9: an LLM paraphrase from the same generator family as the corpus; real
scammers may be cruder.

### The cross-runtime gate at 200 cases (2026-09-18) — the "0 flips / 28" claim withdrawn; itself superseded by the browser gate (2026-09-20, ADR D32)

The retrain flipped one golden case in Node (`real_scam_telecom_verify_ru` 0.672 → 0.588),
so the gate was given the statistical power PLAN B3 asked for: a 200-transcript set with
Python's verdicts (`tests_js/fixtures/runtime_gate.json`; `tests_js/tools/runtime_gate_probe.mjs`
runs it against any heads). On that set the **shipped** model already flipped **3 / 200**
(1.5 %, |Δrisk| max 0.248, p95 0.108) and the retrained one flips **2 / 200** (1.0 %, max
0.270, p95 0.096, median 0.004) — the same runtime residual, invisible at 28 cases. The gate
now asserts the plan's tolerance (≥ 99 % same decision) plus p95 |Δrisk| < 0.15 and max
< 0.30; the retrained heads pass it, the shipped ones would not have. ADR D17's "decision-safe:
0 flips / 28" is superseded by these numbers.

### ASR realism on the on-device recogniser (2026-09-18, PLAN A10 partial, B9)

The three demo scenes were spoken (macOS TTS) into the on-device recogniser (`site/core/asr.js`,
Vosklet, small KK + RU models) and scored by the on-device classifier + meter in real time:

| Scene | Voted language · confidence | Meter | Python on the same ASR text |
|---|---|---|---|
| Bank-security scam (RU) | ru 0.93 / 0.91 (2 utterances) | **81 / 100 critical** | risk 1.000 |
| Real bank call (RU, hard negative) | ru 0.97 | **14 / 100 low** | risk 0.396, meter 19.2 |
| Bank-security scam (KK) | kk 0.96 | **61 / 100 high** | risk 0.997 |

**Finding:** Vosk glued the negation on the legit call — «называть **ненужно**» — and the
reassurance feature (`classifier/reassurance.py`) no longer fired; the call stayed clear on
embedding margin alone (0.396 vs 0.59). The matcher now joins the tokens of every term with
optional whitespace (Python and the JS port alike), which restores the feature on glued or
double-spaced ASR text. Measured before shipping it: the feature value changed on **0 of
1,146** dialogues across train / val / test / authored / ood / adversarial / augment, so the
trained heads are untouched and no retrain or gate re-run was needed. Not addressed (still
A10): ASR word substitutions («коды» → «года», «SMS» → «самая с») — those need training-time
ASR-style augmentation, not a matcher tweak.

### Streaming (`python -m qorgan.eval.stream --split test --split authored_heldout --backend linear`)
| Split | False-Latch Rate [95% CI] | Alert-Hit Rate [95% CI] | Median Turns | P90 Turns | N+ | N- |
|---|---|---|---|---|---|---|
| test | 0.096 [0.032, 0.210] | 0.969 [0.892, 0.996] | 2.000 | 3.000 | 64 | 52 |
| authored_heldout | 0.083 [0.010, 0.270] | 0.889 [0.653, 0.986] | 2.000 | 3.000 | 18 | 24 |

False-latch (the live FPR analogue) improved with the 0.59 / 0.49 hysteresis pair: authored
2/24 (was 4/24 in July), test 0.096 (was 0.115). Two authored scams never latch in
streaming (alert-hit 0.889) — the meter's min-turns / damping work (PLAN A6) must be gated on
both numbers, not false-latch alone.

**Retrained heads (2026-09-18, 20 legit-style scams + 45 institutional-register negatives,
see the A9b section):**

| Split | False-Latch Rate [95% CI] | Alert-Hit Rate [95% CI] | Median Turns | P90 Turns | N+ | N- |
|---|---|---|---|---|---|---|
| test | 0.115 [0.044, 0.234] | 0.984 [0.916, 1.000] | 2.000 | 3.000 | 64 | 52 |
| authored_heldout | 0.125 [0.027, 0.324] | 0.889 [0.653, 0.986] | 2.000 | 3.000 | 18 | 24 |

Versus the shipped heads: test false-latch 5/52 → 6/52, test alert-hit 62/64 → 63/64,
authored false-latch 2/24 → **3/24** (the hairline call analysed above), authored alert-hit
16/18 unchanged, time-to-alert unchanged. The plan's G4 gate (≤ 2/24) was therefore missed by
one call whose second window the shipped model already scores at risk 0.71 — which is the
meter's early-window problem, addressed the next day:

**Meter A6 (2026-09-19, ADR D29): the latch arms from the third utterance unless a hard
signal fired.** Every turn-2 false latch was a legitimate opener that reads as a scam until
context arrives; the rule was chosen with `python -m qorgan.eval.meter_sweep` (per-turn
traces cached once, variants replayed through the production meter: `min_turns_to_arm ∈
{1,2,3}` × damping `∈ {off,2,3}`, both heads, selected on `test`, checked on `authored`).
Same retrained heads, `python -m qorgan.eval.stream`:

| Split | False-Latch Rate [95% CI] | Alert-Hit Rate [95% CI] | Median Turns | P90 Turns | N+ | N- |
|---|---|---|---|---|---|---|
| test | 0.115 [0.044, 0.234] | 0.984 [0.916, 1.000] | 3.000 | 3.000 | 64 | 52 |
| authored_heldout | 0.083 [0.010, 0.270] | 0.889 [0.653, 0.986] | 3.000 | 3.000 | 18 | 24 |

Authored false-latch back to **2/24** with every alert kept (16/18 authored, 63/64 test);
median turns-to-alert 2 → 3, exactly the "+1" the A6 gate allows; P90 unchanged. Short-window
damping (`alpha_up × min(1, turn/N)`) would take test false-latch to 3/52 at N = 2 but
drops a 3-turn scam whose meter can no longer converge — it ships as an off-by-default knob
(`QORGAN_METER_SHORT_WINDOW_TURNS`). With this meter the retrained heads meet every stated
gate; the demo scam scene latches on its third line (87) instead of its second.

### Per-tactic decision thresholds (2026-09-20, ADR D30; replaces PLAN A11's data top-up)

A11 assumed the weak per-tactic rows were a data problem. They were an operating-point
problem: the tactic head **over-predicted** — 2.86 predicted tags per test dialogue against
1.91 true, `otp_request` precision 0.33 at recall 0.90, `safe_account` 0.26, authored
`verification_ploy` 0.08. The fix is one threshold per tactic (`weights.json::tactic_head.thresholds`,
Python and JS decode 1:1), tuned by max-F1 over `0.50–0.70` on **out-of-fold train
probabilities + val** (seeded 5-fold of the same head; a tactic with < 8 tuning positives
keeps 0.5). The risk head, its calibration and the 0.59 alert threshold are untouched
(`risk_clf.joblib` byte-identical), so every FPR / recall / streaming number above stands.

Three tuning sets were measured end-to-end (through `predict.score`, i.e. after the verbatim
cue merge) — flat 0.5 (before), `val` alone (the first cut), and out-of-fold train + val
(adopted):

| Split | Thresholds | micro P | micro R | micro F1 | macro F1 | pred tags / dialogue | true tags / dialogue |
|---|---|---|---|---|---|---|---|
| test | flat 0.5 | 0.611 | 0.914 | 0.733 | 0.705 | 2.86 | 1.91 |
| test | val only | 0.665 | 0.815 | 0.733 | 0.718 | 2.34 | 1.91 |
| test | **oof train + val** | 0.757 | 0.784 | **0.770** | **0.737** | **1.98** | 1.91 |
| authored_heldout | flat 0.5 | 0.494 | 0.702 | 0.580 | 0.628 | 1.93 | 1.36 |
| authored_heldout | val only | 0.597 | 0.649 | 0.622 | 0.626 | 1.48 | 1.36 |
| authored_heldout | **oof train + val** | 0.607 | 0.596 | 0.602 | 0.618 | 1.33 | 1.36 |
| ood | flat 0.5 | 0.259 | 0.844 | 0.396 | 0.520 | 1.23 | 0.38 |
| ood | val only | 0.340 | 0.756 | 0.469 | 0.552 | 0.83 | 0.38 |
| ood | **oof train + val** | 0.388 | 0.689 | **0.496** | **0.593** | **0.67** | 0.38 |

`val` alone (8–31 positives per tactic) moved six tactics and half of those moves reversed on
test; with the out-of-fold rows (14–250 positives per tactic) the tuner has support for every
tactic and the moves hold on test and ood. Authored (42 dialogues, 1–8 positives per tactic)
is within noise either way — micro up, macro down 0.01. Per tactic, flat → adopted:

| tactic | cut | test n+ | test P/R/F1 | authored n+ | authored P/R/F1 | ood n+ | ood P/R/F1 |
|---|---|---|---|---|---|---|---|
| `impersonation_bank` | 0.55 | 30 | 0.77/0.90/0.83 → 0.87/0.90/0.89 | 3 | 0.43/1.00/0.60 → 0.50/1.00/0.67 | 3 | 0.11/0.67/0.19 → 0.14/0.67/0.24 |
| `impersonation_gov_police` | 0.60 | 7 | 0.33/1.00/0.50 → 0.55/0.86/0.67 | 3 | 1.00/0.33/0.50 (=) | 3 | 0.14/0.33/0.20 → 0.20/0.33/0.25 |
| `impersonation_telecom_delivery` | 0.65 | 14 | 0.69/0.79/0.73 → 1.00/0.71/0.83 | 3 | 0.00/0.00/0.00 (=) | 3 | 0.50/1.00/0.67 → 0.67/0.67/0.67 |
| `urgency` | 0.60 | 43 | 0.82/0.86/0.84 → 0.91/0.67/0.77 | 8 | 0.22/0.25/0.24 → 0.25/0.12/0.17 | 3 | 0.11/0.67/0.19 → 0.25/0.67/0.36 |
| `fear_threat` | 0.55 | 39 | 0.83/0.97/0.89 → 0.85/0.90/0.88 | 4 | 0.50/1.00/0.67 → 0.43/0.75/0.55 | 3 | 0.08/0.67/0.14 → 0.12/0.67/0.20 |
| `secrecy` | 0.60 | 4 | 0.43/0.75/0.55 → 0.50/0.75/0.60 | 7 | 1.00/0.86/0.92 → 1.00/0.71/0.83 | 3 | 0.60/1.00/0.75 → 1.00/0.67/0.80 |
| `otp_request` | 0.60 | 10 | 0.33/0.90/0.49 → 0.57/0.80/0.67 | 4 | 0.67/1.00/0.80 → 0.80/1.00/0.89 | 3 | 0.40/0.67/0.50 → 1.00/0.67/0.80 |
| `credentials_request` | 0.60 | 16 | 0.41/0.94/0.57 → 0.52/0.69/0.59 | 7 | 0.50/0.57/0.53 → 0.60/0.43/0.50 | 3 | 0.17/0.67/0.27 → 0.50/0.67/0.57 |
| `safe_account` | 0.60 | 6 | 0.26/0.83/0.40 → 0.56/0.83/0.67 | 1 | 1.00/1.00/1.00 (=) | 3 | 0.33/1.00/0.50 → 0.75/1.00/0.86 |
| `payment_redirect` | 0.55 | 6 | 0.60/1.00/0.75 → 0.83/0.83/0.83 | 5 | 0.67/0.80/0.73 → 0.60/0.60/0.60 | 3 | 0.43/1.00/0.60 → 0.67/0.67/0.67 |
| `remote_access` | 0.60 | 4 | 1.00/1.00/1.00 → 1.00/0.75/0.86 | 2 | 1.00/1.00/1.00 (=) | 3 | 1.00/1.00/1.00 (=) |
| `prize_lottery` | 0.55 | 5 | 0.83/1.00/0.91 → 1.00/0.80/0.89 | 4 | 0.75/0.75/0.75 → 0.67/0.50/0.57 | 3 | 1.00/1.00/1.00 (=) |
| `investment_scam` | 0.50 | 7 | 0.88/1.00/0.93 (=) | 3 | 0.75/1.00/0.86 (=) | 3 | 0.75/1.00/0.86 (=) |
| `mule_recruitment` | 0.60 | 2 | 0.29/1.00/0.44 → 0.20/0.50/0.29 | 2 | 0.50/1.00/0.67 → 0.67/1.00/0.80 | 3 | 0.50/1.00/0.67 → 0.50/0.33/0.40 |
| `verification_ploy` | 0.60 | 29 | 0.61/0.93/0.74 → 0.69/0.69/0.69 | 1 | 0.08/1.00/0.15 → 0.20/1.00/0.33 | 3 | 0.16/1.00/0.27 → 0.17/0.33/0.22 |

Read honestly: the request-cue tactics (`otp_request`, `safe_account`, `credentials_request`,
`secrecy`, `remote_access`) gain precision without losing the cases the verbatim cue merge
already rescues; `impersonation_*` gain on every split. The costs are `urgency` (test recall
0.86 → 0.67) and `verification_ploy` (0.93 → 0.69) — the 208 / 160 out-of-fold positives put
their optimum at 0.60, test's 43 / 29 at 0.50; the larger sample decides and the disagreement
is left on record. `mule_recruitment` (2–3 positives per eval split) is noise in both directions.
The n+ ≤ 3 cells of ood and authored are individual dialogues, not rates. Only real calls (A3)
can settle the thresholds; they are one retrain away (`python -m qorgan.classifier.linear_train`
re-tunes them from whatever `val` holds).

Cross-runtime gate with the tuned cuts (`tests_js/integration/embedding.test.mjs`, 200 cases):
**2 / 200 flips** (the same two borderline transcripts as before — the risk head is unchanged),
p95 |Δrisk| 0.097, max 0.266, **tag-set Jaccard 0.935** (gate ≥ 0.9). Tighter cuts sit
closer to the probability mass, so tags are where runtime drift shows first; the 200-case
Jaccard under the flat cut was not recorded when the gate was introduced (only the 28-case
golden value, 0.977), so the Jaccard cost of the new cuts is not isolated here.
`eval.run` after the change: test 0.000 / 0.953 · authored 0.000 / 1.000 · ood 0.000 / 0.844 —
identical, as the byte-identical risk head requires.

## Addendum (2026-09-20) — the runtime the numbers describe (ADR D32), and the recogniser's register (A10, ADR D31)

> **Every table above this line was computed on the server's ONNX Runtime 1.27.** The
> device runs a different build, and that alone changes single calls near the 0.59 threshold
> (below). From here on the runtime is pinned (`onnxruntime==1.21.*`, bit-identical to
> onnxruntime-node) and the device gate is measured in a real browser. The "0.000" FPR rows
> above were true on that server runtime; on the pinned one the same heads read 1/52 on test.

### What the device actually computes (200-transcript gate set, cosine of the int8 embeddings)

| pair | cosine mean | cosine min | note |
|---|---|---|---|
| Python ORT 1.27 vs onnxruntime-node 1.21 | 0.980 | 0.958 | what D17 / D18 / D28 called the "runtime residual" |
| Python ORT 1.21 vs onnxruntime-node 1.21 | **1.00000** | 1.00000 | same version → bit-identical; the residual was the version |
| browser WASM (onnxruntime-web 1.22.0-dev) vs Python 1.21 | 0.982 | 0.949 | the real device drift; 1.27 is equidistant (0.980 / 0.954) |
| browser **WebGPU** (q8) vs Python | **0.78** | 0.70 | broken — quantised ops not on the GPU provider; also slower (0.7 s vs 0.2 s) → WASM forced |

**Browser gate** (`tests_js/integration/browser_gate.test.mjs`, the browser's own embeddings
captured by `npm run gate:browser`; Python's verdicts vs the JS scorer on those vectors):

| heads | flips / 200 | p95 \|Δrisk\| | max \|Δrisk\| | tag Jaccard |
|---|---|---|---|---|
| previously shipped recipe (D30) | 5 | 0.123 | 0.404 | — |
| shipped now (D31 + D32) | 5 | 0.102 | 0.515 | 0.920 |

Flips (shipped): `real_scam_delivery_customs_ru` py 0.756 / browser 0.571 ·
`ood_impersonation_gov_police_mixed_0` 0.766 / 0.565 · `ood_mule_recruitment_kk_0` 0.553 /
0.736 · `ood_neg_legit_bank_call_mixed_3` **0.669 / 0.154** (the server's one ood false
positive is not one on the device) · `ood_mule_recruitment_mixed_0` 0.529 / 0.619. The gate
is asserted at the measured level (≥ 97 % same decision, p95 < 0.15, Jaccard ≥ 0.9); PLAN
B3's ≥ 99 % was never met on the device.

### The recogniser's register (`python -m qorgan.eval.asr_realism [--drop-latin]`)

Every dialogue scored clean and through `data.asr_style` (lowercase · punctuation → space ·
numerals → words in the dialogue's language · Latin tokens kept, or removed with
`--drop-latin`), paired by id. Format only — misrecognition is not modelled — so floors.

**Previously shipped heads, server runtime 1.27 (the first measurement):**

| Split | Text | FPR [95% CI] | Recall [95% CI] | Flips → alert / → clear | Cues kept | Reassurance kept | N+ | N- |
|---|---|---|---|---|---|---|---|---|
| test | clean | 0.000 [0.000, 0.068] | 0.953 [0.869, 0.990] | - | 6 | 5 | 64 | 52 |
| test | styled | 0.038 [0.005, 0.132] | 0.969 [0.892, 0.996] | 2 / 0 | 6/6 | 5/5 | 64 | 52 |
| authored_heldout | clean | 0.000 [0.000, 0.142] | 1.000 [0.815, 1.000] | - | 13 | 9 | 18 | 24 |
| authored_heldout | styled | 0.000 [0.000, 0.142] | 0.944 [0.727, 0.999] | 0 / 1 | 12/13 | 9/9 | 18 | 24 |
| ood | clean | 0.000 [0.000, 0.048] | 0.844 [0.705, 0.935] | - | 9 | 6 | 45 | 75 |
| ood | styled | 0.013 [0.000, 0.072] | 0.911 [0.788, 0.975] | 1 / 1 | 9/9 | 6/6 | 45 | 75 |

**Previously shipped heads, pinned runtime 1.21:**

| Split | FPR [95% CI] | Precision | Recall [95% CI] | F1 | PR-AUC [95% CI] | N |
|---|---|---|---|---|---|---|
| test | 0.019 [0.000, 0.103] | 0.984 | 0.953 [0.869, 0.990] | 0.968 | 0.997 [0.991, 1.000] | 116 |
| authored_heldout | 0.000 [0.000, 0.142] | 1.000 | 0.944 [0.727, 0.999] | 0.971 | 1.000 [1.000, 1.000] | 42 |
| authored_heldout (clean) | 0.000 [0.000, 0.176] | 1.000 | 0.944 [0.727, 0.999] | 0.971 | 1.000 [1.000, 1.000] | 37 |
| authored_heldout (inspected) | 0.000 [0.000, 0.522] | 0.000 | 0.000 [-] | 0.000 | 0.000 [-] | 5 |
| ood | 0.000 [0.000, 0.048] | 1.000 | 0.867 [0.732, 0.949] | 0.929 | 0.992 [0.976, 1.000] | 120 |

| Split | Text | FPR [95% CI] | Recall [95% CI] | Flips → alert / → clear | Cues kept | Reassurance kept | N+ | N- |
|---|---|---|---|---|---|---|---|---|
| test | clean | 0.019 [0.000, 0.103] | 0.953 [0.869, 0.990] | - | 6 | 5 | 64 | 52 |
| test | styled | 0.096 [0.032, 0.210] | 0.953 [0.869, 0.990] | 4 / 0 | 6/6 | 5/5 | 64 | 52 |
| authored_heldout | clean | 0.000 [0.000, 0.142] | 0.944 [0.727, 0.999] | - | 13 | 9 | 18 | 24 |
| authored_heldout | styled | 0.042 [0.001, 0.211] | 1.000 [0.815, 1.000] | 1 / 0 | 12/13 | 9/9 | 18 | 24 |
| ood | clean | 0.000 [0.000, 0.048] | 0.867 [0.732, 0.949] | - | 9 | 6 | 45 | 75 |
| ood | styled | 0.000 [0.000, 0.048] | 0.889 [0.759, 0.963] | 0 / 3 | 9/9 | 6/6 | 45 | 75 |

| Split | False-Latch Rate [95% CI] | Alert-Hit Rate [95% CI] | Median Turns | P90 Turns | N+ | N- |
|---|---|---|---|---|---|---|
| test | 0.096 [0.032, 0.210] | 0.969 [0.892, 0.996] | 3.000 | 3.000 | 64 | 52 |
| authored_heldout | 0.083 [0.010, 0.270] | 0.889 [0.653, 0.986] | 3.000 | 3.000 | 18 | 24 |

**Augmentation dose** (ASR-styled copies of a seeded share of train, heads refitted on
memoised 1.27 embeddings through the production `train_linear`; dose 0 reproduces the first
table above):

| dose (styled copies) | test clean FPR / R | test styled | authored clean | authored styled | ood clean | ood styled | adversarial R | adversarial_legit R |
|---|---|---|---|---|---|---|---|---|
| 0.00 (0) | 0/52=0.000 / 61/64=0.953 | 2/52=0.038 / 62/64=0.969 | 0/24=0.000 / 18/18=1.000 | 0/24=0.000 / 17/18=0.944 | 0/75=0.000 / 38/45=0.844 | 1/75=0.013 / 41/45=0.911 | 101/109=0.927 | 90/109=0.826 |
| 0.25 (182) | 0/52=0.000 / 62/64=0.969 | 0/52=0.000 / 62/64=0.969 | 0/24=0.000 / 17/18=0.944 | 0/24=0.000 / 16/18=0.889 | 1/75=0.013 / 38/45=0.844 | 0/75=0.000 / 41/45=0.911 | 102/109=0.936 | 88/109=0.807 |
| 0.50 (351) | 0/52=0.000 / 61/64=0.953 | 0/52=0.000 / 61/64=0.953 | 0/24=0.000 / 17/18=0.944 | 0/24=0.000 / 16/18=0.889 | 0/75=0.000 / 39/45=0.867 | 0/75=0.000 / 41/45=0.911 | 101/109=0.927 | 92/109=0.844 |
| 1.00 (711) | 0/52=0.000 / 62/64=0.969 | 0/52=0.000 / 63/64=0.984 | 0/24=0.000 / 17/18=0.944 | 0/24=0.000 / 16/18=0.889 | 0/75=0.000 / 39/45=0.867 | 0/75=0.000 / 42/45=0.933 | 101/109=0.927 | 95/109=0.872 |

**Weighting of the copies, with the browser gate in the loop** (pinned runtime; ‖w‖ is the
risk head's coefficient norm — doubling the rows unweighted raises it 20 → 27 because C is
fixed; pair-weighting restores it but halves the clean register's evidence too):

| scheme (clean_w / styled_w) | ‖w‖ | test clean FPR·R | test styled | authored clean | authored styled | ood clean | ood styled | adv R | adv_legit R | gate flips / p95 / max |
|---|---|---|---|---|---|---|---|---|---|---|
| D30-like clean only (1.0/0.0) | 20.2 | 1/52 · 61/64 | 5/52 · 61/64 | 0/24 · 18/18 | 1/24 · 18/18 | 1/75 · 38/45 | 0/75 · 40/45 | 101/109=0.927 | 87/109=0.798 | 5/200 · 0.123 · 0.404 |
| unweighted 1/1 (1.0/1.0) | 26.7 | 1/52 · 61/64 | 0/52 · 61/64 | 0/24 · 17/18 | 1/24 · 18/18 | 1/75 · 39/45 | 0/75 · 39/45 | 102/109=0.936 | 95/109=0.872 | 5/200 · 0.103 · 0.515 |
| pair 0.5/0.5 (0.5/0.5) | 20.2 | 1/52 · 61/64 | 0/52 · 61/64 | 0/24 · 16/18 | 1/24 · 18/18 | 1/75 · 38/45 | 0/75 · 38/45 | 101/109=0.927 | 83/109=0.761 | 5/200 · 0.115 · 0.464 |
| 1/0.5 (1.0/0.5) | 23.9 | 1/52 · 61/64 | 0/52 · 61/64 | 0/24 · 17/18 | 1/24 · 18/18 | 1/75 · 39/45 | 0/75 · 39/45 | 101/109=0.927 | 86/109=0.789 | 6/200 · 0.117 · 0.458 |
| 1/0.25 (1.0/0.25) | 22.0 | 1/52 · 61/64 | 0/52 · 61/64 | 0/24 · 17/18 | 1/24 · 18/18 | 1/75 · 39/45 | 0/75 · 39/45 | 101/109=0.927 | 84/109=0.771 | 5/200 · 0.118 · 0.390 |
| 0.75/0.75 (0.75/0.75) | 23.9 | 1/52 · 61/64 | 0/52 · 61/64 | 0/24 · 17/18 | 1/24 · 18/18 | 1/75 · 39/45 | 0/75 · 39/45 | 101/109=0.927 | 89/109=0.817 | 5/200 · 0.116 · 0.498 |

Every scheme flips 5–6 / 200 on the browser, so the gate does not separate them; unweighted
1.0 is the best on every eval number and ships. Pair weights (0.5 + 0.5) are rejected on the
recall they cost.

### Shipped (ADR D31 + D32): styled copies of every train row, hyphen-free KK cues, ORT 1.21, WASM only

`python -m qorgan.eval.run --split test --split authored_heldout --split ood --backend linear`

| Split | FPR [95% CI] | Precision | Recall [95% CI] | F1 | PR-AUC [95% CI] | N |
|---|---|---|---|---|---|---|
| test | 0.019 [0.000, 0.103] | 0.984 | 0.953 [0.869, 0.990] | 0.968 | 0.999 [0.995, 1.000] | 116 |
| authored_heldout | 0.000 [0.000, 0.142] | 1.000 | 0.944 [0.727, 0.999] | 0.971 | 1.000 [1.000, 1.000] | 42 |
| authored_heldout (clean) | 0.000 [0.000, 0.176] | 1.000 | 0.944 [0.727, 0.999] | 0.971 | 1.000 [1.000, 1.000] | 37 |
| authored_heldout (inspected) | 0.000 [0.000, 0.522] | 0.000 | 0.000 [-] | 0.000 | 0.000 [-] | 5 |
| ood | 0.013 [0.000, 0.072] | 0.975 | 0.867 [0.732, 0.949] | 0.918 | 0.989 [0.969, 1.000] | 120 |

`python -m qorgan.eval.asr_realism`

| Split | Text | FPR [95% CI] | Recall [95% CI] | Flips → alert / → clear | Cues kept | Reassurance kept | N+ | N- |
|---|---|---|---|---|---|---|---|---|
| test | clean | 0.019 [0.000, 0.103] | 0.953 [0.869, 0.990] | - | 6 | 5 | 64 | 52 |
| test | styled | 0.000 [0.000, 0.068] | 0.953 [0.869, 0.990] | 0 / 0 | 6/6 | 5/5 | 64 | 52 |
| authored_heldout | clean | 0.000 [0.000, 0.142] | 0.944 [0.727, 0.999] | - | 13 | 9 | 18 | 24 |
| authored_heldout | styled | 0.042 [0.001, 0.211] | 1.000 [0.815, 1.000] | 1 / 0 | 13/13 | 9/9 | 18 | 24 |
| ood | clean | 0.013 [0.000, 0.072] | 0.867 [0.732, 0.949] | - | 9 | 6 | 45 | 75 |
| ood | styled | 0.000 [0.000, 0.048] | 0.867 [0.732, 0.949] | 0 / 4 | 9/9 | 6/6 | 45 | 75 |

`--drop-latin`

| Split | Text | FPR [95% CI] | Recall [95% CI] | Flips → alert / → clear | Cues kept | Reassurance kept | N+ | N- |
|---|---|---|---|---|---|---|---|---|
| test | clean | 0.020 [0.000, 0.104] | 0.953 [0.869, 0.990] | - | 6 | 5 | 64 | 51 |
| test | styled | 0.000 [0.000, 0.070] | 0.953 [0.869, 0.990] | 0 / 0 | 5/6 | 5/5 | 64 | 51 |
| authored_heldout | clean | 0.000 [0.000, 0.142] | 0.944 [0.727, 0.999] | - | 13 | 9 | 18 | 24 |
| authored_heldout | styled | 0.042 [0.001, 0.211] | 1.000 [0.815, 1.000] | 1 / 0 | 9/13 | 9/9 | 18 | 24 |
| ood | clean | 0.013 [0.000, 0.072] | 0.867 [0.732, 0.949] | - | 9 | 6 | 45 | 75 |
| ood | styled | 0.000 [0.000, 0.048] | 0.867 [0.732, 0.949] | 0 / 4 | 6/9 | 6/6 | 45 | 75 |
test: 1 skipped (nothing left after styling): neg_legit_gov_service_mixed_6

Adversarial (`python -m qorgan.eval.adversarial [--split adversarial_legit]`):

| Set | N | Recall [95% CI] |
|---|---|---|
| source scams (test + ood) | 109 | 0.917 [0.849, 0.962] |
| adversarial (lexicon-free paraphrases) | 109 | 0.936 [0.872, 0.974] |
Recall drop: **-1.8 points** (2 scams flip to clear, 4 flip to scam) -- within the 15-point gate.

| Set | N | Recall [95% CI] |
|---|---|---|
| source scams (test + ood) | 109 | 0.917 [0.849, 0.962] |
| adversarial_legit (lexicon-free + legit-sounding register) | 109 | 0.872 [0.794, 0.928] |
Recall drop: **4.6 points** (12 scams flip to clear, 7 flip to scam) -- within the 15-point gate.

Streaming (`python -m qorgan.eval.stream --split test --split authored_heldout --backend linear`):

| Split | False-Latch Rate [95% CI] | Alert-Hit Rate [95% CI] | Median Turns | P90 Turns | N+ | N- |
|---|---|---|---|---|---|---|
| test | 0.058 [0.012, 0.159] | 0.969 [0.892, 0.996] | 3.000 | 3.000 | 64 | 52 |
| authored_heldout | 0.125 [0.027, 0.324] | 0.889 [0.653, 0.986] | 3.000 | 3.000 | 18 | 24 |

**Same runtime, before → after:** styled test FPR 5/52 → **0/52**; streaming test false-latch
5/52 → **3/52** (alerts 62/64 unchanged); adversarial-legit recall → **0.872**, cue-free
0.936; clean recall unchanged (test 0.953, authored 17/18, ood 0.867). **Costs, by name:**
`ood_neg_legit_bank_call_mixed_3` 0.669 on the server (0.154 in the browser); a third
transient authored latch, `real_neg_bank_fraud_alert_ru` (an inspected anchor, the A9b
hairline call; the other two, `real_neg_telecom_tariff_ru` and
`real_neg_telecom_sim_reregistration_ru`, latch on both models) — 3/24 against the A6 gate's
2/24, alert-hits unchanged at 16/18. Named calls, not rates: every one of these sits inside
the interval next to it. `neg_legit_gov_service_kk_19` (test, 0.71) is a false positive on
both models under this runtime. Per-tactic thresholds re-tuned on the doubled out-of-fold
set: bank 0.55 · gov_police 0.70 · telecom 0.65 · urgency 0.50 · fear 0.55 · secrecy 0.60 ·
otp 0.65 · credentials 0.60 · safe_account 0.65 · payment 0.60 · remote 0.55 · prize 0.60 ·
investment 0.55 · mule 0.60 · verification 0.65.

**Corpus finding (not fixed here):** three negatives are corrupted text — `neg_legit_gov_service_mixed_6`
(test: control characters inside Latin-script words; nothing survives `--drop-latin`, hence
"1 skipped"), `ood_neg_legit_gov_service_mixed_2` (backspaces), `reassure_legit_bank_call_mixed_1`
(train: literal `\u` escapes) — plus twelve train / val rows wrapped in `\r\n` and stray
quotes. All negatives, all scoring correctly; editing test / ood is an eval change and gets
its own entry when done.

## Addendum (2026-09-21) — the report now describes the device (ADR D33, PLAN B10)

The heads are trained and every table below is computed on the **browser's own embeddings**:
`QORGAN_EMBED_BACKEND=device` sends each text to the site's real embedding worker in headless
Chromium (`npm run device:serve`, WASM, one text per graph run) and caches the vector by
(model, browser build, text). The vectors are bit-identical to those captured from the page
(max |Δ| 0.0), so the browser gate is exact by construction and the JS scorer is checked on
real device input:

`tests_js/integration/browser_gate.test.mjs`: **flips 0 / 200, |Δrisk| p95 0.000, max 0.000,
tag Jaccard 1.000.**

### The same runtime, both head sets — what the device actually decides

Left: the heads shipped by D31/D32 (fitted to the server's native ORT 1.21), evaluated on
device embeddings; right: the heads fitted to device embeddings (shipped now).

`python -m qorgan.eval.run --split test --split authored_heldout --split ood --backend linear`

Previously shipped heads, on the device:

| Split | FPR [95% CI] | Precision | Recall [95% CI] | F1 | PR-AUC [95% CI] | N |
|---|---|---|---|---|---|---|
| test | 0.000 [0.000, 0.068] | 1.000 | 0.953 [0.869, 0.990] | 0.976 | 1.000 [1.000, 1.000] | 116 |
| authored_heldout | 0.000 [0.000, 0.142] | 1.000 | 0.889 [0.653, 0.986] | 0.941 | 1.000 [1.000, 1.000] | 42 |
| authored_heldout (clean) | 0.000 [0.000, 0.176] | 1.000 | 0.889 [0.653, 0.986] | 0.941 | 1.000 [1.000, 1.000] | 37 |
| authored_heldout (inspected) | 0.000 [0.000, 0.522] | 0.000 | 0.000 [-] | 0.000 | 0.000 [-] | 5 |
| ood | 0.000 [0.000, 0.048] | 1.000 | 0.889 [0.759, 0.963] | 0.941 | 0.993 [0.978, 1.000] | 120 |

Device-trained heads (shipped):

| Split | FPR [95% CI] | Precision | Recall [95% CI] | F1 | PR-AUC [95% CI] | N |
|---|---|---|---|---|---|---|
| test | 0.000 [0.000, 0.068] | 1.000 | 0.953 [0.869, 0.990] | 0.976 | 1.000 [1.000, 1.000] | 116 |
| authored_heldout | 0.000 [0.000, 0.142] | 1.000 | 0.889 [0.653, 0.986] | 0.941 | 0.994 [0.969, 1.000] | 42 |
| authored_heldout (clean) | 0.000 [0.000, 0.176] | 1.000 | 0.889 [0.653, 0.986] | 0.941 | 1.000 [1.000, 1.000] | 37 |
| authored_heldout (inspected) | 0.000 [0.000, 0.522] | 0.000 | 0.000 [-] | 0.000 | 0.000 [-] | 5 |
| ood | 0.000 [0.000, 0.048] | 1.000 | 0.867 [0.732, 0.949] | 0.929 | 0.995 [0.981, 1.000] | 120 |

`python -m qorgan.eval.asr_realism` — previously shipped heads, on the device:

| Split | Text | FPR [95% CI] | Recall [95% CI] | Flips → alert / → clear | Cues kept | Reassurance kept | N+ | N- |
|---|---|---|---|---|---|---|---|---|
| test | clean | 0.000 [0.000, 0.068] | 0.953 [0.869, 0.990] | - | 6 | 5 | 64 | 52 |
| test | styled | 0.000 [0.000, 0.068] | 0.953 [0.869, 0.990] | 0 / 0 | 6/6 | 5/5 | 64 | 52 |
| authored_heldout | clean | 0.000 [0.000, 0.142] | 0.889 [0.653, 0.986] | - | 13 | 9 | 18 | 24 |
| authored_heldout | styled | 0.000 [0.000, 0.142] | 1.000 [0.815, 1.000] | 0 / 0 | 13/13 | 9/9 | 18 | 24 |
| ood | clean | 0.000 [0.000, 0.048] | 0.889 [0.759, 0.963] | - | 9 | 6 | 45 | 75 |
| ood | styled | 0.000 [0.000, 0.048] | 0.822 [0.679, 0.920] | 0 / 5 | 9/9 | 6/6 | 45 | 75 |

Device-trained heads:

| Split | Text | FPR [95% CI] | Recall [95% CI] | Flips → alert / → clear | Cues kept | Reassurance kept | N+ | N- |
|---|---|---|---|---|---|---|---|---|
| test | clean | 0.000 [0.000, 0.068] | 0.953 [0.869, 0.990] | - | 6 | 5 | 64 | 52 |
| test | styled | 0.000 [0.000, 0.068] | 0.969 [0.892, 0.996] | 0 / 0 | 6/6 | 5/5 | 64 | 52 |
| authored_heldout | clean | 0.000 [0.000, 0.142] | 0.889 [0.653, 0.986] | - | 13 | 9 | 18 | 24 |
| authored_heldout | styled | 0.000 [0.000, 0.142] | 1.000 [0.815, 1.000] | 0 / 0 | 13/13 | 9/9 | 18 | 24 |
| ood | clean | 0.000 [0.000, 0.048] | 0.867 [0.732, 0.949] | - | 9 | 6 | 45 | 75 |
| ood | styled | 0.000 [0.000, 0.048] | 0.844 [0.705, 0.935] | 0 / 4 | 9/9 | 6/6 | 45 | 75 |

`--drop-latin`, device-trained heads:

| Split | Text | FPR [95% CI] | Recall [95% CI] | Flips → alert / → clear | Cues kept | Reassurance kept | N+ | N- |
|---|---|---|---|---|---|---|---|---|
| test | clean | 0.000 [0.000, 0.070] | 0.953 [0.869, 0.990] | - | 6 | 5 | 64 | 51 |
| test | styled | 0.000 [0.000, 0.070] | 0.969 [0.892, 0.996] | 0 / 0 | 5/6 | 5/5 | 64 | 51 |
| authored_heldout | clean | 0.000 [0.000, 0.142] | 0.889 [0.653, 0.986] | - | 13 | 9 | 18 | 24 |
| authored_heldout | styled | 0.000 [0.000, 0.142] | 0.944 [0.727, 0.999] | 0 / 1 | 9/13 | 9/9 | 18 | 24 |
| ood | clean | 0.000 [0.000, 0.048] | 0.867 [0.732, 0.949] | - | 9 | 6 | 45 | 75 |
| ood | styled | 0.000 [0.000, 0.048] | 0.844 [0.705, 0.935] | 0 / 4 | 6/9 | 6/6 | 45 | 75 |
test: 1 skipped (nothing left after styling): neg_legit_gov_service_mixed_6

Adversarial — previously shipped heads, on the device:

| Set | N | Recall [95% CI] |
|---|---|---|
| source scams (test + ood) | 109 | 0.927 [0.860, 0.968] |
| adversarial (lexicon-free paraphrases) | 109 | 0.917 [0.849, 0.962] |
Recall drop: **0.9 points** (6 scams flip to clear, 5 flip to scam) -- within the 15-point gate.

| Set | N | Recall [95% CI] |
|---|---|---|
| source scams (test + ood) | 109 | 0.927 [0.860, 0.968] |
| adversarial_legit (lexicon-free + legit-sounding register) | 109 | 0.798 [0.711, 0.869] |
Recall drop: **12.8 points** (19 scams flip to clear, 5 flip to scam) -- within the 15-point gate.

Device-trained heads:

| Set | N | Recall [95% CI] |
|---|---|---|
| source scams (test + ood) | 109 | 0.917 [0.849, 0.962] |
| adversarial (lexicon-free paraphrases) | 109 | 0.917 [0.849, 0.962] |
Recall drop: **0.0 points** (6 scams flip to clear, 6 flip to scam) -- within the 15-point gate.

| Set | N | Recall [95% CI] |
|---|---|---|
| source scams (test + ood) | 109 | 0.917 [0.849, 0.962] |
| adversarial_legit (lexicon-free + legit-sounding register) | 109 | 0.807 [0.721, 0.877] |
Recall drop: **11.0 points** (18 scams flip to clear, 6 flip to scam) -- within the 15-point gate.

Streaming — previously shipped heads, on the device:

| Split | False-Latch Rate [95% CI] | Alert-Hit Rate [95% CI] | Median Turns | P90 Turns | N+ | N- |
|---|---|---|---|---|---|---|
| test | 0.058 [0.012, 0.159] | 0.969 [0.892, 0.996] | 3.000 | 3.000 | 64 | 52 |
| authored_heldout | 0.125 [0.027, 0.324] | 0.833 [0.586, 0.964] | 3.000 | 3.000 | 18 | 24 |

Device-trained heads:

| Split | False-Latch Rate [95% CI] | Alert-Hit Rate [95% CI] | Median Turns | P90 Turns | N+ | N- |
|---|---|---|---|---|---|---|
| test | 0.058 [0.012, 0.159] | 0.969 [0.892, 0.996] | 3.000 | 3.000 | 64 | 52 |
| authored_heldout | 0.125 [0.027, 0.324] | 0.833 [0.586, 0.964] | 3.000 | 3.000 | 18 | 24 |

Streaming with the inspection ledger applied (`eval.stream` now reports the `(clean)` /
`(inspected)` rows the single-shot harness has reported since A2):

| Split | False-Latch Rate [95% CI] | Alert-Hit Rate [95% CI] | Median Turns | P90 Turns | N+ | N- |
|---|---|---|---|---|---|---|
| test | 0.058 [0.012, 0.159] | 0.969 [0.892, 0.996] | 3.000 | 3.000 | 64 | 52 |
| authored_heldout | 0.125 [0.027, 0.324] | 0.833 [0.586, 0.964] | 3.000 | 3.000 | 18 | 24 |
| authored_heldout (clean) | 0.053 [0.001, 0.260] | 0.833 [0.586, 0.964] | 3.000 | 3.000 | 18 | 19 |
| authored_heldout (inspected) | 0.400 [0.053, 0.853] | 0.000 [-] | - | - | 0 | 5 |

The generalisation false-latch is **1/19 clean negatives** (`real_neg_telecom_sim_reregistration_ru`);
the other two transient latches are anchors that were read during feature engineering.

**Read together:** on the device the two head sets decide the same within a call. The
"1/52 test" and "1/75 ood" false positives in the 2026-09-20 addendum were the *server's*
runtime — the device never raised them (`neg_legit_gov_service_kk_19`,
`ood_neg_legit_bank_call_mixed_3` are clear here). What the device-trained heads change is
that these numbers are now exactly what the browser computes, with no proxy gap to explain.
**Named, on the device:** two authored scams under the threshold — `real_scam_delivery_customs_ru`
0.515 and `real_scam_prize_phone_kk` 0.514 (authored recall 16/18 [0.653, 0.986]); in
streaming three scams never latch (those two and `real_scam_telecom_verify_ru`, whose first
two windows read 0.34 / 0.39) and the three transient legit latches of D31 remain
(`bank_fraud_alert_ru`, `telecom_tariff_ru`, `telecom_sim_reregistration_ru` — all
institutional openers, two of them inspected anchors): 3/24 against the A6 gate's 2/24.
The legit-sounding adversary costs 11.0 points on the device (0.807) — inside the 15-point
gate, more than the 4.6 the server-side numbers had suggested. Two ood scams sit just under:
`ood_impersonation_gov_police_mixed_0` 0.580, `ood_payment_redirect_mixed_0` 0.539.

### The server is now the proxy

The server keeps its native ONNX Runtime (pinned, D32) for `/api/analyze` and Level-2
scoring and loads the device-trained heads as their documented proxy. Against the device's
own verdicts on the 200 gate transcripts it disagrees on **6 / 200** (p95 |Δrisk| 0.114,
max 0.513 — `ood_neg_legit_bank_call_mixed_3`, device 0.114 / server 0.627, again); on the
28 golden cases Node's native runtime sits at cosine min 0.967 and |Δrisk| max 0.20
(`tests_js/integration/embedding.test.mjs`, now asserting the proxy level when the fixtures
are device-made). These are the numbers that apply to the server fallback, and they are
reported, not gated: the product's Level 1 runs on the device.

## Addendum (2026-09-21, later) — corpus repair (ADR D34): the eval sets' text changes, so the tables are regenerated

`data/clean.py` repaired generation artefacts and dropped rows whose Kazakh letters had
become control characters: `test` 116 → 115, `ood` 120 → 118, train 1,422 → 1,404 (with the
styled copies); `authored_heldout` untouched. Device embeddings throughout (ADR D33); browser
gate re-captured on the new gate set: **0 / 200, |Δrisk| 0.000, Jaccard 1.000**. Among the
dropped ood rows is `ood_impersonation_gov_police_mixed_0` — the transcript with the largest
server-vs-device delta in every earlier gate run was corrupted text.

`python -m qorgan.eval.run --split test --split authored_heldout --split ood --backend linear`

| Split | FPR [95% CI] | Precision | Recall [95% CI] | F1 | PR-AUC [95% CI] | N |
|---|---|---|---|---|---|---|
| test | 0.000 [0.000, 0.070] | 1.000 | 0.953 [0.869, 0.990] | 0.976 | 1.000 [1.000, 1.000] | 115 |
| authored_heldout | 0.000 [0.000, 0.142] | 1.000 | 0.889 [0.653, 0.986] | 0.941 | 1.000 [1.000, 1.000] | 42 |
| authored_heldout (clean) | 0.000 [0.000, 0.176] | 1.000 | 0.889 [0.653, 0.986] | 0.941 | 1.000 [1.000, 1.000] | 37 |
| authored_heldout (inspected) | 0.000 [0.000, 0.522] | 0.000 | 0.000 [-] | 0.000 | 0.000 [-] | 5 |
| ood | 0.000 [0.000, 0.049] | 1.000 | 0.886 [0.754, 0.962] | 0.940 | 0.994 [0.981, 1.000] | 118 |

`python -m qorgan.eval.asr_realism`

| Split | Text | FPR [95% CI] | Recall [95% CI] | Flips → alert / → clear | Cues kept | Reassurance kept | N+ | N- |
|---|---|---|---|---|---|---|---|---|
| test | clean | 0.000 [0.000, 0.070] | 0.953 [0.869, 0.990] | - | 6 | 5 | 64 | 51 |
| test | styled | 0.000 [0.000, 0.070] | 0.953 [0.869, 0.990] | 0 / 0 | 6/6 | 5/5 | 64 | 51 |
| authored_heldout | clean | 0.000 [0.000, 0.142] | 0.889 [0.653, 0.986] | - | 13 | 9 | 18 | 24 |
| authored_heldout | styled | 0.000 [0.000, 0.142] | 1.000 [0.815, 1.000] | 0 / 0 | 13/13 | 9/9 | 18 | 24 |
| ood | clean | 0.000 [0.000, 0.049] | 0.886 [0.754, 0.962] | - | 9 | 6 | 44 | 74 |
| ood | styled | 0.000 [0.000, 0.049] | 0.864 [0.726, 0.948] | 0 / 3 | 9/9 | 6/6 | 44 | 74 |

`--drop-latin`

| Split | Text | FPR [95% CI] | Recall [95% CI] | Flips → alert / → clear | Cues kept | Reassurance kept | N+ | N- |
|---|---|---|---|---|---|---|---|---|
| test | clean | 0.000 [0.000, 0.070] | 0.953 [0.869, 0.990] | - | 6 | 5 | 64 | 51 |
| test | styled | 0.000 [0.000, 0.070] | 0.953 [0.869, 0.990] | 0 / 0 | 5/6 | 5/5 | 64 | 51 |
| authored_heldout | clean | 0.000 [0.000, 0.142] | 0.889 [0.653, 0.986] | - | 13 | 9 | 18 | 24 |
| authored_heldout | styled | 0.000 [0.000, 0.142] | 0.944 [0.727, 0.999] | 0 / 1 | 9/13 | 9/9 | 18 | 24 |
| ood | clean | 0.000 [0.000, 0.049] | 0.886 [0.754, 0.962] | - | 9 | 6 | 44 | 74 |
| ood | styled | 0.000 [0.000, 0.049] | 0.864 [0.726, 0.948] | 0 / 3 | 6/9 | 6/6 | 44 | 74 |

Adversarial (108 source scams now):

| Set | N | Recall [95% CI] |
|---|---|---|
| source scams (test + ood) | 108 | 0.926 [0.859, 0.967] |
| adversarial (lexicon-free paraphrases) | 108 | 0.917 [0.848, 0.961] |
Recall drop: **0.9 points** (6 scams flip to clear, 5 flip to scam) -- within the 15-point gate.

| Set | N | Recall [95% CI] |
|---|---|---|
| source scams (test + ood) | 108 | 0.926 [0.859, 0.967] |
| adversarial_legit (lexicon-free + legit-sounding register) | 108 | 0.815 [0.729, 0.883] |
Recall drop: **11.1 points** (18 scams flip to clear, 6 flip to scam) -- within the 15-point gate.

Streaming (with the inspection ledger):

| Split | False-Latch Rate [95% CI] | Alert-Hit Rate [95% CI] | Median Turns | P90 Turns | N+ | N- |
|---|---|---|---|---|---|---|
| test | 0.059 [0.012, 0.162] | 0.969 [0.892, 0.996] | 3.000 | 3.000 | 64 | 51 |
| authored_heldout | 0.125 [0.027, 0.324] | 0.833 [0.586, 0.964] | 3.000 | 3.000 | 18 | 24 |
| authored_heldout (clean) | 0.053 [0.001, 0.260] | 0.833 [0.586, 0.964] | 3.000 | 3.000 | 18 | 19 |
| authored_heldout (inspected) | 0.400 [0.053, 0.853] | 0.000 [-] | - | - | 0 | 5 |

Versus the D33 tables: test and authored unchanged (the dropped test row was a legit call the
model already cleared); ood recall 0.867 → **0.886** (one of the dropped rows was a scam the
model missed — corrupted text, not a detection failure); everything else within a call.


## Addendum (2026-09-21, evening) — a second generator's calls (ADR D35), utterance-level heads (ADR D36), a server-tier embedder (ADR D37)

### The generator-shift split: `python -m qorgan.eval.run --split shift --by-language`

66 dialogues authored by a second generator (Claude, no sight of the corpus / prompts /
lexicons; `data/README.md`), 33 scams / 33 confusable legit, 22 per language, evaluation-only.
Shipped device-trained heads, device embeddings, threshold 0.59:

| Split | FPR [95% CI] | Precision | Recall [95% CI] | F1 | PR-AUC [95% CI] | N |
|---|---|---|---|---|---|---|
| shift | 0.030 [0.001, 0.158] | 0.889 | 0.242 [0.111, 0.423] | 0.381 | 0.870 [0.749, 0.961] | 66 |
| test | 0.000 [0.000, 0.070] | 1.000 | 0.953 [0.869, 0.990] | 0.976 | 1.000 [1.000, 1.000] | 115 |

| Split | FPR [95% CI] | Precision | Recall [95% CI] | F1 | PR-AUC [95% CI] | N |
|---|---|---|---|---|---|---|
| kk | 0.000 [0.000, 0.285] | 1.000 | 0.182 [0.023, 0.518] | 0.308 | 0.911 [0.738, 1.000] | 22 |
| mixed | 0.091 [0.002, 0.413] | 0.500 | 0.091 [0.002, 0.413] | 0.154 | 0.779 [0.533, 0.991] | 22 |
| ru | 0.000 [0.000, 0.285] | 1.000 | 0.455 [0.167, 0.766] | 0.625 | 0.942 [0.788, 1.000] | 22 |

Read plainly: **the shipped model catches 8 of 33 scams written in another register.** Median
scam risk 0.31; 17 / 33 above 0.30, 14 / 33 above 0.40 (where two legit calls also sit:
`shift_legit_mixed_05`, a teacher collecting money for a class trip, 0.62; `shift_legit_kk_11`,
a deposit sales call, 0.47). The cue lexicon fired on 6 / 33 scams — and one of those is the
*callee* quoting the SMS warning ("никому не говорите", `shift_scam_mixed_03`, risk 0.02).
Textbook schemes score near zero: customs-fee courier (`kk_03`, 0.02), prize with delivery fee
(`kk_06` 0.03, `mixed_06` 0.00), relative-in-trouble (`mixed_08` 0.04, `kk_08` 0.10), mule
recruitment (`mixed_07` 0.07). What still works: the three real fraud-alert calls and the card
courier — the reassurance feature — score ≤ 0.01; the two police / safe-account long calls
score 0.73–0.87; the subtle "read me what the app shows" calls reach 0.21–0.51. The tactic
head is at F1 0 for 11 of 15 tactics on this set. The two adversarial splits (0.917 / 0.807)
did not predict this: they are Gemini paraphrasing Gemini. This number replaces the headline
until real calls exist; the mitigation ladder is in ADR D35.

### Utterance-level tactic head — measured no-go (ADR D36)

Device embeddings (already computed per utterance for the highlights), weak instance labels
(evidence utterances × dialogue tags, styled copies inherit the mask), grouped 5-fold
out-of-fold threshold tuning as in D30, verbatim-cue merge as in `predict`. Micro-F1 over
(dialogue, tactic) pairs after the cue merge; the baseline row reproduces the shipped head on
the device runtime (the D30 table was the server runtime's):

| Variant | test micro-F1 | authored micro-F1 | ood micro-F1 |
|---|---|---|---|
| shipped dialogue head (D30 thresholds) — baseline | 0.804 (P 0.78 / R 0.83, 2.07 tags) | 0.609 (P 0.60 / R 0.61, 1.38 tags) | 0.490 (P 0.35 / R 0.82, 0.87 tags) |
| utterance head, balanced LR, max pooling | 0.545 (P 0.38 / R 0.94, 4.71 tags) | 0.538 (P 0.40 / R 0.81, 2.71 tags) | 0.255 (P 0.15 / R 0.93, 2.36 tags) |
| + 2 MIL relabel rounds | 0.552 (P 0.39 / R 0.95, 4.69 tags) | 0.573 (P 0.44 / R 0.82, 2.55 tags) | 0.248 (P 0.14 / R 0.93, 2.43 tags) |
| utterance head, mean of top-2 | 0.692 (P 0.56 / R 0.91, 3.17 tags) | 0.545 (P 0.52 / R 0.58, 1.52 tags) | 0.376 (P 0.24 / R 0.93, 1.47 tags) |
| utterance head, unbalanced LR, max, wide grid | 0.675 (P 0.82 / R 0.58, 1.37 tags) | 0.458 (P 0.73 / R 0.33, 0.62 tags) | 0.328 (P 0.25 / R 0.48, 0.71 tags) |
| utterance head, balanced C=0.3, softmax pooling | 0.743 (P 0.74 / R 0.74, 1.93 tags) | 0.596 (P 0.60 / R 0.60, 1.36 tags) | 0.475 (P 0.37 / R 0.66, 0.66 tags) |
| stacking: transcript embedding + utterance aggregates | 0.783 (P 0.76 / R 0.81, 2.04 tags) | 0.608 (P 0.69 / R 0.54, 1.07 tags) | 0.416 (P 0.30 / R 0.70, 0.89 tags) |
| ensemble 0.5 / 0.5 (dialogue + utterance-max) | 0.814 (P 0.84 / R 0.79, 1.81 tags) | 0.620 (P 0.72 / R 0.54, 1.02 tags) | 0.493 (P 0.37 / R 0.75, 0.76 tags) |
| ensemble 0.5 / 0.5 (dialogue + softmax pooling) | 0.823 (P 0.85 / R 0.80, 1.81 tags) | 0.598 (P 0.64 / R 0.56, 1.19 tags) | 0.464 (P 0.36 / R 0.66, 0.69 tags) |

Max pooling over-fires (4.7 tags per dialogue); the best ensemble is +0.01–0.02 on `test`
and ±0.01 elsewhere — noise at these sizes. The spans are not tied to tactics, so the instance
labels add nothing the whole-transcript embedding does not already carry.

### Utterance-level risk mixed into the dialogue risk — a knob, not a ship (ADR D36)

A separate calibrated utterance-risk head (same weak labels, 2 MIL rounds); risk = (1 − w) ×
dialogue head + w × max utterance prob. As an extra *feature* or as pure max pooling the FPR
gate fails (1–2 ood negatives fire, one at 0.98) — those rows are in the scratchpad JSON.
The mix, at 0.59:

| w (utterance share) | test FP / max neg | authored FP / max neg | ood FP / max neg | shift FP / max neg | recall test · authored · ood · adv · adv-legit · shift |
|---|---|---|---|---|---|
| 0.0 | 0 / 0.28 | 0 / 0.51 | 0 / 0.31 | 1 / 0.62 | 0.953 · 0.889 · 0.886 · 0.917 · 0.807 · 0.242 |
| 0.15 | 0 / 0.31 | 0 / 0.46 | 0 / 0.30 | 0 / 0.53 | 0.953 · 0.889 · 0.886 · 0.917 · 0.853 · 0.212 |
| 0.25 | 0 / 0.33 | 0 / 0.42 | 0 / 0.31 | 0 / 0.46 | 0.953 · 0.889 · 0.864 · 0.908 · 0.872 · 0.242 |
| 0.35 | 0 / 0.36 | 0 / 0.39 | 0 / 0.40 | 0 / 0.40 | 0.969 · 0.944 · 0.841 · 0.908 · 0.881 · 0.273 |
| 0.5 | 0 / 0.39 | 0 / 0.34 | 0 / 0.52 | 0 / 0.31 | 0.953 · 0.889 · 0.886 · 0.917 · 0.936 · 0.273 |

w = 0.15 is nearly free (only adversarial-legit recall and the authored margin move);
w = 0.35 buys authored 17 / 18 and test 0.969 at the price of two ood scams and a 0.40 ood
margin; w = 0.5 buys adversarial-legit 0.936 with an ood negative at 0.52. None of it moves
`shift` beyond 0.27. Not shipped: a second head + JS port + fixtures + gate re-capture for
≤ 5 adversarial dialogues, before real calls can say which end of the trade matters.

### Streaming on `shift`: `python -m qorgan.eval.stream --split shift --backend linear`

| Split | False-Latch Rate [95% CI] | Alert-Hit Rate [95% CI] | Median Turns | P90 Turns | N+ | N- |
|---|---|---|---|---|---|---|
| shift | 0.000 [0.000, 0.106] | 0.485 [0.308, 0.665] | 5.000 | 8.000 | 33 | 33 |

The live meter (windows + hard-signal floors) alerts on 16 / 33 — twice the whole-transcript
recall — with no false latch on the 33 legit calls; median 5 turns to alert.

### A stronger server-side embedder — `multilingual-e5-large`, same heads (ADR D37)

`QORGAN_EMBED_BACKEND=sentence-transformers QORGAN_EMBED_MODEL_NAME=intfloat/multilingual-e5-large
QORGAN_LINEAR_MODEL_DIR=models/linear_e5large python -m qorgan.eval.run --split shift --split test
--split authored_heldout --split ood --split adversarial --split adversarial_legit --by-language`

| Split | FPR [95% CI] | Precision | Recall [95% CI] | F1 | PR-AUC [95% CI] | N |
|---|---|---|---|---|---|---|
| shift | 0.000 [0.000, 0.106] | 1.000 | 0.242 [0.111, 0.423] | 0.390 | 0.953 [0.897, 0.990] | 66 |
| test | 0.000 [0.000, 0.070] | 1.000 | 0.953 [0.869, 0.990] | 0.976 | 1.000 [0.999, 1.000] | 115 |
| authored_heldout | 0.000 [0.000, 0.142] | 1.000 | 0.889 [0.653, 0.986] | 0.941 | 1.000 [1.000, 1.000] | 42 |
| authored_heldout (clean) | 0.000 [0.000, 0.176] | 1.000 | 0.889 [0.653, 0.986] | 0.941 | 1.000 [1.000, 1.000] | 37 |
| authored_heldout (inspected) | 0.000 [0.000, 0.522] | 0.000 | 0.000 [-] | 0.000 | 0.000 [-] | 5 |
| ood | 0.000 [0.000, 0.049] | 1.000 | 0.864 [0.726, 0.948] | 0.927 | 0.995 [0.985, 1.000] | 118 |
| adversarial | 0.000 [-] | 1.000 | 0.917 [0.849, 0.962] | 0.957 | 0.000 [-] | 109 |
| adversarial_legit | 0.000 [-] | 1.000 | 0.853 [0.773, 0.914] | 0.921 | 0.000 [-] | 109 |

| Split | FPR [95% CI] | Precision | Recall [95% CI] | F1 | PR-AUC [95% CI] | N |
|---|---|---|---|---|---|---|
| kk | 0.000 [0.000, 0.285] | 0.000 | 0.000 [0.000, 0.285] | 0.000 | 0.952 [0.842, 1.000] | 22 |
| mixed | 0.000 [0.000, 0.285] | 1.000 | 0.273 [0.060, 0.610] | 0.429 | 0.975 [0.889, 1.000] | 22 |
| ru | 0.000 [0.000, 0.285] | 1.000 | 0.455 [0.167, 0.766] | 0.625 | 0.969 [0.861, 1.000] | 22 |

Same model on every Gemini split (adversarial-legit +0.04 is the only visible move), the same
8 / 33 on `shift` with Kazakh at 0 / 11, one fewer false positive, a better threshold-free
ranking (PR-AUC 0.953 vs 0.870). A 3× encoder does not buy cross-generator recall: the
bottleneck is one generator's register in `train`. No tier.
