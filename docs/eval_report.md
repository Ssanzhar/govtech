# Eval report — classifier backends

> **2026-09-13 note (PLAN_2026-09 A1–A2).** The split formerly called `real_heldout` is now
> **`authored_heldout`**: it is hand-written by the team (18 scam / 24 legit), not real calls,
> and five of its negatives were *read* during feature engineering (see
> `data/anchors/inspection_ledger.yaml`; the harness now reports `(clean)` / `(inspected)`
> rows). The tables below are July point estimates without intervals: `FPR 0.000` on 24
> negatives has a 95 % Clopper–Pearson interval of `[0.000, 0.142]`. The harness prints
> intervals now; this report is regenerated with them in A7.

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

### On-device parity (`npm test`)
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

### Streaming (`python -m qorgan.eval.stream --split test --split authored_heldout --backend linear`)
| Split | False-Latch Rate [95% CI] | Alert-Hit Rate [95% CI] | Median Turns | P90 Turns | N+ | N- |
|---|---|---|---|---|---|---|
| test | 0.096 [0.032, 0.210] | 0.969 [0.892, 0.996] | 2.000 | 3.000 | 64 | 52 |
| authored_heldout | 0.083 [0.010, 0.270] | 0.889 [0.653, 0.986] | 2.000 | 3.000 | 18 | 24 |

False-latch (the live FPR analogue) improved with the 0.59 / 0.49 hysteresis pair: authored
2/24 (was 4/24 in July), test 0.096 (was 0.115). Two authored scams never latch in
streaming (alert-hit 0.889) — the meter's min-turns / damping work (PLAN A6) must be gated on
both numbers, not false-latch alone.
