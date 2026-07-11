# Eval report — classifier backends

FPR is the **primary** metric (a false alarm on a real bank call destroys trust). Numbers
are reported on `test` **and** `real_heldout` **separately**; `real_heldout` (hand-written,
different source) is the honest generalization signal. Regenerate with:

```bash
python -m qorgan.eval.run --split test --split real_heldout --backend <linear|llm|mock> --by-language
```

The CLI also prints a **per-tactic F1** table and a **recommended alert threshold** — the
cutoff maximising recall subject to `fpr <= 0.05` on `real_heldout` (`eval/threshold.py`).

## Corpus
**713 dialogues**: 360 scams + **353 adversarial hard negatives** (legitimate calls in the
*same domains* as scams — bank, gov, telecom, delivery — that superficially resemble a scam
but contain none of the tactics). Balanced across RU/KK/mixed. Splits: train 524 / val 91 /
test 98 / real_heldout 10 (`data/processed/manifest.json`).

## `linear` backend — **SHIPPING offline model** ✅
Frozen `multilingual-e5-base` embeddings → class-weighted, **calibrated** Logistic
Regression (risk head) + per-tactic LR (tactic head). Trains in seconds on CPU; the export
is ~60 KB. This replaces the collapsed XLM-R fine-tune.

Corpus **833 dialogues** (405 scam / 428 negative), after augmenting with disfluent-style
examples + more adversarial negatives. Evaluated on three sets — in-distribution synthetic,
hand-written (held-out source), and a cross-distribution OOD set (disfluent / ASR style):

| Split | FPR | Precision | Recall | F1 | PR-AUC | N |
|---|---|---|---|---|---|---|
| test (in-distribution synthetic) | 0.000 | 1.000 | 0.969 | 0.984 | 1.000 | 116 |
| **real_heldout (hand-written)** | **0.143** | 0.857 | 0.923 | 0.889 | 0.938 | 27 |
| **ood (disfluent/ASR-style)** | 0.000 | 1.000 | 0.911 | 0.953 | 0.999 | 120 |

**The honest read.** `test` is near-perfect but **in-distribution** (both classes are
Gemini-generated → overstates real performance). The two held-out sets are the trustworthy
signals. A targeted augmentation round (disfluent examples + more adversarial negatives)
**fixed the recall gap** but **not the FPR gap**:

- **Recall improved sharply and durably:** OOD 0.71 → **0.91** (over 120 examples — a real
  gain), real_heldout 0.77 → **0.92**. The disfluent/ASR-style training examples worked.
- **FPR did not move:** real_heldout FPR **0.143**, all of it in **Russian (0.33)**. Concretely
  it is **2 legitimate RU calls out of 14 benign** (the hardest cases — a real bank
  fraud-alert and a "card is ready" call) that the model still flags. More generic adversarial
  negatives raised recall but didn't teach the boundary on these specific hard legit calls.
- Low-FPR tradeoff is still steep (FPR ≤ 0.05 → threshold ~0.99 → recall 0.54).

PR-AUC stays 0.94–1.0 (ranking sound); the residual gap is a precise, small one: a handful of
hard *legitimate* bank/official calls. **Next lever** (diminishing returns from generic data):
either a focused batch of RU legit-bank/official negatives, or a **hybrid feature** — add
explicit hard-signal cues (OTP-ask / CVV-ask / transfer-to-"safe"-account / remote-access
request) alongside the embedding, since a real bank call *never* makes those requests. Weighed
against the sprint deadline, the model is now **materially better (recall +15–20 pts at equal
FPR)** and honestly characterized.

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
