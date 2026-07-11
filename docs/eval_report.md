# Eval report — classifier backends (through Day 3)

FPR is the **primary** metric (a false alarm on a real bank call destroys trust). Numbers
are reported on `test` **and** `real_heldout` **separately**; `real_heldout` is the honest
generalization signal. Regenerate with:

```bash
python -m qorgan.eval.run --split test --split real_heldout --backend <llm|xlmr|mock>
```

## Corpus
510 synthetic dialogues (15 tactics × 3 langs, balanced) + 10 curated `real_heldout`
anchors. Splits: train 382 / val 59 / test 69 / real_heldout 10 (see
`data/processed/manifest.json`).

## `mock` backend (deterministic keyword safety-net, no API)
| Split | FPR | Precision | Recall | F1 | PR-AUC | N |
|---|---|---|---|---|---|---|
| test | 0.000 | 0.000 | 0.000 | 0.000 | 0.795 | 69 |
| real_heldout | 0.000 | 1.000 | 0.600 | 0.750 | 1.000 | 10 |

The mock matcher can't hit LLM-generated wording (→ 0 recall on `test`), but **FPR = 0 on
both splits**: hard negatives never fire. This is a safety net, not the real classifier.

## `xlmr` backend (fine-tuned XLM-R) — **STALLED (D3), not shipping**
Three training runs (CLS→mean pooling, clamped class weights, LR warmup + grad clip, up to
lr 5e-5 × 12 epochs on Apple MPS) all **collapse to the class prior**: the model outputs a
near-constant risk (~0.64) for every input, scam or legitimate.

| Split | FPR | Precision | Recall | F1 | PR-AUC | N |
|---|---|---|---|---|---|---|
| test | 0.000 | 0.000 | 0.000 | 0.000 | 0.883 | 69 |
| real_heldout | 0.000 | 0.000 | 0.000 | 0.000 | 0.743 | 10 |

**Root cause (data, not code):** 382 training examples are too few / too homogeneous for
XLM-R base (270M) to learn a *subtle same-topic* distinction. A scam bank call and a
legitimate bank call share ~80% vocabulary; the discriminative cues (OTP request, secrecy,
safe-account transfer) are a small minority of tokens, so the pooled representations are
near-identical (measured cosine ≈ 1.0 between a scam-bank and legit-bank call, vs 0.6 to an
off-topic chit-chat). The loss floors at the "predict the prior" value.

**Decision:** per the plan's DoD ("if D3-1 stalls, skip D3 — `llm` keeps the demo fully
functional"), **`llm` is the shipping backend.** The `xlmr` *code path* is complete and
smoke-tested (model, class-weighted training, temperature calibration, Captum IG
attribution, `predict` wiring, Colab notebook); only the *trained weights* are unusable on
this corpus.

**To revisit (Phase 1 / more data):** larger + noisier corpus (real ASR transcripts, harder
same-topic negatives, augmentation), longer training, discriminative learning rates, or a
smaller/distilled encoder better suited to a few-hundred-example regime.

## `llm` backend (Gemini structured classifier) — shipping
The `llm` baseline is the shipping/demo backend. It requires `GEMINI_API_KEY`; run
`python -m qorgan.eval.run --split test --split real_heldout --backend llm` to regenerate
its FPR-first tables (billed, ~13 calls/split). Threshold tuning + hysteresis land in Day 4.
