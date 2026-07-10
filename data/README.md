# Data — provenance & processing (ТЗ §9, graded)

Fill this in **as data lands**. Every field below is scored under "Работа с данными".

## Sources
| Source | Type | Language | Role | License |
|---|---|---|---|---|
| Claude-generated dialogues | synthetic | KK/RU/mixed | main train/val/test | own (documented) |
| Scam-baiting call transcripts | real (public) | mostly RU/EN | real anchors, taxonomy grounding | per source |
| Team-collected recordings (consented) | real | KK/RU | `real_heldout` only | consent on file |
| Open dialogue corpora | real (public) | KK/RU | negatives | per source |

Reference methodology: TeleAntiFraud-28k (arXiv:2503.24115) — ASR transcripts + LLM
self-instruct + adversarial synthesis. Scam-stage/script taxonomy informed by
"An analysis of scam baiting calls" (arXiv:2307.01965).

## Structure
- Records validated against `src/qorgan/data/schema.py`.
- Dialogue: `{id, language, utterances[], label:{risk, tactic_tags[], trigger_spans[]}, is_hard_negative}`.
- Trigger spans are **verbatim substrings** of the transcript (validated).

## Generation (synthetic)
- Tactics from `data/taxonomy/tactics.yaml`; balanced across tactics + languages + hard negatives.
- Seeded + config-driven (`src/qorgan/data/generate.py`); reproducible; manifest + hash committed.

## Cleaning
- Text normalization, dedup, **PII scrubbing** (numbers hashed, names redacted).
- Consistent label schema; hard negatives explicitly marked.

## Splits
- `train` / `val` / `test` on synthetic+scraped; **separate `real_heldout`** (real consented).
- Metrics reported on `test` **and** `real_heldout` separately (see `src/qorgan/eval/`).

## Limitations (state honestly)
- No public KZ/RU scam-call transcripts exist → corpus is mostly synthetic; `real_heldout`
  is small. Synthetic data may under-represent real acoustic/linguistic noise. FPR on
  `real_heldout` is the honest generalization signal.

## Directories
- `taxonomy/` scam-tactic definitions · `raw/` sources (gitignored if large) ·
  `synthetic/` generated JSONL · `processed/` splits + manifest.
