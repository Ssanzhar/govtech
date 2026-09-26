---
language:
  - kk
  - ru
license: other
license_name: undecided
task_categories:
  - text-classification
tags:
  - fraud-detection
  - scam-detection
  - social-engineering
  - synthetic
  - code-switching
pretty_name: Qorğan — synthetic Kazakh/Russian phone-scam dialogues
size_categories:
  - 1K<n<10K
---

# Qorğan — synthetic Kazakh / Russian phone-scam dialogues

> **Licence not yet chosen by the maintainers.** Until a licence is published here, no rights
> beyond viewing are granted.
>
> **Every dialogue is synthetic or hand-written. No real call, caller or victim is in this
> dataset.** Names, numbers and account details are invented and scrubbed to placeholders.

Phone-call transcripts in **Russian, Kazakh and code-switched (mixed) speech**, labelled as scam
or legitimate, with the social-engineering **tactics** a scam uses and the **verbatim phrases**
that show them. It trains and evaluates the Qorğan decision-support model
([`sanzh-ts/govtech`](https://huggingface.co/sanzh-ts/govtech); code
[github.com/Ssanzhar/govtech](https://github.com/Ssanzhar/govtech), provenance in
`data/README.md`). Legitimate calls are mostly **hard negatives** — real-sounding bank, telecom,
delivery and government calls built to look like scams — because the product's primary metric
is the false-positive rate.

## Splits

Counts generated from the files (`scam` / `legit` follow the labelled risk, cut at 0.5;
`hard negatives` are the legit calls built to look like scams).

| split | rows | scam | legit | of which hard negatives | ru | kk | mixed |
|---|---|---|---|---|---|---|---|
| `train` | 1642 | 724 | 918 | 918 | 532 | 574 | 536 |
| `val` | 112 | 43 | 69 | 69 | 44 | 30 | 38 |
| `test` | 115 | 64 | 51 | 51 | 37 | 39 | 39 |
| `authored_heldout` | 42 | 18 | 24 | 15 | 18 | 13 | 11 |
| `ood` | 118 | 44 | 74 | 74 | 40 | 40 | 38 |
| `adversarial` | 109 | 109 | 0 | 0 | 37 | 34 | 38 |
| `adversarial_legit` | 109 | 109 | 0 | 0 | 37 | 34 | 38 |
| `shift` | 66 | 33 | 33 | 33 | 22 | 22 | 22 |

- **`train` / `val` / `test`** — Gemini self-instruct generation (`gemini-2.5-flash`), one call per
  (tactic × language) and (hard-negative category × language), then an **independent
  re-label**: a second Gemini pass reads each finished dialogue without the seed and assigns
  risk, tactics and trigger phrases (phrases not found verbatim are discarded). Deterministic
  seeded hash split (seed 42, 0.70 / 0.15 / 0.15). `train` includes 821 **ASR-styled copies**
  (lowercase, no punctuation, numbers as words — what an on-device recogniser emits) and 221
  targeted augmentation rows (`augment/`).
- **`authored_heldout`** — hand-written, **not** real calls; never used for training.
- **`ood`** — disfluent, transcribed-call-style stress set.
- **`adversarial`** — every scam in `test` + `ood` rewritten without its cue words;
  **`adversarial_legit`** — the same scams rewritten to *sound* legitimate. Both are scam-only.
- **`shift`** — 66 dialogues by a **second author (Claude)** with no access to the corpus, its
  prompts or lexicons: the only cross-generator test. Do not train on it.
- `manifest.json` / `shift.manifest.json` — counts and content hashes. Manifests written before
  2026-09-25 used `positives` for "not a hard negative"; it now equals `scam`.

## Record format (one JSON object per line)

```json
{
  "id": "…",
  "language": "ru | kk | mixed",
  "utterances": [{"speaker": "caller", "text": "…"}, {"speaker": "callee", "text": "…"}],
  "label": {
    "risk": 0.9,
    "tactic_tags": [{"id": "otp_request", "weight": 1.0}],
    "trigger_spans": [{"start": 10, "end": 30, "text": "…"}],
    "is_hard_negative": false
  }
}
```

`trigger_spans` offsets index the transcript formed by joining `utterances[*].text` with `\n`,
and every span's `text` is a verbatim substring of it (validated at construction). Tactic ids
(15): `impersonation_bank`, `impersonation_gov_police`, `impersonation_telecom_delivery`,
`urgency`, `fear_threat`, `secrecy`, `otp_request`, `credentials_request`, `safe_account`,
`payment_redirect`, `remote_access`, `prize_lottery`, `investment_scam`, `mule_recruitment`,
`verification_ploy`.

## Privacy

Every utterance is scrubbed (phone → `[PHONE]`, card → `[CARD]`, IIN → `[IIN]`, e-mail →
`[EMAIL]`) and the upload script refuses to publish a split that is not a scrub fixed point
(ADR D45). An earlier upload of `ood.jsonl` carried one synthetic 12-digit IIN next to an
invented name; the repaired file replaces it.

## Evaluation hygiene

Held-out rows that anyone has read while debugging are listed in the code repository's
`data/anchors/inspection_ledger.yaml`; the harness reports `(clean)` and `(inspected)` subsets
separately. Please do the same: never tune thresholds or lexicons on `authored_heldout` or
`shift`.

## Limitations

- **Synthetic.** High scores on splits from the training generator largely measure that
  generator's style; on `shift` the shipped model's recall is 0.364 (see the model card).
- One generator family for train/val/test/ood/adversarial; `shift` is small (66 rows, recall
  interval ±0.17).
- The Kazakh was produced by language models and one author; it has not been reviewed by a
  panel of native speakers.
- No job, romance or tech-support scam categories yet.
- **Dual use.** The scam dialogues are textbook scripts; they describe known tactics and add no
  operational capability, but do not use them to write or rehearse scams.
- The generation terms of the providers (Google Gemini API, Anthropic) apply to their outputs;
  a review of those terms is an open item in the code repository's `docs/LEGAL_ASSESSMENT.md`.
