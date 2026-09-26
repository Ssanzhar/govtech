---
language:
  - kk
  - ru
license: other
license_name: undecided
tags:
  - fraud-detection
  - scam-detection
  - social-engineering
  - text-classification
  - explainability
  - on-device
library_name: sklearn
pipeline_tag: text-classification
base_model: intfloat/multilingual-e5-base
datasets:
  - sanzh-ts/govtech_ds
---

# Qorğan — phone-scam tactic heads (Kazakh / Russian)

> **Licence not yet chosen by the maintainers.** Until a licence is published here, no rights
> beyond viewing are granted. **The training data is synthetic** — see the limitations below
> before relying on any number.

Calibrated logistic-regression heads that score a phone-call **transcript** for
social-engineering (scam) tactics in Kazakh, Russian and code-switched speech. They sit on top of
[`intfloat/multilingual-e5-base`](https://huggingface.co/intfloat/multilingual-e5-base) (int8
ONNX, [`Xenova/multilingual-e5-base`](https://huggingface.co/Xenova/multilingual-e5-base)) and run
**in the citizen's browser** (the heads are exported to `web/weights.json`). Code:
[github.com/Ssanzhar/govtech](https://github.com/Ssanzhar/govtech).

**This is decision support.** The output is a risk score, verbatim trigger phrases and tactic
tags with templated RU/KK reasons and advice; **a person always decides**. The system never
blocks, reports or hangs up on its own, and it analyses the *caller's script as text* — never
voice, emotion or identity.

## Files

| File | What it is |
|---|---|
| `risk_clf.joblib` | risk head: `CalibratedClassifierCV` (3 × LogisticRegression, Platt) over 774 features |
| `tactic_clf.joblib` | 15 one-vs-rest LogisticRegression tactic heads over the 768-d embedding |
| `metadata.json` | label space, per-tactic thresholds, feature version, **lexicon hashes** |
| `web/weights.json` | the same heads as plain JSON for the browser port (`site/core/`) |
| `lexicon/` | the hard-signal cue and reassurance lexicons the bundle hash-validates |

`.joblib` files are pickles: load them only from this repository, never from an untrusted copy.

## How it works

1. The transcript's head + tail rolling window, prefixed `query: `, is embedded by e5-base
   (int8 ONNX; the server pins `onnxruntime` to the version transformers.js bundles, so server
   and browser embeddings are bit-identical).
2. Six interpretable features are appended: five hard-signal request cues (`secrecy`,
   `otp_request`, `credentials_request`, `safe_account`, `remote_access`), matched with a
   bounded-edit matcher that survives speech-recognition errors, and one *reassurance* flag
   ("we will never ask for your code" near a sensitive term).
3. The risk head outputs a calibrated probability; an alert fires at **≥ 0.59** (hysteresis
   0.59 / 0.49 in the live meter). The tactic heads output 15 tags:
   `impersonation_bank`, `impersonation_gov_police`, `impersonation_telecom_delivery`,
   `urgency`, `fear_threat`, `secrecy`, `otp_request`, `credentials_request`, `safe_account`,
   `payment_redirect`, `remote_access`, `prize_lottery`, `investment_scam`,
   `mule_recruitment`, `verification_ploy`.

The heads were trained on the **browser's own embeddings** (ADR D33). The bundle's lexicon
hashes must match the code's lexicons; always pair a code revision with the model revision
published alongside it.

## Evaluation (threshold 0.59, device embeddings, 2026-09-24)

False-positive rate first — a false alarm on a real bank call destroys trust. Intervals are
Clopper–Pearson 95 %. Source: `docs/eval_report.md` (ADRs D42/D43), harness
`python -m qorgan.eval.run`.

| Split | What it is | FPR [95 % CI] | Recall [95 % CI] | N |
|---|---|---|---|---|
| `test` | same generator (Gemini) as train | 0.000 [0.000, 0.070] | 0.984 [0.916, 1.000] | 115 |
| `authored_heldout` | hand-written, not real calls | 0.000 [0.000, 0.142] | 0.889 [0.653, 0.986] | 42 |
| `ood` | disfluent, out-of-distribution | 0.000 [0.000, 0.049] | 0.932 [0.813, 0.986] | 118 |
| `adversarial` | cue-free paraphrases (scams only) | — | 0.945 [0.884, 0.980] | 109 |
| **`shift`** | **a second generator (Claude) with no access to the corpus** | **0.061 [0.007, 0.202]** | **0.364 [0.204, 0.549]** | 66 |

**Read the `shift` row as the honest number.** Every other split shares its generator with
the training data, so its high recall largely measures one model's writing style. On calls
written by a different author the device model finds about a third of the scams (weakest in
Kazakh and code-switched calls). Both `shift` false positives are rows read during debugging;
on the 64 never-read rows FPR is 0.000 [0.000, 0.112].

Live meter (streaming): false-latch 0.125 [0.027, 0.324] on hand-written legit calls (0.053
on the never-read subset), alert-hit 0.889, median 3 utterances to alert.

## Limitations

- **No real calls.** All training and evaluation data is synthetic or hand-written. A locked
  real-call test set is designed (`docs/DATA_INTAKE.md`) but does not exist yet.
- **Cross-generator recall is low** (0.364); Kazakh and code-switched calls are weakest.
- **Taxonomy gaps:** job, romance and tech-support scams have no dedicated tags.
- **Open weights are an adversary's test bench.** Anyone can probe these heads offline for
  evasions; a stronger model, if one is built, belongs behind an API (ADR D15).
- Thresholds were tuned on validation + hand-written sets only; never on `shift`.

## Intended use and misuse

- **Intended:** helping a person on a call recognise manipulation tactics, on their own
  device, with an explanation they can check; aggregate analysis of *consented* reports.
- **Out of scope:** automated blocking or reporting; covert or operator-level call
  interception; scoring or profiling people; emotion or voice analysis; any use without the
  user knowing an AI produces the verdict. In Kazakhstan the Law "On Artificial Intelligence"
  (No. 230-VIII) requires telling users that a service uses AI and bans emotion recognition
  without consent (see `docs/LEGAL_ASSESSMENT.md` in the code repository — not legal advice).

## Third-party components

| Component | Licence |
|---|---|
| `intfloat/multilingual-e5-base` / `Xenova/multilingual-e5-base` | MIT |
| transformers.js, onnxruntime | Apache-2.0 / MIT |
| Vosk models used by the page's microphone mode | Apache-2.0 |
