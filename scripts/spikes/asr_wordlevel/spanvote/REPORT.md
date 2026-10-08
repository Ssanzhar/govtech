# Span vote — live per-word KK/RU/EN tags from three Vosk models (2026-10-08)

Numbers pasted from `data/asr_capture/spanvote_results.json` / script output. TEST = the 440-row
code-switched set (`cs_set.jsonl`); it was scored only, never fitted or tuned on. Disclosure: one
smoke run of the pipeline on TEST with default parameters happened mid-way; all tuning afterwards
used DEV out-of-fold data only.

**Verdict:** GO for live KK/RU word tags; NO-GO for live English tags (EN recall 0.291
[0.151, 0.426]) — English stays provisional until the endpoint pass.

## Method
- **DEV** (`build_dev_set.py`): 250 sequences spliced from `train`-split word windows (200 kk/ru,
  50 with an English fragment), same conditions as TEST → 900 rows, 2,395 s.
- **Decoding** (`decode_partials.py`): kk 0.42 / ru 0.22 / en 0.15, `SetWords` + `SetPartialWords`,
  0.25 s chunks, every final and partial logged with arrival time. Calling `PartialResult()`
  changes some finals (143 / 1,320 TEST model-decodes differ from `cs_decodes.jsonl`), so all
  methods are compared on the same re-decodes.
- **Calibrator** (`calibrate.py`): one logistic regression per model; target = the word aligns as a
  hit to a reference word of this model's language. Features: logit(conf), conf==1, duration,
  log chars/s, log chars, Kazakh-letter flag, Latin flag, utterance mean conf, best rival's
  utterance mean conf, words/s, rival duration-weighted conf over the word, rival words/s over the
  word, train-only char-trigram kk−ru log-ratio (`word_lid.py`). Out-of-fold DEV log loss kk/ru/en
  0.287 / 0.404 / 0.064 (0.414 / 0.457 / 0.064 without the char model; raw conf 2.46 / 1.84 / 2.84).
- **Vote** (`vote.py`): cut the timeline at every word boundary; span score = duration ×
  (logit p + bias), a missing language scores a fixed logit; 3-state Viterbi with a switch
  penalty; each word emitted once (midpoint span). Tuned on DEV (864 grid points, objective =
  right word + right tag, KK/RU→EN false switches ≤ 1 %): switch penalty 0.25, missing logit
  −2.0, biases 0. DEV: acc 0.890, WER 0.271, EN recall 0.302, false switches 0.0095.
- **Freeze** (`realtime.py`): timeline > N s behind the newest decoded word is locked; N = 1.0.

## TEST (440 rows, 1,000 bootstrap resamples over utterances)
| method | word-lang acc | WER | EN recall | KK/RU→EN |
|---|---|---|---|---|
| shipped utterance vote | 0.570 [0.554, 0.586] | 0.423 [0.394, 0.454] | 0.058 [0.000, 0.189] | 0.001 [0.000, 0.003] |
| raw span max-conf | 0.638 [0.618, 0.658] | 0.368 [0.341, 0.401] | 0.535 [0.387, 0.661] | 0.076 [0.063, 0.088] |
| raw span max-conf, EN −0.15 | 0.650 [0.630, 0.670] | 0.346 [0.320, 0.376] | 0.395 [0.260, 0.514] | 0.047 [0.038, 0.056] |
| **span vote (calibrated + HMM)** | **0.873 [0.859, 0.887]** | **0.260 [0.236, 0.283]** | 0.291 [0.151, 0.426] | 0.007 [0.004, 0.010] |
| calibrated, no HMM | 0.864 [0.850, 0.878] | 0.269 [0.245, 0.292] | 0.279 [0.173, 0.381] | 0.017 [0.012, 0.023] |

Per language, span vote: kk 0.943, ru 0.833, en 0.333 (shipped: kk 0.804, ru 0.346, en 0.069).

| condition/channel | span acc | span WER | shipped acc / WER |
|---|---|---|---|
| single/clean | 0.886 [0.860, 0.909] | 0.280 | 0.575 / 0.398 |
| single/phone | 0.832 [0.800, 0.862] | 0.371 | 0.553 / 0.456 |
| spliced/clean | 0.891 [0.865, 0.918] | 0.175 | 0.574 / 0.404 |
| spliced/phone | 0.879 [0.850, 0.907] | 0.232 | 0.576 / 0.435 |

## Real time (TEST, replayed partials)
| live method | final tag acc | latency mean s [CI] | p50 / p90 s | flicker/word [CI] | words flipped |
|---|---|---|---|---|---|
| shipped (kk partial, then vote) | 0.570 | 1.101 [1.076, 1.127] | 1.14 / 1.953 | 0.068 [0.052, 0.085] | 0.066 |
| raw span max-conf | 0.638 | 1.079 [1.059, 1.102] | 1.14 / 1.92 | 0.142 [0.123, 0.160] | 0.126 |
| span vote, no freeze | 0.873 | 1.049 [1.030, 1.068] | 1.115 / 1.86 | 0.110 [0.094, 0.127] | 0.098 |
| **span vote, freeze 1.0 s** | **0.872** | 1.043 [1.026, 1.062] | 1.115 / 1.86 | **0.099 [0.085, 0.114]** | 0.091 |

Latency (~1.0 s after a word ends) is Vosk's, not the vote's; the last word's latency is
understated (FinalResult forced at end of file). The shipped baseline's low flicker is because it
shows everything as kk until the vote.

## CPU cost of `SetPartialWords` (`cpu_cost.py`, 40 rows, process CPU time)
kk 0.3004 vs 0.2979 (+0.8 %), ru 0.3137 vs 0.3186 (−1.5 %), en 0.5452 vs 0.5415 (+0.7 %),
three models 1.1593 vs 1.158 (+0.1 %). Partial confidences are free; three recognisers cost ~1.16
cores of native CPU (not WASM speed).

## Caveats
86 English reference words on TEST (wide CIs); all audio is macOS TTS and DEV/TEST share voices;
`spliced` changes voice exactly at the switch; calibrators fit on finals, applied to partials; no
per-call prior tested.

## JS port parameters
In `spanvote_results.json → js_port`: normalisation (NFC lowercase, ё→е), clamps (conf
[1e-3, 1−1e-3], duration ≥ 0.03 s), feature order, the three calibrators' coefficients, the char
trigram model (~82 KB JSON), switch penalty 0.25, missing logit −2.0, freeze 1.0 s; rerun the vote
on every partial event.
