# English recogniser comparison (2026-10-08)

Numbers pasted from `data/asr_capture/english_results.json` / script output. One thread per
engine, 0.3 s leading silence. Moonshine = sherpa "quantized-2026-02-27" (v2) export; parakeet =
CTC head (sherpa ships TDT only in fp32, 452 MB).

## Sets
- **Primary: FLEURS en_us test**, 200 real read utterances ("≤9 s" = 93-row subset every engine
  can take whole).
- Secondary (macOS TTS): `authored` 120 call sentences × 2 of 6 voices (US, GB, AU, IE, IN, ZA),
  clean + phone = 480 rows; `cs_span` = English spans cut from spliced code-switch rows, 40 rows.
- Probe (no English): 200 `single` code-switch rows (Aru) + 75 FLEURS kk_kz + 75 ru_ru.

## Results (WER [95 % CI])
| engine | FLEURS | FLEURS ≤9 s | TTS phone | TTS cs_span | AUC en vs kk/ru (probe pass @90 % en) | word conf / timestamps | native RTF | WASM RTF | MB |
|---|---|---|---|---|---|---|---|---|---|
| vosk small-en-0.15 | 0.231 [0.210,0.254] | 0.242 | 0.138 | 0.105 | 0.982 (2.6 %) | yes / yes | 0.138 | — | 71 |
| vosk en-0.22-lgraph | 0.175 [0.158,0.192] | 0.173 | 0.081 | 0.093 | 0.997 (0 %) | yes / yes | 0.653 | skipped | 214 |
| FastConformer 480 ms transducer int8 | 0.261 [0.225,0.300] | 0.325 | 0.059 | 0.279 | 0.987 (0.3 %) | token prob / yes | 0.071 | 0.387 | 137 |
| same, fp32 (control) | 0.130 [0.115,0.147] | 0.140 | 0.050 | 0.279 | — | token prob / yes | 0.093 | — | 478 |
| FastConformer 480 ms CTC int8 | 0.133 [0.118,0.149] | 0.128 | 0.121 | 0.395 | n/a | no / yes | 0.082 | 0.421 | 132 |
| Moonshine tiny | 0.677 | 0.120 [0.095,0.151] | 0.041 | 0.465 | n/a | no / no | 0.013 | 0.072 | 44 |
| Moonshine base | 0.670 | 0.102 [0.079,0.124] | 0.038 | 0.384 | n/a | no / no | 0.026 | 0.161 | 141 |
| parakeet-tdt_ctc-110m (CTC) | 0.147 [0.127,0.169] | 0.186 | **0.022** | **0.081** | n/a | no / yes | 0.030 | 0.150 | 132 |
| GigaAM-ML 220M | 0.150 [0.135,0.168] | 0.148 | 0.716 | 0.942 | n/a | no / yes | 0.073 | — | 225 |
| GigaAM-ML 600M | **0.114 [0.100,0.127]** | **0.107** | 0.900 | 1.023 | n/a | no / yes | 0.128 | — | 592 |

FLEURS CER: small 0.125, lgraph 0.083, transducer int8 0.186, CTC 0.059, parakeet 0.077, GigaAM
0.056 / 0.046. Native RTF measured under load 3–20 (ranking only). WASM = npm `sherpa-onnx` 1.13.8
single-thread, 60 TTS-phone utterances, peak 203–375 MB, streaming incl. 1.1 s padding.
GigaAM script: FLEURS 100 % Latin; TTS 24 % (220M) / 15 % (600M) Latin → TTS artefact; kk/ru
probe leaks 0.05 % / 0.02 % Latin words.

## The shipped vote (Vosk kk/ru + small-en), share of utterances English wins
| handicap | FLEURS en (true) | probe_cs (false) | FLEURS kk | FLEURS ru |
|---|---|---|---|---|
| 0 | 0.955 | 0.045 | 0 | 0.013 |
| 0.05 | 0.910 | 0.020 | 0 | 0 |
| 0.10 | 0.750 | 0 | 0 | 0 |
| **0.15 (shipped)** | **0.545** | 0 | 0 | 0 |
| 0.25 | 0.145 | 0 | 0 | 0 |
lgraph in the slot at 0.15: 0.560 / 0 — the handicap, not the model, loses English.

## Findings
1. The int8 streaming transducer is broken by encoder quantisation (40 FLEURS rows: enc int8 →
   0.294 / 0.297; enc fp32 → 0.118 / 0.121). The int8 CTC export of the family is fine (0.133).
2. Moonshine (this export) returns nothing for inputs > ~9 s (107/107 FLEURS rows empty; WASM
   throws) and loops on short spans; no confidences, no timestamps.
3. parakeet-110m is the most even: best on every TTS set, level with GigaAM-220M on FLEURS, WASM RTF 0.15.
4. GigaAM-600M is the most accurate on real English.

## Recommendation
- A dedicated English recogniser is probably not needed for transcription if GigaAM replaces kk/ru
  Vosk — but only FLEURS (US read speech) supports that, and TTS flips GigaAM to Cyrillic. Record
  30–50 real Kazakh-accented English utterances first; if GigaAM writes them in Cyrillic, keep a
  dedicated English engine.
- (i) Per-utterance vote: keep small-en as detector, lower handicap to **0.05** (en wins 0.910 of
  FLEURS; false wins 0.020 on Aru TTS, 0 on real kk/ru); re-transcribe English winners with
  **parakeet-110m**. Reject lgraph (no better in the vote, slow, 214 MB).
- (ii) Per-word span vote: only Vosk and the transducer have word confidence + timestamps; vote on
  Vosk words, re-decode English spans with parakeet-110m (cs_span 0.081). Untried: re-quantise the
  transducer encoder (fp16 / partial int8).

## Risks
No real accented or phone-call English; no real code-switched English; cs_span n=40. Streaming
FastConformer drops the first word without leading silence (6/6). sherpa offline CTC gives no
confidences (calibration needs raw logits). Two WASM runtimes on Android are memory-heavy.

## Files
`english/{sentences.jsonl, build_en_set.py, fetch_fleurs.py, fetch_models.py, decode_en.py,
score_en.py, bench_wasm.cjs}`; `data/asr_capture/english_{set.jsonl, fleurs.jsonl, decodes_*.jsonl,
results.json, wasm_bench.jsonl}`; models in `models/asr/<name>/`; audio gitignored.
