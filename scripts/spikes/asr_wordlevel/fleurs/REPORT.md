# Real-speech check: GigaAM-Multilingual vs the shipped Vosk vote (2026-10-08)

Numbers pasted from `data/asr_capture/fleurs_results.json`. 1,362 decodes (681 clean, 681 phone).
**Short answer:** the TTS result holds on real speech, and the gap is larger — on kk/ru GigaAM is
~3× better than the shipped vote on clean audio and 4–5× better through the phone channel.

## Data
- **FLEURS `test`:** first 250 kk_kz, 250 ru_ru, 150 en_us in archive order, no filtering
  (202 / 192 / 128 unique sentences; 68 / 48 / 24 min of read Wikipedia text).
- **Tim2190/kazakh-codeswitch-asr:** CC BY 4.0, ungated; 31 natural KK–RU code-switched clips
  (stand-up, vlogs; 3.7 min).
- Channels: clean and `phone_channel` (300–3400 Hz, G.711 μ-law 8 kHz, 25 dB SNR).
- **KSC2 skipped:** one 78 GB tar.gz in 8.4 GB parts, no separate test archive.
- Digits: num2words supports Kazakh (`kz`), no rows dropped; 49/250 kk, 58/250 ru, 28/150 en
  references contain digits, spelled as cardinals; digit-free WER reported too.

## Results — WER [95 % bootstrap CI] · CER
| set / channel | GigaAM 220M | GigaAM 600M | Vosk vote (shipped) | Vosk, true-language model |
|---|---|---|---|---|
| kk clean | 0.074 [0.063, 0.086] · 0.022 | 0.065 [0.055, 0.074] · 0.021 | 0.217 [0.197, 0.239] · 0.075 | 0.211 · 0.073 |
| kk phone | 0.087 [0.076, 0.100] · 0.025 | 0.073 [0.063, 0.084] · 0.023 | 0.424 [0.385, 0.466] · 0.257 | 0.416 · 0.282 |
| ru clean | 0.070 [0.059, 0.082] · 0.021 | 0.058 [0.047, 0.071] · 0.020 | 0.203 [0.185, 0.221] · 0.057 | 0.194 · 0.052 |
| ru phone | 0.085 [0.073, 0.099] · 0.025 | 0.061 [0.050, 0.073] · 0.021 | 0.306 [0.285, 0.327] · 0.112 | 0.278 · 0.094 |
| en clean | 0.130 [0.114, 0.148] · 0.047 | 0.100 [0.088, 0.114] · 0.037 | 0.592 [0.522, 0.660] · 0.500 | 0.221 · 0.119 |
| en phone | 0.180 [0.160, 0.201] · 0.076 | 0.121 [0.107, 0.137] · 0.048 | 0.792 [0.743, 0.843] · 0.703 | 0.555 · 0.424 |
| KK–RU code-switch clean (n=31) | 0.186 [0.136, 0.237] | 0.189 [0.142, 0.239] | 0.300 [0.249, 0.349] | 0.300 (kk model) |
| KK–RU code-switch phone (n=31) | 0.221 [0.172, 0.276] | 0.300 [0.218, 0.400] | 0.393 [0.314, 0.478] | 0.369 |

Digit-free WER (220M / 600M / vote / true-language model): kk clean (n=201) 0.053 / 0.045 /
0.202 / 0.194; ru clean (n=192) 0.046 / 0.031 / 0.185 / 0.175. Code-switch clips against the
dictionary-form reference: 0.074 / 0.076 / 0.234 / 0.234 clean (much of GigaAM's literal-reference
error is spelling normalisation: кароче→короче, влажение→вложение).

**Vote's language choice** (share where the true language's model won): clean kk 0.988, ru 0.964,
en 0.520; phone kk 0.880, ru 0.912, en 0.347 (phone: 28/250 kk → ru model; English 74/150 → ru,
24 → kk). Code-switch clips: kk model won 31/31 clean, 30/31 phone.

**English script.** On real speech GigaAM writes English in Latin: 1.000 / 1.000 of words clean
(220M / 600M), 0.996 / 0.999 phone. (The Cyrillic transliteration seen earlier occurs on macOS TTS
audio — a TTS artefact.) The vote's English output is 37–44 % Cyrillic words.

**Word tagger on GigaAM output** (share tagged with the utterance's language): kk 0.956 / 0.953
clean, 0.955 / 0.950 phone; ru 0.971 / 0.970 clean, 0.970 / 0.970 phone → ~4–5 % false switch
points in monolingual Kazakh is the floor.

**Cue survival:** no reference contains a hard-signal cue; no engine false-fired. Uninformative.

**Speed** (single-thread native, median per-utterance RTF, kk clean): 220M 0.075, 600M 0.180,
three Vosk summed 0.275, one Vosk 0.063. Across sets three-Vosk 0.28–0.83, 220M ~0.08–0.10.
(Laptop shared and slept mid-run; medians reported; browser WASM is slower.)

## Conclusions
1. The result holds on real speech, and the gap is larger (kk/ru clean 0.07 vs 0.21; phone kk
   0.087 vs 0.424; natural code-switch 0.186 vs 0.300).
2. The per-utterance vote breaks English: the 0.15 handicap hands 48 % of clean English utterances
   to kk/ru models (0.592 vs 0.221 for the English model alone).
3. GigaAM 220M is the pick for calls: 600M wins monolingual FLEURS by 1–2.5 pts but collapsed on
   5/31 phone-channel code-switch clips (0.300 vs 0.221; n=31 — look before reconsidering) and is
   2.4× slower.
4. 220M is cheaper than what ships (~0.08 RTF vs the three-Vosk total).

## Caveats
GigaAM's fine-tuning data is not listed — if it saw FLEURS train, FLEURS flatters it (our kk 0.074
is close to the card's 5.2 %); the 31 code-switch clips are the independent check and agree. Not
scam calls; FLEURS is read speech; phone channel simulated. Speeds relative only.

## Files
`fleurs/{fetch.sh, prepare.py, decode.py, score.py}`; `data/asr_capture/fleurs_{set.jsonl,
decodes_clean.jsonl, decodes_phone.jsonl, results.json}`; audio in gitignored
`data/asr_capture/_wav/fleurs/`.
