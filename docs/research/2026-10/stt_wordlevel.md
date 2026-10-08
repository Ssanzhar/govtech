# Per-word, real-time KK/RU/EN decisions and better English: research report (2026-10-08)

Builds on `docs/research/2026-10/stt.md` (2026-10-07), which it does not repeat. Every claim is marked **[V]** (verified today, with the source) or **[U]** (unverified: an estimate or a claim I could not check). "Our data" means `data/asr_capture/tri_val*.jsonl` (val split, so reading it is allowed).

## 0. Headline findings

1. **A new joint model makes per-word language come almost free.** Sber's `ai-sage/GigaAM-Multilingual` ctc (220M, **MIT**, released 2026-07-14) is a Conformer with a **character-level CTC head** over 70 symbols: `a–z`, the Russian Cyrillic letters and `әғқңөұүһі`. Its card reports FLEURS WER **kk 5.2, ru 4.4, en 12.2** and CV **kk 17.2, ru 7.1, en 26.0** [V, https://huggingface.co/ai-sage/GigaAM-Multilingual]. A sherpa-onnx NeMo-CTC export exists as a single `model.int8.onnx` of **224.8 MB** with `tokens.txt` (71 entries), MIT. Its card gives RTF ≈0.06 at int8 on Apple silicon with 4 threads [V, https://huggingface.co/fussraider/GigaAM-Multilingual-sherpa-onnx-ctc]. Because the model writes characters, every word comes out in a script you can read. Latin letters mean EN. Cyrillic with `әғқңөұүһі` means KK. Plain Cyrillic stays ambiguous and goes to a small text classifier. Limit: the model is not streaming.
2. **Combining three monolingual recognisers word by word has no published method that works across different languages.** ROVER and confusion-network combination assume the systems compete for the *same* words. When the vocabularies don't overlap there is no agreement to vote on, so only confidence-based selection is left. The methods that do work for code-switching (multi-graph decoding, the union of monolingual graphs) share **one acoustic model** inside one beam search [V, https://arxiv.org/pdf/1906.07523]. That rules them out for three independent Kaldi models. Per-span selection over calibrated confidences is the best you can do with today's stack. Its ceiling is set by the LM-coherence problem (§A).
3. **Audio LID cannot break ties per word.** The VoxLingua107 ECAPA model covers kk, ru and en (training data: kk 78 h, ru 73 h, en 49 h) [V, https://cs.taltech.ee/staff/tanel.alumae/data/voxlingua107/]. An ONNX export runs 83–87 MB fp32, Apache-2.0. Its own duration sweep says "clips under 5 s are the noisy regime" [V, https://huggingface.co/christopherthompson81/voxlingua107-lid-onnx]. Words last 0.3–1 s. Use audio LID, if at all, for a per-call prior.
4. **Text-side per-word LID handles about two thirds of Kazakh tokens deterministically.** On our data, **67.8 %** of Kazakh reference tokens contain a Kazakh-only letter (1,841/2,717). Russian has 0/4,922 and English 0/3,725. The KK Vosk model, run on *Russian* speech, still writes a Kazakh letter in **6.9 %** of its words (350/5,067). So a Kazakh letter is strong evidence, not proof. (Computed today from `tri_val*.jsonl`.)
5. **Best English per MB:**
   - *Streaming, separate slot:* NVIDIA FastConformer hybrid streaming EN (114M, CC-BY-4.0). LibriSpeech test-other **5.7** (transducer, 480 ms lookahead) or **6.7** (CTC) [V, https://huggingface.co/nvidia/stt_en_fastconformer_hybrid_large_streaming_multi]. It ships as sherpa-onnx int8 tarballs of **~100–106 MB** [V, sherpa-onnx `asr-models` release, `gh release view`].
   - *Small:* Moonshine. The non-streaming v1 tiny is a **29.9 MB** quantized sherpa tarball, with LS-other 11.71 and Earnings-22 21.25. Moonshine-streaming tiny (34M, MIT) scores LS-other 12.09 and Earnings-22 20.27 [V, sources in §D].
   - *0 extra MB:* if you adopt GigaAM-Multilingual, its English (FLEURS 12.2) comes inside the joint model.

## A. Combining several monolingual recognisers word by word

**ROVER (Fiscus 1997)** aligns the 1-best outputs into a word transition network and scores each slot as α·(vote count) + (1−α)·(confidence) [V, NIST SCTK doc http://www1.icsi.berkeley.edu/Speech/docs/sctk-1.2/rover.htm]. With KK, RU and EN models on one stretch of audio, the three words in a slot almost never match, so the vote term is zero. ROVER then reduces to "highest confidence per slot", which is your current rule moved from utterances to words. Confusion-network combination (Evermann & Woodland 2000) has the same assumption [U: classic result, not re-fetched].

**Code-switching evidence:**
- *Multi-graph decoding* (Yılmaz et al., Interspeech 2019; Frisian–Dutch): parallel monolingual and bilingual subgraphs compete in **one shared beam**, with a single acoustic model. It gives a 10.7 % relative WER reduction on Dutch utterances with no loss on Frisian or code-mixed speech [V, https://arxiv.org/pdf/1906.07523]. Not reproducible with three Kaldi models whose phone sets and acoustic models differ.
- *Confidence-based selection among language-biased systems* (Slottje et al. 2021, Mandarin–English): selecting whole transcripts gives a **19 % relative** improvement for semi-supervised labelling [V, https://arxiv.org/abs/2106.07699]. This is utterance-level selection, the same thing you do today.
- *Kazakh–Russian*: one bilingual model beat monolingual models on embedded Russian words, 59.53 → 44.84 WER (prior report; CEUR Vol-2590 short13) [V, previous report].
- I found **no published per-word combination result for kk–ru, tt–ru or uk–ru surzhyk** with separate recognisers [U: absence after two extended searches].

**What breaks if you mix words from different models:**
1. *LM context bleed.* Each Kaldi graph forces its own language onto the whole utterance. Near a switch point the right model's words are degraded too, because its LM pulled them toward in-language words. Our data shows it: on Russian speech the KK model writes "банкі мұзы фиксингі" for "банка мы зафиксировали".
2. *Segmentation mismatch.* One KK word can match two RU words ("қызметіненмін" against "из службы"). So align by **time spans**, not word slots. Vosk gives word `start`/`end`/`conf` in finals and, with `setPartialWords`, in partials. Partial confidences are real MBR posteriors over the partial lattice (`recognizer.cc` `PartialResult` calls `MinimumBayesRisk` → `GetOneBestConfidences`) [V, https://github.com/alphacep/vosk-api/blob/master/src/recognizer.cc]. Computing them costs CPU on every partial.
3. *Confident garbage.* Already measured: EN "the grass to you mean yes" at 0.71. Kaldi MBR posteriors are known to overestimate the probability that a word is correct, which is why Kaldi ships a calibration step (§E).
4. *Raw confidences from different models are not comparable.* Each is a posterior inside that model's own lattice; a narrow, wrong-language lattice is confident by construction.

**Practical algorithm (no new models):**
1. Cut the timeline at the union of all word boundaries from the three streams.
2. For each elementary span, score each language as the sum, over that language's words overlapping the span, of overlap duration × logit(calibrated p_correct).
3. Run a 3-state HMM/Viterbi over spans. The emissions are those scores. The transition cost is a switch penalty, with no penalty at a pause longer than 150 ms.
4. Emit each chosen language's words whose midpoints fall inside its run.

On partials, rerun steps 1–4 on the uncommitted tail every partial event (~100–300 ms). Freeze spans once they are 1 s behind the newest audio, so the display stops flickering. This is hysteresis, the same idea as the meter's.

## B. Language ID as a tie-breaker

| Option | Covers kk? | Size | Short-segment behaviour | Verdict |
|---|---|---|---|---|
| SpeechBrain ECAPA VoxLingua107 (21M) | yes (`kk: Kazakh` in `label_encoder.txt`) [V, https://huggingface.co/speechbrain/lang-id-voxlingua107-ecapa] | 84.5 MB ckpt; ONNX 85–87 MB fp32 [V, https://huggingface.co/beginning-ai/speechbrain-lang-id-voxlingua107-ecapa-onnx]; int8 ~22 MB [U] | "under 5 s noisy regime" [V]; trained on 1–20 s segments [V, https://arxiv.org/html/2406.09290v1] | per-call or per-utterance prior only |
| MMS-LID-126/256 (1B) | yes (`kaz`, `rus` in `langs.txt`) | 3.86 GB, **CC-BY-NC** [V, https://huggingface.co/facebook/mms-lid-126] | — | dead |
| Whisper LID token (tiny 39M) | yes | sherpa `whisper-tiny` tarball 116 MB [V, release] | the encoder always processes a 30 s padded window, so cost is per call rather than per word [U: architectural] | dead for per-word use |
| sherpa-onnx spoken-LID | uses multilingual Whisper models [V, sherpa docs via search] | as above | as above | dead |
| fastText `lid.176.ftz` (text) | yes (`kk`) | **917 kB**, CC-BY-SA-3.0 [V, https://fasttext.cc/docs/en/language-identification.html] | trained on documents; single Cyrillic words are unreliable [U] | weak feature only |
| GlotLID (text) | yes | 1.7 GB [V, HF API] | — | dead |
| **Own per-word text LID** | built for it | <1 MB | script rule decides 67.8 % of KK tokens [V, our data]; train a char-n-gram NB on the `words.txt` of the KK and RU models plus our corpus, and add the KSC2 rule ("a Russian stem with a Kazakh suffix counts as Russian") | **use** |

Text LID has a known trap. Kazakh–Russian studies find that off-the-shelf LID and letter heuristics "over-label … as mixed" because loanwords look like switches. The deciding factor is where the annotation draws the loanword/switch boundary, not which model you use [V, https://arxiv.org/abs/2608.00581]. Decide our convention once ("банк", "карта", "код" inside Kazakh = KK), write it into the labelling guide, and score against it.

## C. Joint models that emit language per token

| Model | Tokens / language signal | Size (int8) | Streams? | Quality | Licence |
|---|---|---|---|---|---|
| **GigaAM-Multilingual ctc 220M** | 71 characters (Latin, Cyrillic, KK letters) → script gives the language | **224.8 MB** single file [V] | no; offline CTC (buffered re-decode possible) | FLEURS kk 5.2 / ru 4.4 / en 12.2; CV 17.2 / 7.1 / 26.0; internal kk 18.8, ru 7.6 [V] | **MIT** [V] |
| `nur-dev/asr-220m-kk-ru-en` (2026-10-05) | fine-tune of GigaAM-Multilingual, CTC + trigram LM + mBERT punctuation; word timestamps | 441.7 MB checkpoint (PyTorch) [V] | buffered: 12 s window re-decoded every 0.64 s [V] | FLEURS lexical kk 11.46 / ru 9.61 / en 13.26 (different normalisation from Sber's); at 10 dB noise 10.99 / 9.27 / 14.14 [V] | gated, "component licenses" (GigaAM MIT, OpenMDW-1.1, mBERT Apache) [V card; NOTICE gated, U] |
| `nur-dev/realtime-streaming-asr-kk-ru-en` (120M TDT) | 4096 trilingual BPE | 457 MB fp32 encoder [V]; ~115 MB int8 [U] | yes, cache-aware | FLEURS 12.19 / 14.24 / 17.43 at 560 ms (measured by nur-dev) [V] | **CC-BY-NC-4.0**; card: "For commercial licensing … contact the author", Nurgali Kadyrbek, kadyrbeknurgali@gmail.com [V, https://huggingface.co/nur-dev/realtime-streaming-asr-kk-ru-en] |
| `nvidia/stt_kk_ru_fastconformer_hybrid_large` | SentencePiece-1024, kk+ru | ~115–130 MB [U] | no | see prior report | CC-BY-4.0 |
| sherpa `nemo-fast-conformer-ctc-be-de-en-es-fr-hr-it-pl-ru-uk-20k-int8` | joint BPE, **ru+en, no kk** | 102.3 MB tar [V, release] | no | — | — |

**Deriving word language from a joint model:**
- *Character CTC (GigaAM):* a word with any Latin letter → EN. Any `әғқңөұүһі` → KK. Otherwise, run text LID with a per-call prior. A Kazakh word without a Kazakh-only letter ("мен", "бе") is still decoded with the Kazakh *acoustics*. Only the label is uncertain, and a label error on such words costs little downstream, because cue matching runs on the text.
- *BPE models:* tag each token by the language whose training text uses it most, and merge to words by majority. Your cue matcher only needs the text.

**Browser feasibility.**
- sherpa-onnx WASM is single-threaded (prior report).
- GigaAM 220M int8 RTF there is **unknown**. Scaling the 0.06 figure (4 threads, Apple silicon) to one thread and a WASM-SIMD penalty of 2–4× gives **~0.4–0.9 on a mid-range laptop** [U]. That is affordable once per endpointed utterance. Re-decoding a 12 s buffer every 0.64 s, as nur-dev does on an L40 GPU, is not affordable.
- So GigaAM is a **second-pass, per-utterance** decoder. Live partials still need a streaming first pass.
- Memory: the 225 MB weights plus sherpa's `INITIAL_MEMORY=512MB` [V, prior report] are fine on a laptop and risky on a 3–4 GB phone [U].
- Provenance: Sber is a Russian state-owned bank. MIT is permissive, but a KZ GovTech procurement review may ask about it [U: policy judgement].

## D. English options

| Option | Size | Streaming | Browser runtime | LS other | Accented / phone proxy | Licence |
|---|---|---|---|---|---|---|
| Vosk `small-en-us-0.15` (today) | 39.3 MiB | yes | Vosklet | — (clean 9.85, TED 10.38) [V, https://alphacephei.com/vosk/models] | none published | Apache-2.0 |
| Vosk `en-us-0.22-lgraph` | 124.5 MiB [V, model-list.json] | yes | Vosklet (lgraph is standard Kaldi; loading not tested [U]) | — (clean 7.82, TED 8.20) [V] | callcenter 29.78 is for the *big* 0.22 [V] | Apache-2.0 |
| Moonshine v1 tiny (27M) | 29.9 MB sherpa quantized tarball [V, release]; onnx-community int8 encoder 7.9 MB + decoder 20.2 MB [V, HF API] | no (VAD segments) | sherpa WASM or transformers.js | 11.71 | Earnings-22 21.25, AMI 22.77 [V, https://arxiv.org/html/2410.15608] | MIT |
| Moonshine v1 base (61M) | 111.3 MB sherpa quantized tarball [V] | no | same | 8.18 | Earnings-22 17.65, AMI 17.79 [V] | MIT |
| **Moonshine streaming tiny / small** (34M / 123M; Jan 2026) | published `.ort` files are fp32: tiny ≈ 120 MB (decoder 96.6 MB), small ≈ 460 MB [V, https://huggingface.co/moonshine-ai/moonshine-streaming]; int8 ~35 / ~125 MB [U] | yes (80 ms lookahead) | `@moonshine-ai/moonshine-wasm` 0.1.5, MIT, 13.2 MB wasm. **Threads + SAB by default**; a `MOONSHINE_WASM_SINGLE_THREAD` build exists [V, npm tarball README]. "Word-timestamp extras" are mentioned [V]; confidences not found [U] | tiny 12.09 / small 6.78 | Earnings-22 20.27 / 13.53; AMI 19.03 / 12.54 [V, https://huggingface.co/moonshine-ai/moonshine-streaming-small] | MIT |
| **NVIDIA FastConformer hybrid streaming EN** (114M) | sherpa int8 tarballs: transducer 102.8–105.9 MB, CTC 99.5–102.6 MB [V] | yes (0 / 80 / 480 / 1040 ms) | sherpa WASM (online transducer/CTC); token timestamps + `ys_probs` [V, `online-recognizer.h`] | transducer 7.0 / 6.4 / 5.7 / 5.4; CTC 8.4 / 7.8 / 6.7 / 6.2 [V] | none on card | CC-BY-4.0 |
| parakeet-tdt_ctc-110m | 104.3 MB int8 tarball [V] | no | sherpa offline | 5.2 | AMI 15.88, Earnings-22 12.42, GigaSpeech 10.52 [V, https://huggingface.co/nvidia/parakeet-tdt_ctc-110m] | CC-BY-4.0 |
| Kroko EN community zipformer | 57.3 MB tarball [V] | yes | sherpa | not on card | — | **CC-BY-SA** (share-alike on the model) [V, https://huggingface.co/Banafo/Kroko-ASR] |
| sherpa streaming zipformer EN 2023 | 68 MB int8 encoder [V, sherpa docs] | yes | sherpa | LibriSpeech-only training [V] → poor on accents [U] | — | Apache |
| Kyutai STT 1B en/fr | 2.36 GB [V] | yes | no | — | — | dead |
| distil-whisper small.en | 353 MB fp32 ONNX encoder [V] | no | transformers.js | — | — | dead (size, no streaming) |
| alphacep streaming EN zipformer | **does not exist**: the alphacep HF org has only ru/bn/uz/tajik streaming models [V, HF API] | — | — | — | — | — |

The closest public proxies for "accented speakerphone" speech are Earnings-22 (accented, call audio) and AMI (far-field). On both, **parakeet-110m (12.42 / 15.88)** and **Moonshine-streaming-small (13.53 / 12.54)** lead. Moonshine tiny is roughly 20. None of these models has been tested on KZ-accented English [U].

**Two WASM runtimes on one page** (Vosklet plus sherpa-onnx or moonshine-wasm): each Emscripten module owns its own `WebAssembly.Memory` and runs in its own worker. No shared heap is needed, so they coexist [U: architectural, not tested]. The costs are memory and CPU:
- sherpa reserves 512 MB up front.
- Vosklet already runs three pthread workers.
- On Android WebView only the single-thread builds work (sherpa, or moonshine-wasm's single-thread flag); Vosklet doesn't.

Measure peak RSS in Playwright before committing.

## E. Replacing the 0.15 handicap

- **Kaldi's own answer:** `steps/conf/train_calibration.sh` trains a per-word **logistic regression** on these features: logit of the MBR posterior, log word length in characters, log lattice depth at the word, log frames-per-character, and (in the data prep) log unigram probability [V, https://github.com/kaldi-asr/kaldi/blob/master/egs/wsj/s5/steps/conf/train_calibration.sh]. Microsoft's MaxEnt calibration adds constraints on the word distribution [V, https://www.microsoft.com/en-us/research/?p=158883]. Both calibrate *within* one model. For our case, **fit one calibrator per model**, with target = "this word is correct and in the right language", against time-aligned references.
- **Features Vosk exposes** (no lattice depth, no unigram probability through Vosklet):
  - `conf`, duration, characters, chars/second
  - has-KK-letter, Latin-only
  - whether the word appears in the other models' `words.txt`
  - the competing models' calibrated scores on the same span
  - partial vs final
- **Then a per-call prior.** Keep Dirichlet counts of the languages of finalized words and add log π_lang to every span score. Seed it with a small uniform pseudo-count, so one EN utterance in a KK call has to earn its way.
- **Calibrate per model, then choose by context.** This replaces the fixed handicap and keeps the hysteresis the team already uses for the meter. No published system does exactly this for KK/RU/EN; it is standard calibration plus HMM smoothing [U: synthesis, not a cited method].
- **Calibration data:** `val` only (ADRs D35/D43); cross-validate by dialogue.

## Recommendations

**(1) Architecture for per-word, real-time decisions**

- **Pass 1 (live, while the person is talking):** keep Vosklet. Upgrade KK to `small-kz-0.42` (56.9 MiB). Turn on `setWords(true)` and `setPartialWords(true)` for all three models. Run the time-span HMM vote from §A on partials, using per-model calibrators (§E), the per-call prior and the text-LID/script features from §B. Freeze spans after 1 s. Display: words greyed while provisional and tagged KK/RU/EN.
- **Pass 2 (at the Vosk endpoint):** decode the utterance audio once with **GigaAM-Multilingual 220M int8 in sherpa-onnx WASM** and replace pass 1's text for that utterance. Word language comes from script + text LID + the call prior. The cue matcher and meter consume pass 2 only, so scores never depend on flickering partials.
  - Budget: 225 (GigaAM) + 57 (KK) + 44 (RU) + 39 (EN) ≈ **365 MB**. To get to ≤300 MB, drop the RU and EN Vosk models from pass 1. The KK model already writes Russian words acceptably on Russian speech; our data has "здравствуйте меня зовут алина … службы безопасности". That leaves 225 + 57 ≈ **282 MB**, with live partials less accurate for RU/EN until the endpoint.
- **Fallback** (GigaAM too slow or too heavy on the device, or rejected on provenance): pass 1 alone. For English, replace Vosk-small-en with the NVIDIA streaming FastConformer CTC 480 ms in sherpa WASM (~100 MB).
- **Android WebView:** sherpa only, single-thread, which means GigaAM as the per-utterance decoder plus a VAD.

**(2) Best English option, with numbers**

- **If you adopt the joint model:** GigaAM's built-in English, FLEURS 12.2 / CV 26.0, at **+0 MB** [V].
- **If you keep a dedicated English slot:** NVIDIA FastConformer hybrid streaming EN, int8 ~100–106 MB, LibriSpeech test-other **5.7** (transducer, 480 ms) or **6.7** (CTC, 480 ms), CC-BY-4.0, sherpa single-thread WASM [V]. That compares with Vosk small-en's test-clean **9.85**; Vosk publishes no test-other figure.
- **Smallest credible option:** Moonshine v1 tiny, 29.9 MB, LS-other 11.71, Earnings-22 21.25, MIT [V]. Prefer parakeet-110m (Earnings-22 12.42) if streaming is not needed.

**(3) Three ranked experiments**

**Word timings must be recaptured.** The jsonl stores only `text` plus a mean `confidence` per model. No word arrays, and the WAVs are not kept (`_wav` holds 2 files). `capture_tri.py` already calls `SetWords(True)` on EN, so storing `result` arrays (and partials via `SetPartialWords`) is a small change. The TTS audio is resynthesised (`say` voices Milena/Aru/Samantha). The current "mixed" rows are Kazakh sentences read by the *Russian* voice, not intra-sentence switches, so a real code-switch set has to be built.

1. **Recapture + calibrated span voting on the current stack (12–16 h).**
   - Recapture 1,047 val utterances with word arrays and partials (~3 h compute).
   - Build ≥150 intra-sentence switch utterances by synthesising KK, RU and EN chunks with matching voices and splicing them with 80–200 ms gaps (~4 h).
   - Fit the three per-model calibrators (5-fold by dialogue) and the HMM switch penalty (~5 h).

   **Produces:** word-level language accuracy and per-language WER for span voting vs today's utterance vote; English recall on `tri_val_en` (today 65.5 %) at ≤1 % KK/RU utterances lost to EN; flicker rate (tag changes per committed word); the partial-decode CPU cost of `setPartialWords`.
2. **GigaAM-Multilingual 220M int8 as the per-utterance decoder (10–16 h).** Run it in Node `sherpa-onnx` first, then in the WASM build under Playwright, on the same 1,047 + CS set and FLEURS kk/ru/en subsets degraded with the telephone chain.

   **Produces:** WER kk/ru/en/CS; word-language accuracy from script + text LID; cue survival and the 0/160 false-fire gate; **single-thread WASM RTF and peak memory** on the laptop. This decides go/no-go for the 282 MB design.
3. **English shoot-out (6–10 h).** Compare Vosk small-en, the NVIDIA streaming FastConformer CTC 480 ms int8, Moonshine v1 tiny and GigaAM's English on `tri_val_en` (165 utterances) plus 300 FLEURS `en_us` utterances through the phone chain. Also run each model on the KK/RU val audio to measure **false-English confidence**.

   **Produces:** EN WER per model, the EN false-win rate under calibrated voting, and size/RTF. This picks the English slot.

**(4) Dead ends**

- Word-level ROVER / CNC voting across languages (nothing to vote on).
- Multi-graph / union-graph decoding with three separate Kaldi acoustic models.
- Per-word audio LID: ECAPA under 5 s is noisy; MMS-LID is 1B and CC-BY-NC; Whisper LID costs a 30 s window per call.
- GlotLID (1.7 GB).
- fastText as the main per-word decider (document-level model).
- Kyutai STT 1B; distil-whisper small.en.
- An alphacep streaming EN zipformer (none exists).
- Kroko community models (share-alike licence on the model).
- LibriSpeech-only sherpa EN zipformers, for accented phone speech.
- `nur-dev/realtime-streaming-asr-kk-ru-en` until the author relicenses.
- Re-decoding a 220M model on a sliding window every 0.6 s in single-thread WASM.
- A fixed per-language handicap as a long-term rule.
