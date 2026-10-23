# Android text core: spike (2026-10-21)

**Goal:** a native Kotlin core that NITEC could embed in eGov Mobile without a WebView. It is the
on-device decision support of `site/core/*.js`, ported to Kotlin and checked against the same
golden fixtures. It is text only: speech-to-text is not part of this spike. As on the web, a person
always decides. The core never blocks, reports or hangs up, it stores nothing, and no text or audio
leaves the process.

```
android/qorgan-core/
  core/   pure Kotlin/JVM, tested on the Mac (JUnit 4) against tests_js/fixtures/*
  aar/    Android library built from the same sources + onnxruntime-android + DJL tokenizer-native
```

## Status per component (measured on the Mac with JUnit 4)

| Component | Kotlin | Fixture | Result |
|---|---|---|---|
| LR heads + calibration, cue features, reassurance | `session/Heads.kt`, `Scorer.kt`, `cue/Lexicon.kt` | `parity.json` (29 cases, recorded embeddings) | risk max \|Δ\| **5.2e-8**; tags, weights and spans identical |
| Explanation (templated RU/KK) | `session/Explainer.kt` | `parity.json` | **58/58** identical (role slot of ADR D63 not ported) |
| Meter, hysteresis, rolling window, evidence, advice | `session/Meter.kt`, `LiveSession.kt`, `Advice.kt` | `parity.json` | **29 sessions, 128 turns** identical; meter max \|Δ\| 4.6e-6 |
| Cue matcher (bounded edit, ADR D39) | `cue/CueMatcher.kt` | `cue_match.json` | **3200/3200** cases (142 with a span), 0 mismatches |
| PII scrubber (ADR D66) | `scrub/*.kt` | `scrub.json` | **102/102**, idempotent |
| Tokenizer + int8 ONNX embedder | `embed/DjlTokenizer.kt`, `OnnxEmbedder.kt` | `parity.json` embeddings (`embed_backend: onnx`) | cosine **1.000000**, max \|Δcomponent\| 8.9e-8, risk \|Δ\| 1.65e-7; 145 ms/transcript |
| Runtime gate, 200 transcripts | `gate/RuntimeGate.kt` | `runtime_gate.json` | vs Python: **0/200 flips**, p95 \|Δrisk\| 0.000, tag Jaccard 1.000, 60–71 ms/transcript (Mac M-series, JVM) |
| Same gate vs **browser** embeddings | ditto | `runtime_gate_browser.f32` | cosine mean 0.9822 / min 0.9493; **4/200 decision flips**, p95 \|Δrisk\| 0.091, max 0.572 (see the risk below) |

Test run: 6 suites, **16 tests, 0 failures, 0 skipped** (`./gradlew :core:test -PruntimeGate=true`).

Tokenizer decision: **DJL `ai.djl.huggingface:tokenizers` 0.33.0**. It is a JNI binding of the
same Rust HF `tokenizers` the Python server uses, and it reads the model's `tokenizer.json`
directly, including the XLM-R precompiled normaliser. It reproduces the server bit for bit. 0.33.0
is pinned because it is the only DJL release with an Android build (`ai.djl.android:tokenizer-native`).
Alternatives considered:
- a pure-Kotlin Unigram tokenizer: no native code (saves 8 MB per ABI), but the precompiled
  charsmap trie must be ported. Its acceptance test would be token-id equality with DJL on the gate.
- onnxruntime-extensions' SentencePiece op: it needs the `.model` file, not `tokenizer.json`.

ONNX Runtime is pinned to **1.21.0**, the version `pyproject.toml` pins (ADR D32). With it, the JVM
is bit-identical to the server.

## Public API (sketch for NITEC)

```kotlin
val core = QorganAndroid.load(context, modelFile, tokenizerFile)   // or QorganCore.fromJson(cfg, weights, embedder)
val call = core.newSession(locale = "kk")                          // ru | kk | en; one per call, memory only
val a: Assessment = call.feed(utteranceText, asrConfidence = 1.0)
// a.score 0–100, a.band LOW|MEDIUM|HIGH|CRITICAL, a.latched (hysteresis 0.59/0.49, armed from turn 3),
// a.windowRisk, a.tactics [Tag(id, weight)], a.hardSignalIds, a.newEvidence [Span(text,start,end)],
// a.recommendation (advices / verification questions, from the reviewed YAML via qorgan-config.json)
core.analyze(transcript, "ru")   // one-shot: ScoreResult + Explanation + flagged
core.scrub(reportDraftText)      // PII scrub before the citizen reviews a report
core.advance(state, text)        // pure variant: immutable SessionState in, (state, update) out
```

Language does not enter the model, because e5 is multilingual. The locale only selects the advice
and explanation texts. All model content (cues, reassurance terms, advice, tactic names, templates,
thresholds) is read from `site/core/qorgan-config.json` and `site/models/weights.json`, which a
Gradle task copies into the AAR's assets at build time. Nothing is duplicated in code. The scrubber
rules are code, as they are in the JS port. `CachingEmbedder` keeps a bounded in-memory LRU, because
each turn re-embeds the window's utterances. It is never persisted.

## Sizes

| Item | Size |
|---|---|
| `aar-release.aar` (classes 206 KB + config 48 KB + weights 279 KB) | **341 KB** |
| `libonnxruntime.so` + JNI, arm64-v8a (onnxruntime-android 1.21.0) | 19.3 MB (uncompressed) |
| `libdjl_tokenizer.so`, arm64-v8a | 8.1 MB |
| Embedder `model_quantized.onnx` (not in the AAR by default) | 266 MB |
| `tokenizer.json` (not in the AAR by default) | 17 MB |
| Instrumented-test APK, all 4 ABIs, unminified | 116 MB (x86 / x86_64 / armv7 natives included) |

A host app should set `abiFilters "arm64-v8a"` and exclude `native/lib/**`: DJL's JVM jar carries
about 40 MB of desktop tokenizer binaries, and the AAR's own packaging already drops them for its
test APK. It should also exclude `com/sun/jna/**` (pulled in by `ai.djl:api`). The 266 MB model
cannot ship inside the base APK. NITEC has to choose between Play Asset Delivery (install-time or
on-demand pack) and its own CDN download with a hash check. `QorganAndroid.load(context, modelFile,
tokenizerFile)` takes whichever path results. `-PqorganBundleModel=true` is meant to put the model in the AAR
for demo builds only. That path has not been built yet.

## The device-drift risk (read this before trusting a phone's numbers)

The shipped heads were trained on the **browser's** embeddings: `weights.json` says
`embed_backend: device` (ADR D33). Native ONNX Runtime, as used on the server, the JVM and Android,
is only a proxy of that runtime. On the 200-case gate, native vs browser embeddings have cosine
0.982 mean / 0.949 min and give **4/200 different decisions**:

| Case | Browser risk | Native risk |
|---|---|---|
| `ood_neg_legit_bank_call_mixed_3` (a **legitimate** call) | 0.138 | **0.710** |
| `real_scam_delivery_customs_ru` | 0.498 | 0.655 |
| `real_scam_telecom_verify_ru` | 0.689 | 0.514 |
| `aug_prize_lottery_ru_0` | 0.717 | 0.574 |

So the native core moves FPR on OOD data, and by the project's evaluation rules it cannot ship as
the citizen-facing scorer without an ADR. There are two ways out:
1. **Retrain the heads on native embeddings** (`QORGAN_EMBED_BACKEND=onnx`) for the native app,
   keeping a separate bundle per runtime. Then re-run the browser and native gates and evaluate
   FPR per split.
2. **Keep the WebView/TWA for scoring** and use the native core only where the browser embeddings
   are reproduced. That never happens with native ORT.

**How to measure on a phone:** `aar/src/androidTest/.../DeviceRuntimeGateTest.kt` runs this same
gate on the device. It reports flips against Python and against the browser, ms per transcript,
model load time, PSS and native heap. It is compiled (`assembleDebugAndroidTest`) but **not run**:
no device has been attached yet.

```bash
D=/sdcard/Android/data/kz.qorgan.core.test/files
adb shell mkdir -p $D
adb push site/models/Xenova/multilingual-e5-base/onnx/model_quantized.onnx $D/
adb push site/models/Xenova/multilingual-e5-base/tokenizer.json $D/
(cd android/qorgan-core && ./gradlew :aar:connectedAndroidTest)
adb logcat -s QorganGate
```

On a phone, expect the vs-Python numbers to be identical to the Mac if Android arm64 ORT 1.21 is
bit-identical to the desktop build. That is plausible (same MLAS kernels) but unverified. The
vs-browser numbers are the ones that matter for the shipped heads.

## Not done / open

- **STT:** not in this spike. The candidate is sherpa-onnx Android (GigaAM / Vosk-class models).
  The web light mode's GigaAM-only path (ADR D64/D67) is the reference. Audio must stay on the
  device, as on the web.
- **Real-device memory and latency:** unmeasured. The 60–145 ms per transcript above is a Mac M-series.
- The **role slot** (ADR D63, `role.js`), the post-call summary and the report builder are not
  ported. Only the scrubber, which the report needs, is.
- The DJL tokenizer on Android is **compiled into the AAR but never executed on a device**.
- No R8/minify configuration has been tested. `consumer-rules.pro` keeps the JNI classes.
- `onnxruntime` 1.21 must move together with the server and transformers.js pins (ADR D32).

## Build / run

```bash
export JAVA_HOME=/opt/homebrew/opt/openjdk@17/libexec/openjdk.jdk/Contents/Home  # brew openjdk@17
export ANDROID_HOME=$HOME/Library/Android/sdk
cd android/qorgan-core
echo "sdk.dir=$ANDROID_HOME" > local.properties     # gitignored
./gradlew :core:test                                # parity tests (embedder tests skip without the model)
./gradlew :core:test -PruntimeGate=true             # + the 200-case runtime gate (~20 s)
./gradlew :aar:assembleRelease                      # aar/build/outputs/aar/aar-release.aar
./gradlew :aar:assembleDebugAndroidTest             # device gate test APK
```
The model files come from `python scripts/deploy_bootstrap.py` (`site/models/Xenova/...`, gitignored).
