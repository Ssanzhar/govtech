# B7 spike — on-device ASR bake-off (Vosklet), reproducible bench

Outcome and numbers: `docs/DECISIONS.md` D25. This folder reproduces them.

```bash
# 1. clips: synthesize the two demo scenes + a Kazakh scam line (macOS `say`), 16 kHz mono WAV
python scripts/spikes/vosklet_bench/make_clips.py --out /tmp/b7/clips
# 2. models: plain ustar tarballs of the small Vosk models (Vosklet rejects macOS metadata entries)
for m in vosk-model-small-ru-0.22 vosk-model-small-kz-0.42; do
  COPYFILE_DISABLE=1 tar --format=ustar --no-xattrs --no-mac-metadata -czf /tmp/b7/models/$m.tar.gz -C ~/.cache/vosk $m
done
# 3. serve WITH cross-origin isolation (Vosklet needs SharedArrayBuffer), open, click Run
cp scripts/spikes/vosklet_bench/{index.html,serve.py} /tmp/b7/ && (cd /tmp/b7 && python serve.py)   # http://127.0.0.1:8022/index.html
```

Vosklet caches a model by the id passed to `createModel`; bump the id after replacing an
archive or you will keep loading the old extraction. The page feeds decoded 16 kHz PCM to
the recognizer faster than real time and reports RTF = processing time / audio duration.

## The three demo scenes through the on-device recogniser (B9 regression)

`scenes_harness.html` drives `site/core/asr.js` + `site/core/device.js` with the synthesized
clips through `HTMLMediaElement.captureStream()` — the exact microphone code path minus
`getUserMedia` — and prints each voted utterance with the meter it produced.

```bash
mkdir -p /tmp/b9 && ln -s "$PWD/site/core" /tmp/b9/core && ln -s "$PWD/site/models" /tmp/b9/models && ln -s "$PWD/site/vendor" /tmp/b9/vendor \
  && ln -s /tmp/b7/clips /tmp/b9/clips && cp scripts/spikes/vosklet_bench/{scenes_harness.html,serve.py} /tmp/b9/
(cd /tmp/b9 && sed -i '' 's/8022/8023/' serve.py && python serve.py)   # http://127.0.0.1:8023/scenes_harness.html
```

Expected (2026-09-18, laptop, dual small models): RU scam → 81/100 critical (2 utterances,
both voted RU); real bank call → 14/100 low; KK scam → 61/100 high (voted KK 0.96).
