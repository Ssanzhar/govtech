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
