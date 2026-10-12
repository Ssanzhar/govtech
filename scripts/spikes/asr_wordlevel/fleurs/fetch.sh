#!/bin/sh
# Fetch the real-speech test data for the FLEURS ASR comparison (word-level STT track).
# huggingface_hub's downloader stalled on this network (2026-10-08), so plain resumable curl.
#   google/fleurs (CC BY 4.0): test.tsv + test audio tarball for kk_kz, ru_ru, en_us (~1.35 GB)
#   Tim2190/kazakh-codeswitch-asr (CC BY 4.0, not gated): 31 real KK-RU code-switched clips
# issai/Kazakh_Speech_Corpus_2 is ONE 78 GB tar.gz split into 8.4 GB parts (no separate test
# archive), so its test subsets cannot be fetched under the 2 GB budget -- skipped.
#   sh scripts/spikes/asr_wordlevel/fleurs/fetch.sh
set -eu
REPO=$(cd "$(dirname "$0")/../../../.." && pwd)
DL="$REPO/data/asr_capture/_wav/fleurs/_dl"
HF=https://huggingface.co/datasets
get() { mkdir -p "$(dirname "$2")"; [ -f "$2.done" ] && return 0
        curl -sSL --retry 5 -C - -o "$2" "$1" && touch "$2.done"; }
for lang in kk_kz ru_ru en_us; do
  get "$HF/google/fleurs/resolve/main/data/$lang/test.tsv" "$DL/fleurs/$lang/test.tsv"
  get "$HF/google/fleurs/resolve/main/data/$lang/audio/test.tar.gz" "$DL/fleurs/$lang/test.tar.gz"
done
get "$HF/Tim2190/kazakh-codeswitch-asr/resolve/main/metadata.csv" "$DL/tim/metadata.csv"
for i in $(seq 1 31); do
  get "$HF/Tim2190/kazakh-codeswitch-asr/resolve/main/audio/kzaudio_$i.wav" "$DL/tim/audio/kzaudio_$i.wav"
done
echo fetched
