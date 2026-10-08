"""Build the real-speech ASR test set: FLEURS kk/ru/en + Tim2190 KK-RU code-switch clips.

Inputs come from `fetch.sh` (gitignored `data/asr_capture/_wav/fleurs/_dl/`).

- FLEURS (google/fleurs, CC BY 4.0, read Wikipedia sentences, split `test`): the FIRST N
  utterances of each language in archive order (the order the HF streaming loader yields them),
  no filtering -- 250 kk_kz, 250 ru_ru, 150 en_us. The reference is `raw_transcription` with
  digits spelled out by num2words (kk via num2words' `kz`), see `spell_digits`; rows that had
  digits are flagged so the score can be re-run on the digit-free subset.
- Tim2190/kazakh-codeswitch-asr (CC BY 4.0, ungated): 31 natural KK-RU code-switched clips
  (stand-up, interviews, vlogs). References: `transcript_verbatim` and
  `transcript_normalized_written`.

Every clip -> 16 kHz mono int16, two channels: `clean` and `phone` (the CS set's
`phone_channel`: 300-3400 Hz, G.711 mu-law at 8 kHz, white noise at 25 dB SNR, seeded per row).

    python scripts/spikes/asr_wordlevel/fleurs/prepare.py   # -> data/asr_capture/fleurs_set.jsonl
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
import tarfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from build_cs_set import phone_channel, seed_for, write_wav  # noqa: E402

REPO = HERE.parents[3]
DL = REPO / "data/asr_capture/_wav/fleurs/_dl"
WAV_DIR = REPO / "data/asr_capture/_wav/fleurs"
OUT = REPO / "data/asr_capture/fleurs_set.jsonl"
COUNTS = {"kk_kz": 250, "ru_ru": 250, "en_us": 150}
LANG = {"kk_kz": "kk", "ru_ru": "ru", "en_us": "en"}
NUM2WORDS_LANG = {"kk": "kz", "ru": "ru", "en": "en"}
PERCENT = {"kk": " пайыз", "ru": " процентов", "en": " percent"}
CHANNELS = ("clean", "phone")
SAMPLE_RATE = 16000


def to_16k_mono(data: bytes) -> np.ndarray:
    """Any audio container -> 16 kHz mono int16 via ffmpeg (stdin -> stdout)."""
    out = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", "pipe:0", "-ac", "1", "-ar", str(SAMPLE_RATE),
                          "-f", "s16le", "pipe:1"], input=data, capture_output=True, check=True).stdout
    return np.frombuffer(out, dtype=np.int16)


def spell_digits(text: str, lang: str) -> str:
    """Digits -> words. Cardinal (nominative) forms only, so inflected / year readings still
    mismatch; the scorer reports the digit-free subset as the clean comparison."""
    from num2words import num2words

    code = NUM2WORDS_LANG[lang]
    text = re.sub(r"(?<=\d),(?=\d{3}\b)", "", text) if lang == "en" else re.sub(r"(?<=\d)\s(?=\d{3}\b)", "", text)
    text = re.sub(r"(\d)\s*%", lambda m: m.group(1) + PERCENT[lang], text)
    if lang == "en":
        text = re.sub(r"\b(\d+)(st|nd|rd|th)\b", lambda m: " " + num2words(int(m.group(1)), lang="en", to="ordinal") + " ", text)
        text = re.sub(r"\b(1[1-9]\d\d|20\d\d)\b", lambda m: " " + num2words(int(m.group(0)), lang="en", to="year") + " ", text)
    text = re.sub(r"\d+[.,]\d+", lambda m: " " + num2words(float(m.group(0).replace(",", ".")), lang=code) + " ", text)
    text = re.sub(r"\d+", lambda m: " " + num2words(int(m.group(0)), lang=code) + " ", text)
    return re.sub(r"\s+", " ", text).strip()


def fleurs_rows(lang_dir: str, count: int):
    """First `count` test utterances in archive order: (tsv fields, audio bytes)."""
    meta = {}
    for line in (DL / "fleurs" / lang_dir / "test.tsv").read_text(encoding="utf-8").splitlines():
        fields = line.split("\t")
        meta[fields[1]] = {"fleurs_id": fields[0], "raw": fields[2], "gender": fields[6] if len(fields) > 6 else None}
    taken = 0
    with tarfile.open(DL / "fleurs" / lang_dir / "test.tar.gz", "r|gz") as archive:
        for member in archive:
            name = Path(member.name).name
            if not member.isfile() or name not in meta:
                continue
            yield name, meta[name], archive.extractfile(member).read()
            taken += 1
            if taken == count:
                return


def tim_rows():
    with (DL / "tim/metadata.csv").open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            yield row, (DL / "tim" / row["file_name"]).read_bytes()


def emit(sink, base: dict, clean: np.ndarray) -> int:
    for channel in CHANNELS:
        key = f"{base['id']}_{channel}"
        audio = clean if channel == "clean" else phone_channel(clean, seed_for(key))
        wav = WAV_DIR / base["source"] / f"{key}.wav"
        write_wav(wav, audio)
        sink.write(json.dumps({**base, "channel": channel, "wav": str(wav.relative_to(REPO)),
                               "duration": round(len(audio) / SAMPLE_RATE, 3)}, ensure_ascii=False) + "\n")
    return len(CHANNELS)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    written = 0
    with args.out.open("w", encoding="utf-8") as sink:
        for lang_dir, count in COUNTS.items():
            lang = LANG[lang_dir]
            for order, (name, meta, data) in enumerate(fleurs_rows(lang_dir, count)):
                base = {"id": f"fleurs_{lang}_{order:03d}", "source": "fleurs", "lang": lang, "file": name,
                        "fleurs_id": meta["fleurs_id"], "reference_raw": meta["raw"],
                        "reference": spell_digits(meta["raw"], lang), "has_digits": bool(re.search(r"\d", meta["raw"]))}
                written += emit(sink, base, to_16k_mono(data))
            print(f"{lang_dir}: {order + 1} utterances", flush=True)
        for row, data in tim_rows():
            base = {"id": f"tim_{row['audio_id']}", "source": "tim", "lang": "kk+ru", "reference": row["transcript_verbatim"],
                    "reference_written": row["transcript_normalized_written"], "has_digits": False,
                    "flags": {k: int(row[k]) for k in ("has_contraction", "has_dialect_slang", "has_barbarisms", "has_propers")}}
            written += emit(sink, base, to_16k_mono(data))
    print(f"wrote {written} rows to {args.out}")


if __name__ == "__main__":
    main()
