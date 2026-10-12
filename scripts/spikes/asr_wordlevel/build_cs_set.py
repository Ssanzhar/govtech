"""Build the code-switched ASR test set (word-level STT track, docs/research/2026-10/stt_wordlevel.md).

The earlier `mixed` capture read Kazakh sentences with a Russian voice, so nothing measured a
mid-sentence switch. Each row of `cs_utterances.jsonl` (authored 2026-10-08 for this test only --
never training data) is a sentence of language segments. Two synthesis conditions:

- `single`  -- one Kazakh voice (Aru) reads the whole kk/ru sentence, like a bilingual speaker
               inserting Russian words. Rows with English segments are skipped (Aru cannot read
               Latin text naturally).
- `spliced` -- each segment in its own native voice (Aru / Milena / Samantha), joined with a
               short gap. The speaker change is a cue a real call does not have; report both.

Two channels: `clean` (16 kHz) and `phone` (300-3400 Hz band-pass, G.711 mu-law at 8 kHz, back to
16 kHz, white noise at 25 dB SNR, seeded per row). Audio goes to the gitignored
`data/asr_capture/_wav/cs/`; the manifest (`data/asr_capture/cs_set.jsonl`) carries the reference,
per-word language labels and, for `spliced`, each segment's time span.

    python scripts/spikes/asr_wordlevel/build_cs_set.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import tempfile
import wave
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
SOURCE = Path(__file__).with_name("cs_utterances.jsonl")
OUT = REPO / "data/asr_capture/cs_set.jsonl"
WAV_DIR = REPO / "data/asr_capture/_wav/cs"
VOICES = {"kk": "Aru", "ru": "Milena", "en": "Samantha"}
SINGLE_VOICE = "Aru"
SPEECH_RATE = "175"
SAMPLE_RATE = 16000
GAP_SECONDS = 0.06
PHONE_SNR_DB = 25.0
CONDITIONS = ("single", "spliced")
CHANNELS = ("clean", "phone")


def say_to_samples(text: str, voice: str) -> np.ndarray:
    """macOS `say` -> mono int16 samples at 16 kHz."""
    with tempfile.TemporaryDirectory() as tmp:
        aiff, wav_path = Path(tmp) / "s.aiff", Path(tmp) / "s.wav"
        subprocess.run(["say", "-v", voice, "-r", SPEECH_RATE, "-o", str(aiff), text], check=True)
        subprocess.run(["afconvert", "-f", "WAVE", "-d", f"LEI16@{SAMPLE_RATE}", "-c", "1", str(aiff), str(wav_path)], check=True)
        return read_wav(wav_path)


def read_wav(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as handle:
        if handle.getframerate() != SAMPLE_RATE or handle.getnchannels() != 1:
            raise ValueError(f"{path}: expected mono {SAMPLE_RATE} Hz")
        return np.frombuffer(handle.readframes(handle.getnframes()), dtype=np.int16)


def write_wav(path: Path, samples: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(SAMPLE_RATE)
        handle.writeframes(samples.astype(np.int16).tobytes())


def phone_channel(samples: np.ndarray, seed: int) -> np.ndarray:
    """Band-limit and mu-law code at 8 kHz with ffmpeg, then add seeded white noise."""
    with tempfile.TemporaryDirectory() as tmp:
        src, coded, back = Path(tmp) / "a.wav", Path(tmp) / "b.wav", Path(tmp) / "c.wav"
        write_wav(src, samples)
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(src), "-af", "highpass=f=300,lowpass=f=3400",
                        "-ar", "8000", "-c:a", "pcm_mulaw", str(coded)], check=True)
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(coded), "-ar", str(SAMPLE_RATE),
                        "-c:a", "pcm_s16le", str(back)], check=True)
        signal = read_wav(back).astype(np.float64)
    power = float(np.mean(signal ** 2)) or 1.0
    noise = np.random.default_rng(seed).normal(0.0, np.sqrt(power / 10 ** (PHONE_SNR_DB / 10)), signal.shape)
    return np.clip(signal + noise, -32768, 32767).astype(np.int16)


def reference_words(segments: list[list[str]]) -> list[dict]:
    """Per-word labels: every word takes its segment's language (a Russian stem with a Kazakh
    suffix is written inside the Kazakh segment, so the suffix decides -- the KSC2 convention)."""
    return [{"word": w, "lang": lang} for lang, text in segments for w in text.split()]


def synthesize(row: dict, condition: str) -> tuple[np.ndarray, list[dict]] | None:
    segments = row["segments"]
    if condition == "single":
        if any(lang == "en" for lang, _ in segments):
            return None
        return say_to_samples(" ".join(text for _, text in segments), SINGLE_VOICE), []
    gap = np.zeros(int(GAP_SECONDS * SAMPLE_RATE), dtype=np.int16)
    parts, spans, cursor = [], [], 0
    for lang, text in segments:
        audio = say_to_samples(text, VOICES[lang])
        spans.append({"lang": lang, "start": cursor / SAMPLE_RATE, "end": (cursor + len(audio)) / SAMPLE_RATE})
        parts += [audio, gap]
        cursor += len(audio) + len(gap)
    return np.concatenate(parts[:-1]), spans


def seed_for(key: str) -> int:
    return int(hashlib.sha256(key.encode()).hexdigest()[:8], 16)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    rows = [json.loads(line) for line in SOURCE.read_text(encoding="utf-8").splitlines() if line.strip()]
    written = 0
    with args.out.open("w", encoding="utf-8") as sink:
        for row in rows:
            for condition in CONDITIONS:
                made = synthesize(row, condition)
                if made is None:
                    continue
                clean, spans = made
                for channel in CHANNELS:
                    key = f"{row['id']}_{condition}_{channel}"
                    audio = clean if channel == "clean" else phone_channel(clean, seed_for(key))
                    wav_path = WAV_DIR / f"{key}.wav"
                    write_wav(wav_path, audio)
                    sink.write(json.dumps({
                        "id": row["id"], "condition": condition, "channel": channel, "matrix": row["matrix"],
                        "label": row["label"], "wav": str(wav_path.relative_to(REPO)),
                        "duration": round(len(audio) / SAMPLE_RATE, 3),
                        "reference": " ".join(text for _, text in row["segments"]),
                        "words": reference_words(row["segments"]), "spans": spans,
                    }, ensure_ascii=False) + "\n")
                    written += 1
    print(f"wrote {written} rows to {args.out}")


if __name__ == "__main__":
    main()
