"""Build the synthetic part of the English recogniser test set.

Sets (manifest `data/asr_capture/english_set.jsonl`, audio in gitignored `data/asr_capture/_wav/english/`):
- `authored`  -- the 120 call-context sentences of `sentences.jsonl` (written 2026-10-08 for this
                 test only, never training data), each read by two of six macOS English voices
                 (US, GB, AU, IE, IN, ZA), clean and phone channel.
- `cs_span`   -- the English segments of the spliced code-switched rows of `cs_set.jsonl`, cut out
                 of the clean and phone WAVs by their recorded span times.
- `probe_cs`  -- the kk/ru `single` rows of `cs_set.jsonl` (false-win probe; no English in them).
FLEURS rows (`fleurs_en`, `probe_fleurs_kk`, `probe_fleurs_ru`) come from `fetch_fleurs.py`.

    .venv/bin/python scripts/spikes/asr_wordlevel/english/build_en_set.py
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from build_cs_set import SAMPLE_RATE, SPEECH_RATE, phone_channel, read_wav, seed_for, write_wav  # noqa: E402

REPO = HERE.parents[3]
SENTENCES = HERE / "sentences.jsonl"
CS_SOURCE = HERE.parent / "cs_utterances.jsonl"
CS_SET = REPO / "data/asr_capture/cs_set.jsonl"
OUT = REPO / "data/asr_capture/english_set.jsonl"
WAV_DIR = REPO / "data/asr_capture/_wav/english"
VOICES = ("Samantha", "Daniel", "Karen", "Moira", "Rishi", "Tessa")
VOICE_OFFSET = 3  # second reading uses the voice three steps along the ring
SPAN_PAD = 0.03


SAY_TIMEOUT = 60
SAY_ATTEMPTS = 3


def say_to_samples(text: str, voice: str):
    """build_cs_set.say_to_samples with a timeout: `say` occasionally hangs on a voice."""
    for attempt in range(SAY_ATTEMPTS):
        with tempfile.TemporaryDirectory() as tmp:
            aiff, wav_path = Path(tmp) / "s.aiff", Path(tmp) / "s.wav"
            try:
                subprocess.run(["say", "-v", voice, "-r", SPEECH_RATE, "-o", str(aiff), text], check=True, timeout=SAY_TIMEOUT)
            except subprocess.TimeoutExpired:
                print(f"  say timed out ({voice}, attempt {attempt + 1}): {text[:40]}", flush=True)
                continue
            subprocess.run(["afconvert", "-f", "WAVE", "-d", f"LEI16@{SAMPLE_RATE}", "-c", "1", str(aiff), str(wav_path)], check=True)
            return read_wav(wav_path)
    raise RuntimeError(f"say failed for {voice}: {text}")


def rel(path: Path) -> str:
    return str(path.relative_to(REPO))


def authored_rows() -> list[dict]:
    rows = []
    sentences = [json.loads(l) for l in SENTENCES.read_text(encoding="utf-8").splitlines() if l.strip()]
    for index, sentence in enumerate(sentences):
        for voice in (VOICES[index % len(VOICES)], VOICES[(index + VOICE_OFFSET) % len(VOICES)]):
            clean_wav = WAV_DIR / f"{sentence['id']}_{voice}_clean.wav"
            clean = read_wav(clean_wav) if clean_wav.exists() else say_to_samples(sentence["text"], voice)
            for channel in ("clean", "phone"):
                key = f"{sentence['id']}_{voice}_{channel}"
                wav = WAV_DIR / f"{key}.wav"
                if wav.exists():
                    audio = read_wav(wav)
                else:
                    audio = clean if channel == "clean" else phone_channel(clean, seed_for(key))
                    write_wav(wav, audio)
                rows.append({"id": key, "set": "authored", "lang": "en", "voice": voice, "channel": channel,
                             "domain": sentence["domain"], "label": sentence["label"], "wav": rel(wav),
                             "duration": round(len(audio) / SAMPLE_RATE, 3), "reference": sentence["text"]})
    return rows


def cs_rows() -> list[dict]:
    segments = {r["id"]: r["segments"] for r in map(json.loads, CS_SOURCE.read_text(encoding="utf-8").splitlines()) if r}
    rows = []
    for row in map(json.loads, CS_SET.read_text(encoding="utf-8").splitlines()):
        if row["condition"] == "single":
            rows.append({"id": f"{row['id']}_{row['channel']}", "set": "probe_cs", "lang": row["matrix"],
                         "voice": "Aru", "channel": row["channel"], "wav": row["wav"],
                         "duration": row["duration"], "reference": row["reference"]})
            continue
        samples = read_wav(REPO / row["wav"])
        for number, (span, (lang, text)) in enumerate(zip(row["spans"], segments[row["id"]])):
            if span["lang"] != "en":
                continue
            assert lang == "en", row["id"]
            start = max(0, int((span["start"] - SPAN_PAD) * SAMPLE_RATE))
            end = min(len(samples), int((span["end"] + SPAN_PAD) * SAMPLE_RATE))
            key = f"{row['id']}_seg{number}_{row['channel']}"
            wav = WAV_DIR / f"{key}.wav"
            write_wav(wav, samples[start:end])
            rows.append({"id": key, "set": "cs_span", "lang": "en", "voice": "Samantha", "channel": row["channel"],
                         "wav": rel(wav), "duration": round((end - start) / SAMPLE_RATE, 3), "reference": text})
    return rows


def main() -> None:
    rows = cs_rows() + authored_rows()
    with OUT.open("w", encoding="utf-8") as sink:
        for row in rows:
            sink.write(json.dumps(row, ensure_ascii=False) + "\n")
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["set"]] = counts.get(row["set"], 0) + 1
    print(f"wrote {len(rows)} rows to {OUT}: {counts}")


if __name__ == "__main__":
    main()
