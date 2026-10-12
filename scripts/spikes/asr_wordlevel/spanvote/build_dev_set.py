"""Build the span-vote DEV set from `train`-split text only (never val / test / held-out).

Same recipe as `../build_cs_set.py` (which builds the 440-row TEST set from authored sentences),
but every segment is a contiguous word window of a monolingual `train` utterance (language kk, ru
or en; `mixed` dialogues are skipped). ~80 % of sequences are kk/ru switches, ~20 % embed an English
fragment. Conditions/channels as in the test set: `spliced` (native voice per segment:
Aru / Milena / Samantha, true segment times kept) and `single` (Aru reads a kk/ru sequence);
`clean` and `phone`.

    python scripts/spikes/asr_wordlevel/spanvote/build_dev_set.py   # -> data/asr_capture/spanvote_dev_set.jsonl
"""

from __future__ import annotations

import json
import random
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from build_cs_set import (CHANNELS, GAP_SECONDS, SAMPLE_RATE, SINGLE_VOICE, VOICES,  # noqa: E402
                          phone_channel, say_to_samples, seed_for, write_wav)

import numpy as np  # noqa: E402

REPO = HERE.parents[3]
TRAIN = REPO / "data/processed/train.jsonl"
OUT = REPO / "data/asr_capture/spanvote_dev_set.jsonl"
WAV_DIR = REPO / "data/asr_capture/_wav/spanvote/dev"
N_SEQUENCES = 250
EN_SHARE = 0.20
SEED = 20261009
WORD_RE = {"kk": re.compile(r"^[а-яәғқңөұүһі]+$"), "ru": re.compile(r"^[а-яё]+$"), "en": re.compile(r"^[a-z']+$")}
PATTERNS_CS = (("A", "B", "A"), ("A", "B"), ("B", "A"))   # A = matrix, B = embedded
PATTERNS_EN = (("A", "E", "A"), ("A", "E"), ("E", "A"))


def utterance_pool() -> dict[str, list[list[str]]]:
    pool = {"kk": [], "ru": [], "en": []}
    for line in TRAIN.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row["language"] not in pool:
            continue
        for utt in row["utterances"]:
            words = utt["text"].lower().split()
            if len(words) >= 2:
                pool[row["language"]].append(words)
    return pool


def fragment(rng: random.Random, pool: list[list[str]], lang: str, lo: int, hi: int) -> list[str]:
    """A contiguous window of lo..hi letter-only words from one train utterance."""
    for _ in range(1000):
        words = rng.choice(pool)
        n = rng.randint(lo, hi)
        if len(words) < n:
            continue
        start = rng.randint(0, len(words) - n)
        window = words[start:start + n]
        if all(WORD_RE[lang].match(w) for w in window):
            return window
    raise RuntimeError(f"no clean {lang} fragment")


def make_sequences(pool) -> list[dict]:
    rng = random.Random(SEED)
    rows = []
    for i in range(N_SEQUENCES):
        with_en = i < int(N_SEQUENCES * EN_SHARE)
        matrix = rng.choice(("kk", "ru"))
        embedded = "en" if with_en else ("ru" if matrix == "kk" else "kk")
        pattern = rng.choice(PATTERNS_EN if with_en else PATTERNS_CS)
        segments = []
        for slot in pattern:
            lang = matrix if slot == "A" else embedded
            lo, hi = (2, 4) if slot == "A" else (1, 3)
            segments.append([lang, " ".join(fragment(rng, pool[lang], lang, lo, hi))])
        rows.append({"id": f"dev_{i:03d}", "matrix": matrix, "segments": segments})
    rng.shuffle(rows)
    return rows


def synthesize(row: dict, condition: str):
    segments = row["segments"]
    if condition == "single":
        if any(lang == "en" for lang, _ in segments):
            return None
        return say_to_samples(" ".join(t for _, t in segments), SINGLE_VOICE), []
    gap = np.zeros(int(GAP_SECONDS * SAMPLE_RATE), dtype=np.int16)
    parts, spans, cursor = [], [], 0
    for lang, text in segments:
        audio = say_to_samples(text, VOICES[lang])
        spans.append({"lang": lang, "start": cursor / SAMPLE_RATE, "end": (cursor + len(audio)) / SAMPLE_RATE})
        parts += [audio, gap]
        cursor += len(audio) + len(gap)
    return np.concatenate(parts[:-1]), spans


def main() -> None:
    rows = make_sequences(utterance_pool())
    written = 0
    with OUT.open("w", encoding="utf-8") as sink:
        for row in rows:
            for condition in ("single", "spliced"):
                made = synthesize(row, condition)
                if made is None:
                    continue
                clean, spans = made
                for channel in CHANNELS:
                    key = f"{row['id']}_{condition}_{channel}"
                    audio = clean if channel == "clean" else phone_channel(clean, seed_for(key))
                    wav = WAV_DIR / f"{key}.wav"
                    write_wav(wav, audio)
                    sink.write(json.dumps({
                        "id": row["id"], "condition": condition, "channel": channel, "matrix": row["matrix"],
                        "wav": str(wav.relative_to(REPO)), "duration": round(len(audio) / SAMPLE_RATE, 3),
                        "reference": " ".join(t for _, t in row["segments"]),
                        "words": [{"word": w, "lang": l} for l, t in row["segments"] for w in t.split()],
                        "spans": spans}, ensure_ascii=False) + "\n")
                    written += 1
    print(f"wrote {written} rows to {OUT}")


if __name__ == "__main__":
    main()
