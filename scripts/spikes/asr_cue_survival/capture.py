"""Capture REAL speech-recogniser output for the cue-survival experiment (Tier A).

Why not a synthetic error model: a hand-written corruption model proves whatever it was
designed to prove. This speaks each utterance with macOS `say` and decodes it with the very
Vosk models the browser ships, so the errors are the recogniser's own. Synthesised speech is
cleaner than a speakerphone in a kitchen, so every number this produces is a LOWER bound on
the real error rate -- and therefore a lower bound on what better matching can recover.

    python scripts/spikes/asr_cue_survival/capture.py --out data/asr_capture/pairs.jsonl

Writes one JSON object per utterance: {id, split, language, voice, reference, hypothesis,
confidence, cue_tactics_reference}. Resumable -- ids already in the output file are skipped.
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import wave
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))

# macOS voices used by the B7 bench. A Kazakh voice exists (Aru); `mixed` is a Russian matrix
# with Kazakh insertions, so Milena reads it -- which is itself a realistic stress case.
VOICES = {"ru": "Milena", "kk": "Aru", "mixed": "Milena"}
SPEECH_RATE = "175"
CHUNK_BYTES = 4000


def cue_tactics(text: str, flat: list[tuple[str, str]]) -> list[str]:
    low = text.lower()
    return sorted({tactic for tactic, cue in flat if cue.lower() in low})


def load_flat_cues() -> list[tuple[str, str]]:
    import yaml

    cues = yaml.safe_load((REPO / "data/lexicon/hard_signal_cues.yaml").read_text(encoding="utf-8"))["cues"]
    return [(tactic, cue) for tactic, cue_list in cues.items() for cue in cue_list]


def select_utterances(splits: list[str], per_split_negatives: int, seed: int) -> list[dict]:
    """Every cue-bearing utterance (the positives the matcher must recover) plus a seeded
    sample of cue-free utterances (the false-alarm side). Styled `-asr` copies are skipped:
    they are the same speech."""
    flat = load_flat_cues()
    positives, negatives = [], []
    for split in splits:
        path = REPO / "data/processed" / f"{split}.jsonl"
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            dialogue = json.loads(line)
            if dialogue["id"].endswith("-asr"):
                continue
            for index, utterance in enumerate(dialogue["utterances"]):
                text = utterance["text"]
                if len(text.split()) > 60:  # one very long turn would dominate the audio budget
                    continue
                row = {
                    "id": f"{dialogue['id']}#{index}",
                    "split": split,
                    "language": dialogue["language"],
                    "reference": text,
                    "cue_tactics_reference": cue_tactics(text, flat),
                }
                (positives if row["cue_tactics_reference"] else negatives).append(row)
    rng = random.Random(seed)
    sampled: list[dict] = []
    by_split: dict[str, list[dict]] = {}
    for row in negatives:
        by_split.setdefault(row["split"], []).append(row)
    for split, rows in by_split.items():
        sampled.extend(rng.sample(rows, min(per_split_negatives, len(rows))))
    return positives + sampled


def synthesize(text: str, voice: str, wav_path: Path) -> None:
    aiff = wav_path.with_suffix(".aiff")
    subprocess.run(["say", "-v", voice, "-r", SPEECH_RATE, "-o", str(aiff), text], check=True)
    subprocess.run(
        ["afconvert", "-f", "WAVE", "-d", "LEI16@16000", "-c", "1", str(aiff), str(wav_path)],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    aiff.unlink(missing_ok=True)


def decode(wav_path: Path, recognizers) -> tuple[str, str, float]:
    """The shipped dual-recogniser vote over one clip: (language, text, confidence)."""
    from qorgan.asr.stream import CommittedUtterance
    from qorgan.asr.vosk_stream import recognize_stream

    with wave.open(str(wav_path), "rb") as handle:
        frames = handle.readframes(handle.getnframes())
    chunks = [frames[i : i + CHUNK_BYTES] for i in range(0, len(frames), CHUNK_BYTES)]
    parts, languages, confidences = [], [], []
    for event in recognize_stream(iter(chunks), recognizers=recognizers):
        # Only endpoint-committed utterances: partials are rolling prefixes of the same words
        # and concatenating them would fabricate text no user ever sees.
        if not isinstance(event, CommittedUtterance):
            continue
        text = event.text.strip()
        if text:
            parts.append(text)
            languages.append(event.language)
            confidences.append(float(event.confidence))
    language = max(set(languages), key=languages.count) if languages else ""
    confidence = sum(confidences) / len(confidences) if confidences else 0.0
    return language, " ".join(parts), confidence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=REPO / "data/asr_capture/pairs.jsonl")
    parser.add_argument("--splits", nargs="+", default=["test", "authored_heldout", "shift", "train"])
    parser.add_argument("--negatives-per-split", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=0, help="stop after N utterances (smoke run)")
    args = parser.parse_args()

    from qorgan.asr.vosk_stream import _load_recognizers

    rows = select_utterances(args.splits, args.negatives_per_split, args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if args.out.exists():
        done = {json.loads(l)["id"] for l in args.out.read_text(encoding="utf-8").splitlines() if l.strip()}
    todo = [r for r in rows if r["id"] not in done]
    if args.limit:
        todo = todo[: args.limit]
    positives = sum(1 for r in todo if r["cue_tactics_reference"])
    print(f"{len(todo)} utterances to capture ({positives} cue-bearing); {len(done)} already done", flush=True)

    recognizers = _load_recognizers()
    scratch = args.out.parent / "_wav"
    scratch.mkdir(exist_ok=True)
    with args.out.open("a", encoding="utf-8") as sink:
        for number, row in enumerate(todo, 1):
            voice = VOICES[row["language"]]
            wav = scratch / "clip.wav"
            try:
                synthesize(row["reference"], voice, wav)
                language, hypothesis, confidence = decode(wav, recognizers)
            except Exception as exc:  # a clip that will not synthesise must not kill the run
                print(f"  !! {row['id']}: {type(exc).__name__}: {exc}", flush=True)
                continue
            sink.write(json.dumps({**row, "voice": voice, "decoded_language": language,
                                   "hypothesis": hypothesis, "confidence": confidence}, ensure_ascii=False) + "\n")
            sink.flush()
            if number % 25 == 0 or number == len(todo):
                print(f"  {number}/{len(todo)}", flush=True)
    wav.unlink(missing_ok=True)
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
