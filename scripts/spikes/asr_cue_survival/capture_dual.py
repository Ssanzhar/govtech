"""Capture BOTH recognisers' output per utterance, in dialogue order (STT Tier B).

The shipped microphone mode runs a Kazakh and a Russian recogniser concurrently for the whole
call and votes per utterance. Two WASM instances on a 3 GB phone is the open gate from ADR
D25, so the question is whether the second one can be switched off once the language is
known. Deciding that needs what `capture.py` does not store: what EACH recogniser said for
EACH utterance, in order, so any locking policy can be replayed offline (the A6 meter-sweep
pattern -- decode once, simulate many).

    python scripts/spikes/asr_cue_survival/capture_dual.py --out data/asr_capture/dual.jsonl

One row per utterance: {dialogue_id, index, language, is_hard_negative, reference,
kk: {text, confidence}, ru: {text, confidence}}. Resumable by dialogue id.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import wave
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))

VOICES = {"ru": "Milena", "kk": "Aru", "mixed": "Milena"}
SPEECH_RATE = "175"
CHUNK_BYTES = 4000
MAX_WORDS = 60


def synthesize(text: str, voice: str, wav_path: Path) -> None:
    aiff = wav_path.with_suffix(".aiff")
    subprocess.run(["say", "-v", voice, "-r", SPEECH_RATE, "-o", str(aiff), text], check=True)
    subprocess.run(
        ["afconvert", "-f", "WAVE", "-d", "LEI16@16000", "-c", "1", str(aiff), str(wav_path)],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    aiff.unlink(missing_ok=True)


def decode_both(wav_path: Path, recognizers) -> dict[str, dict]:
    """Feed the same PCM to each recogniser independently and take its final hypothesis."""
    from qorgan.asr.vosk_stream import _parse_result

    with wave.open(str(wav_path), "rb") as handle:
        frames = handle.readframes(handle.getnframes())
    out: dict[str, dict] = {}
    for language, recognizer in recognizers.items():
        recognizer.Reset()
        for start in range(0, len(frames), CHUNK_BYTES):
            recognizer.AcceptWaveform(frames[start : start + CHUNK_BYTES])
        text, confidence = _parse_result(recognizer.FinalResult())
        out[language] = {"text": text, "confidence": confidence}
    return out


def dialogues(splits: list[str]):
    for split in splits:
        path = REPO / "data/processed" / f"{split}.jsonl"
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                dialogue = json.loads(line)
                if not dialogue["id"].endswith("-asr"):
                    yield split, dialogue


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=REPO / "data/asr_capture/dual.jsonl")
    parser.add_argument("--splits", nargs="+", default=["authored_heldout", "shift"])
    parser.add_argument("--limit-dialogues", type=int, default=0)
    args = parser.parse_args()

    from qorgan.asr.vosk_stream import _load_recognizers

    args.out.parent.mkdir(parents=True, exist_ok=True)
    done: set[str] = set()
    if args.out.exists():
        done = {json.loads(l)["dialogue_id"] for l in args.out.read_text(encoding="utf-8").splitlines() if l.strip()}
    todo = [(s, d) for s, d in dialogues(args.splits) if d["id"] not in done]
    if args.limit_dialogues:
        todo = todo[: args.limit_dialogues]
    print(f"{len(todo)} dialogues to capture; {len(done)} already done", flush=True)

    recognizers = _load_recognizers()
    scratch = args.out.parent / "_wav"
    scratch.mkdir(exist_ok=True)
    wav = scratch / "dual.wav"
    with args.out.open("a", encoding="utf-8") as sink:
        for number, (split, dialogue) in enumerate(todo, 1):
            voice = VOICES[dialogue["language"]]
            for index, utterance in enumerate(dialogue["utterances"]):
                text = utterance["text"]
                if len(text.split()) > MAX_WORDS:
                    continue
                try:
                    synthesize(text, voice, wav)
                    heard = decode_both(wav, recognizers)
                except Exception as exc:
                    print(f"  !! {dialogue['id']}#{index}: {type(exc).__name__}: {exc}", flush=True)
                    continue
                sink.write(json.dumps({
                    "dialogue_id": dialogue["id"], "split": split, "index": index,
                    "language": dialogue["language"], "speaker": utterance["speaker"],
                    "is_hard_negative": dialogue["label"]["is_hard_negative"],
                    "reference": text, **heard,
                }, ensure_ascii=False) + "\n")
            sink.flush()
            if number % 10 == 0 or number == len(todo):
                print(f"  {number}/{len(todo)} dialogues", flush=True)
    wav.unlink(missing_ok=True)
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
