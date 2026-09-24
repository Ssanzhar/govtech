"""Is a grammar-constrained recogniser worth a third decode? (STT Tier B, item 5)

Proper domain biasing means rebuilding the decoding graph with an interpolated language model
-- that needs a Kaldi/OpenFST toolchain this machine does not have. Vosk offers the reachable
cousin: `KaldiRecognizer(model, rate, json_phrase_list)` builds a small LM on the fly, giving
a recogniser that can only emit those phrases (or `[unk]`). That is a keyword spotter, not a
bias, so it must be judged on both sides at once:

  * does it recover cue phrases the open recogniser mangles?
  * how often does it hallucinate a cue on speech that contained none?

It would cost a THIRD decode per utterance, which pulls against the language locking of the
same tier, so the bar is high.

    python scripts/spikes/asr_cue_survival/grammar_probe.py --per-class 40
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

from qorgan.classifier.cue_match import find_cue  # noqa: E402

VOICES = {"ru": "Milena", "kk": "Aru", "mixed": "Milena"}
MODELS = {"ru": "vosk-model-small-ru-0.22", "kk": "vosk-model-small-kz-0.42"}
CHUNK = 4000


def load_cues() -> dict[str, list[str]]:
    import yaml

    return yaml.safe_load((REPO / "data/lexicon/hard_signal_cues.yaml").read_text(encoding="utf-8"))["cues"]


def synthesize(text: str, voice: str, wav: Path) -> None:
    aiff = wav.with_suffix(".aiff")
    subprocess.run(["say", "-v", voice, "-r", "175", "-o", str(aiff), text], check=True)
    subprocess.run(["afconvert", "-f", "WAVE", "-d", "LEI16@16000", "-c", "1", str(aiff), str(wav)],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    aiff.unlink(missing_ok=True)


def decode(recognizer, wav: Path) -> str:
    from qorgan.asr.vosk_stream import _parse_result

    recognizer.Reset()
    with wave.open(str(wav), "rb") as handle:
        frames = handle.readframes(handle.getnframes())
    for start in range(0, len(frames), CHUNK):
        recognizer.AcceptWaveform(frames[start : start + CHUNK])
    return _parse_result(recognizer.FinalResult())[0]


def build(language: str, phrases: list[str] | None):
    """An open recogniser, or one restricted to `phrases`. Words outside the model's lexicon
    make Vosk refuse the grammar, so unusable phrases are reported and dropped."""
    from vosk import KaldiRecognizer, Model

    model = Model(model_name=MODELS[language])
    if phrases is None:
        recognizer = KaldiRecognizer(model, 16000)
    else:
        usable = list(phrases)
        while usable:
            try:
                recognizer = KaldiRecognizer(model, 16000, json.dumps([*usable, "[unk]"], ensure_ascii=False))
                break
            except Exception as exc:  # noqa: BLE001 - vosk raises a bare exception naming the word
                bad = str(exc)
                dropped = [p for p in usable if any(w in bad for w in p.lower().split())]
                if not dropped:
                    raise
                usable = [p for p in usable if p not in dropped]
        else:
            raise RuntimeError(f"no usable grammar phrases for {language}")
        print(f"  {language}: grammar built from {len(usable)}/{len(phrases)} phrases", flush=True)
    recognizer.SetWords(True)
    return recognizer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-class", type=int, default=40)
    parser.add_argument("--language", default="ru", choices=["ru", "kk"])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cues = load_cues()
    flat = [(t, c) for t, cl in cues.items() for c in cl]

    rows = []
    for split in ("test", "authored_heldout", "shift"):
        path = REPO / "data/processed" / f"{split}.jsonl"
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            dialogue = json.loads(line)
            if dialogue["language"] != args.language:
                continue
            for utterance in dialogue["utterances"]:
                text = utterance["text"]
                if 3 <= len(text.split()) <= 45:
                    hits = sorted({t for t, c in flat if c.lower() in text.lower()})
                    rows.append({"text": text, "cues": hits})
    rng = random.Random(args.seed)
    positives = [r for r in rows if r["cues"]][: args.per_class]
    negatives = rng.sample([r for r in rows if not r["cues"]], min(args.per_class, len(rows)))
    print(f"{args.language}: {len(positives)} cue-bearing, {len(negatives)} cue-free")

    open_rec = build(args.language, None)
    grammar_rec = build(args.language, [c for _, c in flat])

    scratch = REPO / "data/asr_capture/_wav"
    scratch.mkdir(parents=True, exist_ok=True)
    wav = scratch / "probe.wav"

    def cue_set(text: str) -> set[str]:
        return {t for t, cl in cues.items() if any(find_cue(text, c) for c in cl)}

    stats = {"open_hit": 0, "gram_hit": 0, "open_false": 0, "gram_false": 0}
    examples = []
    for row in positives:
        synthesize(row["text"], VOICES[args.language], wav)
        o, g = decode(open_rec, wav), decode(grammar_rec, wav)
        want = set(row["cues"])
        oh, gh = bool(cue_set(o) & want), bool(cue_set(g) & want)
        stats["open_hit"] += oh
        stats["gram_hit"] += gh
        if gh and not oh and len(examples) < 5:
            examples.append((row["text"], o, g))
    for row in negatives:
        synthesize(row["text"], VOICES[args.language], wav)
        o, g = decode(open_rec, wav), decode(grammar_rec, wav)
        stats["open_false"] += bool(cue_set(o))
        stats["gram_false"] += bool(cue_set(g))
    wav.unlink(missing_ok=True)

    print(f"\n| recogniser | cue recovered | false fire |")
    print(f"|---|---|---|")
    print(f"| open (ships today) | {stats['open_hit']}/{len(positives)} | {stats['open_false']}/{len(negatives)} |")
    print(f"| grammar-constrained | {stats['gram_hit']}/{len(positives)} | {stats['gram_false']}/{len(negatives)} |")
    for reference, o, g in examples:
        print(f"\nrecovered only by the grammar:\n  REF : {reference[:95]}\n  open: {o[:95]}\n  gram: {g[:95]}")


if __name__ == "__main__":
    main()
