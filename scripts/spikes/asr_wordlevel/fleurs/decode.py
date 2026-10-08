"""Decode the real-speech set (FLEURS + Tim2190 code-switch clips) with every candidate engine.

Reuses the code-switch spike's helpers (`decode_cs.py`): GigaAM-Multilingual 220M / 600M CTC int8
via sherpa-onnx at ONE thread, the three shipped Vosk models and the replica of the shipped
per-utterance vote (kk/ru/en, English handicap 0.15). Each Vosk model's own decode time is
recorded so the score can sum them (the browser runs all three on every utterance).

    python scripts/spikes/asr_wordlevel/fleurs/decode.py --channel clean   # -> fleurs_decodes_clean.jsonl
    python scripts/spikes/asr_wordlevel/fleurs/decode.py --channel phone

Resumable: rows already in the output are skipped. Run the channels as separate processes.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from decode_cs import (GIGAAM_MODELS, decode_gigaam, decode_vosk_one, gigaam_recognizer,  # noqa: E402
                       read_samples, vosk_models, vote_utterance)

REPO = HERE.parents[3]
MANIFEST = REPO / "data/asr_capture/fleurs_set.jsonl"
OUT_PATTERN = "data/asr_capture/fleurs_decodes_{channel}.jsonl"
ENGINES = ("gigaam220", "gigaam600", "vosk")


def decode_vosk_timed(models: dict, samples) -> dict:
    hyps = {}
    for lang, model in models.items():
        started = time.perf_counter()
        hyp = decode_vosk_one(model, samples)
        hyps[lang] = {**hyp, "seconds": round(time.perf_counter() - started, 4)}
    winner = vote_utterance(hyps)
    return {"per_model": hyps, "winner": winner, "text": hyps[winner]["text"],
            "seconds": round(sum(h["seconds"] for h in hyps.values()), 4)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--channel", choices=("clean", "phone"), required=True)
    parser.add_argument("--engines", nargs="+", default=list(ENGINES))
    args = parser.parse_args()
    out_path = REPO / OUT_PATTERN.format(channel=args.channel)
    rows = [r for r in map(json.loads, MANIFEST.read_text(encoding="utf-8").splitlines()) if r["channel"] == args.channel]
    done = set()
    if out_path.exists():
        done = {json.loads(l)["id"] for l in out_path.read_text(encoding="utf-8").splitlines() if l.strip()}
    todo = [r for r in rows if r["id"] not in done]
    print(f"{args.channel}: {len(todo)} rows to decode, {len(done)} done", flush=True)

    giga = {name: gigaam_recognizer(path) for name, path in GIGAAM_MODELS.items() if name in args.engines}
    vosk = vosk_models() if "vosk" in args.engines else {}
    with out_path.open("a", encoding="utf-8") as sink:
        for number, row in enumerate(todo, 1):
            samples = read_samples(REPO / row["wav"])
            out = {k: row[k] for k in ("id", "channel", "duration")}
            for name, recognizer in giga.items():
                out[name] = decode_gigaam(recognizer, samples)
            if vosk:
                out["vosk"] = decode_vosk_timed(vosk, samples)
            sink.write(json.dumps(out, ensure_ascii=False) + "\n")
            sink.flush()
            if number % 50 == 0 or number == len(todo):
                print(f"  {number}/{len(todo)}", flush=True)


if __name__ == "__main__":
    main()
