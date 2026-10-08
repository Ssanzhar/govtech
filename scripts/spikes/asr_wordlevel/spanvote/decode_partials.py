"""Decode a manifest with the three shipped Vosk models, logging finals AND partials with words.

Each model gets `SetWords(True)` + `SetPartialWords(True)`; audio is fed in 0.25 s chunks (the
same chunk as `../decode_cs.py`). After every chunk we log either the endpoint final (`Result()`)
or the partial (`PartialResult()`), stamped with its arrival time = audio seconds fed so far.
Consecutive identical partials are logged once (the screen would not change); `n_partials`
counts them all. Single thread per process (Vosk decodes on the calling thread).

    python .../decode_partials.py --set dev  [--shard 0/3]   # -> data/asr_capture/spanvote_dev_decodes[.k].jsonl
    python .../decode_partials.py --set test [--shard 0/3]   # 440 test rows (cs_set.jsonl)
"""

from __future__ import annotations

import argparse
import json
import time
import wave
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
MANIFESTS = {"dev": REPO / "data/asr_capture/spanvote_dev_set.jsonl", "test": REPO / "data/asr_capture/cs_set.jsonl"}
OUTS = {"dev": REPO / "data/asr_capture/spanvote_dev_decodes", "test": REPO / "data/asr_capture/spanvote_test_decodes"}
VOSK_MODELS = {"kk": "vosk-model-small-kz-0.42", "ru": "vosk-model-small-ru-0.22", "en": "vosk-model-small-en-us-0.15"}
SAMPLE_RATE = 16000
CHUNK_SECONDS = 0.25
CHUNK_BYTES = int(CHUNK_SECONDS * SAMPLE_RATE) * 2


def read_raw(path: Path) -> bytes:
    with wave.open(str(path), "rb") as handle:
        return handle.readframes(handle.getnframes())


def slim(words: list[dict]) -> list[list]:
    return [[w["word"], round(w["start"], 3), round(w["end"], 3), round(w.get("conf", 1.0), 4)] for w in words]


def decode_one(model, raw: bytes, partial_words: bool = True) -> dict:
    from vosk import KaldiRecognizer

    rec = KaldiRecognizer(model, SAMPLE_RATE)
    rec.SetWords(True)
    rec.SetPartialWords(partial_words)
    finals, partials, n_partials, last = [], [], 0, None
    started = time.perf_counter()
    for offset in range(0, len(raw), CHUNK_BYTES):
        t = round(min(offset + CHUNK_BYTES, len(raw)) / 2 / SAMPLE_RATE, 3)
        if rec.AcceptWaveform(raw[offset:offset + CHUNK_BYTES]):
            finals.append({"t": t, "words": slim(json.loads(rec.Result()).get("result", []))})
            last = None
        else:
            words = slim(json.loads(rec.PartialResult()).get("partial_result", []))
            n_partials += 1
            if words != last:
                partials.append({"t": t, "words": words})
                last = words
    t_end = round(len(raw) / 2 / SAMPLE_RATE, 3)
    tail = slim(json.loads(rec.FinalResult()).get("result", []))
    if tail:
        finals.append({"t": t_end, "words": tail})
    seconds = time.perf_counter() - started
    return {"finals": finals, "partials": partials, "n_partials": n_partials, "seconds": round(seconds, 4)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--set", choices=sorted(MANIFESTS), required=True)
    parser.add_argument("--shard", default="0/1")
    args = parser.parse_args()
    k, n = map(int, args.shard.split("/"))
    from vosk import Model, SetLogLevel

    SetLogLevel(-1)
    models = {lang: Model(model_name=name) for lang, name in VOSK_MODELS.items()}
    rows = [json.loads(l) for l in MANIFESTS[args.set].read_text(encoding="utf-8").splitlines() if l.strip()]
    rows = rows[k::n]
    out = Path(f"{OUTS[args.set]}.{k}.jsonl")
    done = set()
    if out.exists():
        done = {(r["id"], r["condition"], r["channel"]) for r in map(json.loads, out.read_text(encoding="utf-8").splitlines())}
    todo = [r for r in rows if (r["id"], r["condition"], r["channel"]) not in done]
    print(f"shard {k}/{n}: {len(todo)} to decode, {len(done)} done", flush=True)
    with out.open("a", encoding="utf-8") as sink:
        for i, row in enumerate(todo, 1):
            raw = read_raw(REPO / row["wav"])
            rec = {key: row[key] for key in ("id", "condition", "channel", "duration")}
            rec["models"] = {lang: decode_one(m, raw) for lang, m in models.items()}
            sink.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if i % 50 == 0:
                sink.flush()
                print(f"  {i}/{len(todo)}", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
