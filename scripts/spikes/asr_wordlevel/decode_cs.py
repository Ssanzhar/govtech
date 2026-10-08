"""Decode the code-switched test set with every candidate engine (word-level STT track).

Engines:
- `gigaam220` / `gigaam600` -- GigaAM-Multilingual CTC int8 via sherpa-onnx (one model, kk+ru+en),
  one decode per utterance, ONE thread (the browser's WASM build is single-threaded; native
  single-thread speed is an optimistic proxy for it).
- `vosk` -- the three shipped Vosk models (kk 0.42, ru 0.22, en 0.15), each with word timings,
  plus the shipped per-utterance vote (a Python replica of `site/core/asr.js::voteFinal`:
  confidence minus the English handicap, ties within 0.02 broken by the Kazakh-letter share).

    python scripts/spikes/asr_wordlevel/decode_cs.py            # -> data/asr_capture/cs_decodes.jsonl

Resumable: rows already in the output are skipped.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import wave
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))

MANIFEST = REPO / "data/asr_capture/cs_set.jsonl"
OUT = REPO / "data/asr_capture/cs_decodes.jsonl"
GIGAAM_DIR = REPO / "models/asr/gigaam-ml"
GIGAAM_MODELS = {"gigaam220": GIGAAM_DIR / "model.int8.onnx", "gigaam600": GIGAAM_DIR / "large/model.int8.onnx"}
VOSK_MODELS = {"kk": "vosk-model-small-kz-0.42", "ru": "vosk-model-small-ru-0.22", "en": "vosk-model-small-en-us-0.15"}
VOSK_ORDER = ("kk", "ru", "en")
EN_HANDICAP = 0.15
TIE_EPSILON = 0.02
KAZAKH_LETTERS = set("әғқңөұүһі")
KAZAKH_SHARE_FOR_TIE = 0.08
SAMPLE_RATE = 16000
CHUNK = 4000


def read_samples(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as handle:
        return np.frombuffer(handle.readframes(handle.getnframes()), dtype=np.int16)


def gigaam_recognizer(model: Path):
    import sherpa_onnx

    return sherpa_onnx.OfflineRecognizer.from_nemo_ctc(
        model=str(model), tokens=str(GIGAAM_DIR / "tokens.txt"), num_threads=1,
        sample_rate=SAMPLE_RATE, feature_dim=64, decoding_method="greedy_search")


def decode_gigaam(recognizer, samples: np.ndarray) -> dict:
    started = time.perf_counter()
    stream = recognizer.create_stream()
    stream.accept_waveform(SAMPLE_RATE, samples.astype(np.float32) / 32768.0)
    recognizer.decode_stream(stream)
    elapsed = time.perf_counter() - started
    return {"text": stream.result.text.strip(), "seconds": round(elapsed, 4)}


def vosk_models():
    from vosk import Model, SetLogLevel

    SetLogLevel(-1)
    return {lang: Model(model_name=name) for lang, name in VOSK_MODELS.items()}


def decode_vosk_one(model, samples: np.ndarray) -> dict:
    from vosk import KaldiRecognizer

    rec = KaldiRecognizer(model, SAMPLE_RATE)
    rec.SetWords(True)
    words: list[dict] = []
    raw = samples.tobytes()
    for offset in range(0, len(raw), CHUNK * 2):
        if rec.AcceptWaveform(raw[offset:offset + CHUNK * 2]):
            words += json.loads(rec.Result()).get("result", [])
    words += json.loads(rec.FinalResult()).get("result", [])
    text = " ".join(w["word"] for w in words)
    conf = float(np.mean([w.get("conf", 1.0) for w in words])) if words else 0.0
    return {"text": text, "confidence": round(conf, 4),
            "words": [{"word": w["word"], "start": w["start"], "end": w["end"], "conf": round(w.get("conf", 1.0), 4)} for w in words]}


def kazakh_share(text: str) -> float:
    letters = [c for c in text if c.isalpha()]
    return sum(c in KAZAKH_LETTERS for c in letters) / len(letters) if letters else 0.0


def vote_utterance(hyps: dict) -> str:
    """Replica of `voteFinal(hypotheses, preferred="kk", {en: 0.15})`."""
    ranked = sorted(((hyps[l]["confidence"] - (EN_HANDICAP if l == "en" else 0.0), l)
                     for l in VOSK_ORDER if hyps[l]["text"]), reverse=True)
    if not ranked:
        return "kk"
    best_rank, best = ranked[0]
    tied = [l for r, l in ranked if best_rank - r <= TIE_EPSILON]
    if len(tied) > 1 and "kk" in tied and "ru" in tied:
        return "kk" if kazakh_share(hyps["kk"]["text"]) >= KAZAKH_SHARE_FOR_TIE else "ru"
    if len(tied) > 1:
        return next(l for l in VOSK_ORDER if l in tied)
    return best


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--engines", nargs="+", default=["gigaam220", "gigaam600", "vosk"])
    args = parser.parse_args()
    rows = [json.loads(l) for l in MANIFEST.read_text(encoding="utf-8").splitlines() if l.strip()]
    done = set()
    if args.out.exists():
        done = {(r["id"], r["condition"], r["channel"]) for r in map(json.loads, args.out.read_text(encoding="utf-8").splitlines())}
    todo = [r for r in rows if (r["id"], r["condition"], r["channel"]) not in done]
    print(f"{len(todo)} rows to decode, {len(done)} done", flush=True)

    giga = {name: gigaam_recognizer(path) for name, path in GIGAAM_MODELS.items() if name in args.engines}
    vosk = vosk_models() if "vosk" in args.engines else {}
    with args.out.open("a", encoding="utf-8") as sink:
        for number, row in enumerate(todo, 1):
            samples = read_samples(REPO / row["wav"])
            out = {k: row[k] for k in ("id", "condition", "channel", "duration")}
            for name, recognizer in giga.items():
                out[name] = decode_gigaam(recognizer, samples)
            if vosk:
                hyps = {lang: decode_vosk_one(model, samples) for lang, model in vosk.items()}
                winner = vote_utterance(hyps)
                out["vosk"] = {"per_model": hyps, "winner": winner, "text": hyps[winner]["text"]}
            sink.write(json.dumps(out, ensure_ascii=False) + "\n")
            if number % 40 == 0 or number == len(todo):
                sink.flush()
                print(f"  {number}/{len(todo)}", flush=True)


if __name__ == "__main__":
    main()
