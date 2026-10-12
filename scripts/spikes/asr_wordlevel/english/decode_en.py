"""Decode the English recogniser test set with one candidate engine (one thread, native).

Engines (models under models/asr/ or ~/.cache/vosk):
- `vosk_small`   vosk-model-small-en-us-0.15 (shipped baseline)
- `vosk_lgraph`  vosk-model-en-us-0.22-lgraph
- `vosk_kk`, `vosk_ru` the shipped kk/ru Vosk models (needed for the vote simulation)
- `fc_rnnt480`   NeMo FastConformer hybrid streaming EN, transducer head, 480 ms, int8 (OnlineRecognizer)
- `fc_rnnt480_fp32` the same transducer in fp32 (control for int8 loss; English sets only)
- `fc_ctc480`    same model family, CTC head, 480 ms, int8 (OnlineRecognizer)
- `moon_tiny`, `moon_base`  Moonshine tiny / base, sherpa "v2" quantized export (OfflineRecognizer)
- `gigaam220`, `gigaam600`  GigaAM-Multilingual CTC int8 (the kk/ru candidate, one model for kk+ru+en)
- `parakeet110`  parakeet-tdt_ctc-110m, CTC head, int8 (OfflineRecognizer)

Each decode records text, wall seconds, and whatever confidence the engine exposes: Vosk per-word
`conf` + times; sherpa per-token log-probabilities (`ys_probs` online / `ys_log_probs` offline) and
token timestamps when present.

    .venv/bin/python scripts/spikes/asr_wordlevel/english/decode_en.py ENGINE [--sets ...]

Output: data/asr_capture/english_decodes_<engine>.jsonl (resumable).
"""

from __future__ import annotations

import argparse
import json
import math
import time
import wave
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[4]
MODELS = REPO / "models/asr"
VOSK_CACHE = Path.home() / ".cache/vosk"
MANIFESTS = (REPO / "data/asr_capture/english_set.jsonl", REPO / "data/asr_capture/english_fleurs.jsonl")
SAMPLE_RATE = 16000
VOSK_CHUNK = 4000
ONLINE_CHUNK = 1600  # 0.1 s pushes, like a microphone worklet
ONLINE_TAIL = 0.8  # seconds of silence so the 480 ms right context flushes
LEAD_PAD = 0.3  # leading silence for EVERY engine: the streaming FastConformer drops the first word
                # of a stream that starts on speech (measured; see REPORT.md)
FC_RNNT = MODELS / "sherpa-onnx-nemo-streaming-fast-conformer-transducer-en-480ms-int8"
FC_RNNT_FP32 = MODELS / "sherpa-onnx-nemo-streaming-fast-conformer-transducer-en-480ms"  # fp32 control
FC_CTC = MODELS / "sherpa-onnx-nemo-streaming-fast-conformer-ctc-en-480ms-int8"
MOON = {"moon_tiny": MODELS / "sherpa-onnx-moonshine-tiny-en-quantized-2026-02-27",
        "moon_base": MODELS / "sherpa-onnx-moonshine-base-en-quantized-2026-02-27"}
GIGAAM = {"gigaam220": MODELS / "gigaam-ml/model.int8.onnx", "gigaam600": MODELS / "gigaam-ml/large/model.int8.onnx"}
GIGAAM_TOKENS = MODELS / "gigaam-ml/tokens.txt"
PARAKEET = MODELS / "sherpa-onnx-nemo-parakeet_tdt_ctc_110m-en-36000-int8"
VOSK_DIRS = {"vosk_small": VOSK_CACHE / "vosk-model-small-en-us-0.15", "vosk_lgraph": MODELS / "vosk-model-en-us-0.22-lgraph",
             "vosk_kk": VOSK_CACHE / "vosk-model-small-kz-0.42", "vosk_ru": VOSK_CACHE / "vosk-model-small-ru-0.22"}


def read_samples(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as handle:
        return np.frombuffer(handle.readframes(handle.getnframes()), dtype=np.int16)


class VoskEngine:
    def __init__(self, model_dir: Path):
        from vosk import Model, SetLogLevel

        SetLogLevel(-1)
        self.model = Model(str(model_dir))

    def decode(self, samples: np.ndarray) -> dict:
        from vosk import KaldiRecognizer

        rec = KaldiRecognizer(self.model, SAMPLE_RATE)
        rec.SetWords(True)
        words: list[dict] = []
        raw = samples.tobytes()
        for offset in range(0, len(raw), VOSK_CHUNK * 2):
            if rec.AcceptWaveform(raw[offset:offset + VOSK_CHUNK * 2]):
                words += json.loads(rec.Result()).get("result", [])
        words += json.loads(rec.FinalResult()).get("result", [])
        confs = [w.get("conf", 1.0) for w in words]
        return {"text": " ".join(w["word"] for w in words), "confidence": round(float(np.mean(confs)), 4) if confs else 0.0,
                "words": [{"w": w["word"], "s": w["start"], "e": w["end"], "c": round(w.get("conf", 1.0), 4)} for w in words]}


def token_summary(tokens, timestamps, logprobs) -> dict:
    out = {"tokens": list(tokens), "timestamps": [round(t, 2) for t in timestamps]}
    lp = [float(x) for x in logprobs] if logprobs is not None else []
    out["logprobs"] = [round(x, 4) for x in lp]
    out["confidence"] = round(float(np.mean([math.exp(x) for x in lp])), 4) if lp else None
    return out


class OnlineEngine:
    def __init__(self, kind: str):
        import sherpa_onnx

        if kind in ("fc_rnnt480", "fc_rnnt480_fp32"):
            d, q = (FC_RNNT, ".int8") if kind == "fc_rnnt480" else (FC_RNNT_FP32, "")
            self.rec = sherpa_onnx.OnlineRecognizer.from_transducer(
                tokens=str(d / "tokens.txt"), encoder=str(d / f"encoder{q}.onnx"),
                decoder=str(d / f"decoder{q}.onnx"), joiner=str(d / f"joiner{q}.onnx"),
                num_threads=1, sample_rate=SAMPLE_RATE, feature_dim=80, decoding_method="greedy_search")
        else:
            self.rec = sherpa_onnx.OnlineRecognizer.from_nemo_ctc(
                tokens=str(FC_CTC / "tokens.txt"), model=str(FC_CTC / "model.int8.onnx"), num_threads=1,
                sample_rate=SAMPLE_RATE, feature_dim=80, decoding_method="greedy_search")

    def decode(self, samples: np.ndarray) -> dict:
        stream = self.rec.create_stream()
        audio = samples.astype(np.float32) / 32768.0
        for offset in range(0, len(audio), ONLINE_CHUNK):
            stream.accept_waveform(SAMPLE_RATE, audio[offset:offset + ONLINE_CHUNK])
            while self.rec.is_ready(stream):
                self.rec.decode_stream(stream)
        stream.accept_waveform(SAMPLE_RATE, np.zeros(int(ONLINE_TAIL * SAMPLE_RATE), dtype=np.float32))
        stream.input_finished()
        while self.rec.is_ready(stream):
            self.rec.decode_stream(stream)
        result = self.rec.get_result_all(stream)
        return {"text": result.text.strip(), **token_summary(result.tokens, result.timestamps, result.ys_probs)}


class OfflineEngine:
    def __init__(self, kind: str):
        import sherpa_onnx

        if kind in GIGAAM:  # same settings as ../decode_cs.py::gigaam_recognizer
            self.rec = sherpa_onnx.OfflineRecognizer.from_nemo_ctc(
                model=str(GIGAAM[kind]), tokens=str(GIGAAM_TOKENS), num_threads=1,
                sample_rate=SAMPLE_RATE, feature_dim=64, decoding_method="greedy_search")
        elif kind in MOON:
            d = MOON[kind]
            self.rec = sherpa_onnx.OfflineRecognizer.from_moonshine_v2(
                encoder=str(d / "encoder_model.ort"), decoder=str(d / "decoder_model_merged.ort"),
                tokens=str(d / "tokens.txt"), num_threads=1)
        else:
            self.rec = sherpa_onnx.OfflineRecognizer.from_nemo_ctc(
                model=str(PARAKEET / "model.int8.onnx"), tokens=str(PARAKEET / "tokens.txt"), num_threads=1,
                sample_rate=SAMPLE_RATE, feature_dim=80, decoding_method="greedy_search")

    def decode(self, samples: np.ndarray) -> dict:
        stream = self.rec.create_stream()
        stream.accept_waveform(SAMPLE_RATE, samples.astype(np.float32) / 32768.0)
        self.rec.decode_stream(stream)
        r = stream.result
        try:
            tokens = r.tokens
        except UnicodeDecodeError:  # Moonshine's byte-level tokens can split a UTF-8 character
            tokens = []
        return {"text": r.text.strip(), **token_summary(tokens, r.timestamps, r.ys_log_probs)}


def build(engine: str):
    if engine in VOSK_DIRS:
        return VoskEngine(VOSK_DIRS[engine])
    if engine in ("fc_rnnt480", "fc_rnnt480_fp32", "fc_ctc480"):
        return OnlineEngine(engine)
    if engine in MOON or engine in GIGAAM or engine == "parakeet110":
        return OfflineEngine(engine)
    raise SystemExit(f"unknown engine {engine}")


def decode_files(engine: str) -> list[Path]:
    """The engine's decode file plus its `--shard` files (not other engines sharing a prefix)."""
    base = REPO / "data/asr_capture"
    return sorted([base / f"english_decodes_{engine}.jsonl", *base.glob(f"english_decodes_{engine}__s*.jsonl")])


def load_rows(sets: list[str] | None) -> list[dict]:
    rows = []
    for manifest in MANIFESTS:
        if manifest.exists():
            rows += [json.loads(l) for l in manifest.read_text(encoding="utf-8").splitlines() if l.strip()]
    return [r for r in rows if not sets or r["set"] in sets]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("engine")
    parser.add_argument("--sets", nargs="*")
    parser.add_argument("--shard", default="0/1", help="k/n: decode every n-th row from k, into its own file")
    args = parser.parse_args()
    k, n = map(int, args.shard.split("/"))
    suffix = "" if n == 1 else f"__s{k}"
    out = REPO / f"data/asr_capture/english_decodes_{args.engine}{suffix}.jsonl"
    done = set()
    for path in filter(Path.exists, decode_files(args.engine)):
        done |= {json.loads(l)["id"] for l in path.read_text(encoding="utf-8").splitlines() if l.strip()}
    todo = [r for i, r in enumerate(load_rows(args.sets)) if i % n == k and r["id"] not in done]
    print(f"{args.engine}: {len(todo)} to decode, {len(done)} done", flush=True)
    engine = build(args.engine)
    with out.open("a", encoding="utf-8") as sink:
        for number, row in enumerate(todo, 1):
            samples = np.concatenate([np.zeros(int(LEAD_PAD * SAMPLE_RATE), np.int16), read_samples(REPO / row["wav"])])
            started = time.perf_counter()
            result = engine.decode(samples)
            seconds = time.perf_counter() - started
            sink.write(json.dumps({"id": row["id"], "set": row["set"], "duration": row["duration"],
                                   "seconds": round(seconds, 4), **result}, ensure_ascii=False) + "\n")
            if number % 100 == 0:
                sink.flush()
                print(f"  {number}/{len(todo)}", flush=True)
    print(f"{args.engine}: done", flush=True)


if __name__ == "__main__":
    main()
