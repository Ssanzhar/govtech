"""Static (calibrated) int8 quantisation of the embedder -- an EXPERIMENT, measured no-go.

RESULT (2026-09-17, ADR D18): the hypothesis below does not hold. With MinMax moving-average
calibration on 128 train texts, MatMul+Gather quantised and the 24 activation-activation
MatMuls left fp32, the static graph scored cosine 0.944 mean / 0.924 min vs fp32 on 88
transcripts (the Hub's dynamic graph: 0.992 / 0.981) AND still drifted between Python ORT
1.27 and onnxruntime-node 1.21: 0.992 mean / 0.971 min (dynamic: 0.985 / 0.970). The
cross-runtime floor is kernel/fusion differences, not run-time activation scales, so better
calibration cannot remove it. The shipped embedder stays the dynamic graph; the parity gate
stays decision-level (`tests_js/integration/embedding.test.mjs`). Kept as tooling for a
future attempt (percentile/entropy calibration, SmoothQuant-style outlier handling); it
needs `pip install -e ".[quant]"` and writes only under the gitignored site/models/qorgan/.

Hypothesis it tested: the Hub's dynamically-quantised `model_quantized.onnx` derives
activation scales at run time, so its outputs drift across ONNX Runtime implementations
(Python vs node vs web; measured cosine 0.985-0.994) and even across batch compositions.
Static quantisation bakes calibrated scales into the graph (QDQ format), so every runtime
would perform the same integer arithmetic on the same scales -- the property
"server == device" needs.

What it does:
1. downloads the fp32 export `Xenova/multilingual-e5-base/onnx/model.onnx` (1.1 GB, cached);
2. calibrates on `--calibration-samples` transcripts + utterances from the training split
   (one text per run, like inference). Default method: MinMax with a moving average across
   samples (shape-agnostic, outlier-damped); `--method percentile` needs `--pad-to N`
   because ORT's percentile/entropy collectors stack activations across samples;
3. quantises MatMul weights (per-channel int8) + activations (uint8) into
   `site/models/<QORGAN_STATIC_MODEL_ID>/onnx/model_quantized.onnx`, next to copies of the
   tokenizer/config files so both transformers.js and `OnnxEmbedder` can load the dir;
4. reports size and cosine vs fp32 on a held-out sample.

Run: `python scripts/quantize_embedder.py [--calibration-samples 128] [--percentile 99.99]`
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np
from huggingface_hub import hf_hub_download
from onnxruntime.quantization import CalibrationDataReader, CalibrationMethod, QuantFormat, QuantType, quantize_static
from onnxruntime.quantization.shape_inference import quant_pre_process
from tokenizers import Tokenizer

from qorgan.classifier.embed import ONNX_MAX_LENGTH, OnnxEmbedder
from qorgan.data.schema import Dialogue

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_REPO = "Xenova/multilingual-e5-base"
SOURCE_FILES = ("config.json", "tokenizer.json", "tokenizer_config.json")
STATIC_MODEL_ID = "qorgan/multilingual-e5-base-static-int8"
# Per-channel DequantizeLinear (the `axis` attribute) needs opset >= 13; the Hub export is older.
MIN_OPSET = 13
# MatMul = the transformer's weights; Gather = the 250k x 768 token-embedding table (768 MB in
# fp32 -- without it the "int8" graph would be 3x larger than the Hub's).
QUANTIZED_OPS = ["MatMul", "Gather"]
_E5_PREFIX = "query: "
_HELDOUT_EVAL_SAMPLES = 40


class TextCalibrationReader(CalibrationDataReader):
    """Feeds tokenised texts one at a time (the inference batch size)."""

    def __init__(self, tokenizer: Tokenizer, texts: list[str], input_names: set[str], pad_to: int | None = None) -> None:
        self._tokenizer = tokenizer
        self._texts = iter(texts)
        self._input_names = input_names
        self._pad_to = pad_to
        self.count = 0

    def get_next(self) -> dict | None:
        text = next(self._texts, None)
        if text is None:
            return None
        enc = self._tokenizer.encode(_E5_PREFIX + text)
        ids_list, mask_list = list(enc.ids), list(enc.attention_mask)
        if self._pad_to is not None:
            ids_list = (ids_list + [1] * self._pad_to)[: self._pad_to]  # <pad> = 1
            mask_list = (mask_list + [0] * self._pad_to)[: self._pad_to]
        ids = np.asarray([ids_list], dtype=np.int64)
        feed = {"input_ids": ids, "attention_mask": np.asarray([mask_list], dtype=np.int64)}
        if "token_type_ids" in self._input_names:
            feed["token_type_ids"] = np.zeros_like(ids)
        self.count += 1
        return feed


def _load_texts(path: Path, limit: int, *, seed: int) -> list[str]:
    dialogues = [Dialogue.model_validate_json(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    rng = np.random.default_rng(seed)
    rng.shuffle(dialogues)
    texts: list[str] = []
    for d in dialogues:
        texts.append(d.transcript())
        texts.extend(u.text for u in d.utterances)
        if len(texts) >= limit:
            break
    return texts[:limit]


def _activation_matmuls(path: Path) -> list[str]:
    """MatMul nodes whose inputs are both activations (attention QK^T and attn.V): their
    ranges swing per input, so static per-tensor scales wreck them. Dynamic quantisation
    leaves them fp32 too -- we do the same and quantise only the weight projections."""
    import onnx

    model = onnx.load(str(path), load_external_data=False)
    initializers = {init.name for init in model.graph.initializer}
    return [
        node.name
        for node in model.graph.node
        if node.op_type == "MatMul" and not any(name in initializers for name in node.input)
    ]


def _ensure_opset(path: Path, minimum: int) -> None:
    import onnx
    from onnx import version_converter

    model = onnx.load(str(path))
    current = next((o.version for o in model.opset_import if o.domain in ("", "ai.onnx")), None)
    if current is not None and current < minimum:
        onnx.save(version_converter.convert_version(model, minimum), str(path))
        print(f"opset {current} -> {minimum}")


def _cosine_report(a: np.ndarray, b: np.ndarray) -> str:
    cos = (a * b).sum(axis=1)
    return f"cosine vs fp32: mean={cos.mean():.4f} min={cos.min():.4f}"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Statically quantise the e5 embedder.")
    parser.add_argument("--calibration-samples", type=int, default=128)
    parser.add_argument("--method", choices=("minmax", "percentile"), default="minmax")
    parser.add_argument("--percentile", type=float, default=99.99)
    parser.add_argument("--pad-to", type=int, default=None, help="fixed token length (required for --method percentile)")
    parser.add_argument("--exclude-activation-matmuls", action=argparse.BooleanOptionalAction, default=True,
                        help="keep attention QK^T / attn.V MatMuls in fp32 (default on)")
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "site" / "models" / STATIC_MODEL_ID)
    parser.add_argument("--processed-dir", type=Path, default=REPO_ROOT / "data" / "processed")
    args = parser.parse_args(argv)

    fp32_path = Path(hf_hub_download(SOURCE_REPO, "onnx/model.onnx"))
    tokenizer = Tokenizer.from_file(hf_hub_download(SOURCE_REPO, "tokenizer.json"))
    tokenizer.enable_truncation(ONNX_MAX_LENGTH)
    print(f"fp32 graph: {fp32_path} ({fp32_path.stat().st_size / 1e6:.0f} MB)")

    work = args.out_dir / "onnx"
    work.mkdir(parents=True, exist_ok=True)
    preprocessed = work / "model_fp32_preprocessed.onnx"
    if not preprocessed.exists():
        t0 = time.perf_counter()
        quant_pre_process(str(fp32_path), str(preprocessed), skip_symbolic_shape=True)
        _ensure_opset(preprocessed, MIN_OPSET)
        print(f"pre-processed in {time.perf_counter() - t0:.0f}s")

    import onnxruntime as ort

    input_names = {i.name for i in ort.InferenceSession(str(preprocessed), providers=["CPUExecutionProvider"]).get_inputs()}
    calibration = _load_texts(args.processed_dir / "train.jsonl", args.calibration_samples, seed=42)
    if args.method == "percentile" and args.pad_to is None:
        raise SystemExit("--method percentile requires --pad-to (ORT stacks activations across samples)")
    excluded = _activation_matmuls(preprocessed) if args.exclude_activation_matmuls else []
    print(f"leaving {len(excluded)} activation-activation MatMuls in fp32")
    reader = TextCalibrationReader(tokenizer, calibration, input_names, pad_to=args.pad_to)
    target = work / "model_quantized.onnx"
    method = CalibrationMethod.Percentile if args.method == "percentile" else CalibrationMethod.MinMax
    extra = {"ActivationSymmetric": False, "WeightSymmetric": True}
    extra.update({"CalibPercentile": args.percentile} if args.method == "percentile" else {"CalibMovingAverage": True})
    t0 = time.perf_counter()
    quantize_static(
        str(preprocessed),
        str(target),
        reader,
        quant_format=QuantFormat.QDQ,
        op_types_to_quantize=QUANTIZED_OPS,
        per_channel=True,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        calibrate_method=method,
        nodes_to_exclude=excluded,
        extra_options=extra,
    )
    preprocessed.unlink()
    print(f"quantised with {reader.count} calibration texts in {time.perf_counter() - t0:.0f}s -> {target} ({target.stat().st_size / 1e6:.0f} MB)")

    for name in SOURCE_FILES:
        shutil.copyfile(hf_hub_download(SOURCE_REPO, name), args.out_dir / name)
    (args.out_dir / "QUANTIZATION.json").write_text(json.dumps({
        "source": SOURCE_REPO, "format": "QDQ", "ops": QUANTIZED_OPS, "opset": MIN_OPSET, "weights": "int8 per-channel symmetric",
        "activations": "uint8 per-tensor asymmetric",
        "fp32_nodes": {"activation_matmuls_excluded": args.exclude_activation_matmuls, "count": len(excluded)},
        "calibration": {"method": args.method, "percentile": args.percentile if args.method == "percentile" else None,
                        "moving_average": args.method == "minmax", "pad_to": args.pad_to, "samples": reader.count, "split": "train"},
    }, indent=2), encoding="utf-8")

    heldout = _load_texts(args.processed_dir / "authored_heldout.jsonl", _HELDOUT_EVAL_SAMPLES, seed=7)
    fp32 = OnnxEmbedder(session=ort.InferenceSession(str(fp32_path), providers=["CPUExecutionProvider"]), tokenizer=Tokenizer.from_file(str(args.out_dir / "tokenizer.json")))
    static = OnnxEmbedder.from_dir(args.out_dir)
    prefixed = [_E5_PREFIX + t for t in heldout]
    t0 = time.perf_counter(); a = static.encode(prefixed); dt = (time.perf_counter() - t0) / len(prefixed)
    b = fp32.encode(prefixed)
    print(f"static int8: {_cosine_report(a, b)} | {1000 * dt:.0f} ms/text")


if __name__ == "__main__":
    main()
