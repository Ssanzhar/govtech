"""Dump embeddings of the parity-fixture texts for a given ONNX model dir (Python ORT side of
the cross-runtime drift probe, PLAN B8). Pair with `node tests_js/tools/runtime_drift.mjs`.

Run: `python scripts/embed_probe.py --model-dir site/models/<id> --out /tmp/probe.json`
"""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path

from qorgan.classifier.embed import OnnxEmbedder

REPO_ROOT = Path(__file__).resolve().parents[1]
_E5_PREFIX = "query: "


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--fixtures", type=Path, default=REPO_ROOT / "tests_js" / "fixtures" / "parity.json")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    texts = [c["transcript"] for c in json.loads(args.fixtures.read_text(encoding="utf-8"))["cases"]]
    embedder = OnnxEmbedder.from_dir(args.model_dir)
    vectors = embedder.encode([_E5_PREFIX + t for t in texts])
    payload = {
        "model_dir": str(args.model_dir), "texts": texts,
        "embeddings": [base64.b64encode(v.tobytes()).decode("ascii") for v in vectors],
    }
    args.out.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    print(f"{len(texts)} embeddings -> {args.out}")


if __name__ == "__main__":
    main()
