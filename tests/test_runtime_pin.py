"""The server must run the SAME ONNX Runtime version as the device runtime (ADR D32): with
matching versions the int8 embeddings are bit-identical (cosine 1.00000 on the 200-case gate);
with Python 1.27 vs Node 1.21 they drifted to cosine 0.958 min and flipped decisions. The pin
lives in pyproject.toml; this test catches the two drifting apart on upgrade."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
_NODE_ORT = _REPO / "node_modules" / "onnxruntime-node" / "package.json"


def _major_minor(version: str) -> tuple[int, int]:
    match = re.match(r"(\d+)\.(\d+)", version)
    assert match, version
    return int(match.group(1)), int(match.group(2))


def test_pyproject_pins_onnxruntime_exactly():
    text = (_REPO / "pyproject.toml").read_text(encoding="utf-8")
    assert re.search(r'"onnxruntime==\d+\.\d+\.\*"', text), "onnxruntime must be pinned to the device runtime's major.minor"


@pytest.mark.skipif(not _NODE_ORT.exists(), reason="node_modules not installed")
def test_python_onnxruntime_matches_the_node_runtime_transformers_js_bundles():
    import onnxruntime

    node_version = json.loads(_NODE_ORT.read_text(encoding="utf-8"))["version"]
    assert _major_minor(onnxruntime.__version__) == _major_minor(node_version), (
        f"python onnxruntime {onnxruntime.__version__} vs onnxruntime-node {node_version}: "
        "embeddings will drift between server and device (ADR D32)"
    )
