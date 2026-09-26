"""The shipped runtime needs none of the optional extras in pyproject.toml.

The server image used to install CPU torch, transformers, faster-whisper and Streamlit (several
GB) although the product path is the int8 ONNX embedder + sklearn heads. The runtime
dependencies are now only what that path needs; this test keeps it that way. A subprocess
blocks every extra's top-level module *and records each attempt* -- `/api/analyze` degrades to
the mock backend on any exception, so an import that fails silently would otherwise pass --
then imports the server and the deploy bootstrap and scores one transcript.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tomllib
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]

# Top-level modules of the optional extras (dev tooling excepted: the test itself needs httpx).
_EXTRA_MODULES = (
    "torch", "transformers", "captum", "sentence_transformers", "hdbscan",  # research
    "streamlit", "faster_whisper",                                          # harness
    "vosk", "streamlit_webrtc", "sounddevice",                              # live
    "google.genai",                                                         # cloud
    "onnx",                                                                 # quant (not onnxruntime)
)

_PROBE = r"""
import importlib.abc, importlib.util, json, sys
blocked, attempts = set(json.loads(sys.argv[1])), []

class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name in blocked or name.split(".")[0] in blocked:
            attempts.append(name)
            raise ImportError(f"{name} is an optional extra")
        return None

sys.meta_path.insert(0, Block())
from fastapi.testclient import TestClient
import qorgan.api, qorgan.classifier.linear_train  # server + the bootstrap's retrain path
spec = importlib.util.spec_from_file_location("deploy_bootstrap", sys.argv[2])
spec.loader.exec_module(importlib.util.module_from_spec(spec))
response = TestClient(qorgan.api.app).post(
    "/api/analyze", json={"transcript": "Здравствуйте, служба безопасности банка. Назовите код из СМС.", "locale": "ru"}
)
print(json.dumps({"attempts": attempts, "status": response.status_code, "backend": response.json().get("backend")}))
"""


def _run_probe() -> dict:
    completed = subprocess.run(
        [sys.executable, "-c", _PROBE, json.dumps(_EXTRA_MODULES), str(_REPO / "scripts" / "deploy_bootstrap.py")],
        capture_output=True, text=True, cwd=_REPO, timeout=300,
    )
    assert completed.returncode == 0, completed.stderr[-2000:]
    return json.loads(completed.stdout.strip().splitlines()[-1])


def test_runtime_dependencies_exclude_the_optional_extras():
    project = tomllib.loads((_REPO / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    runtime = {dep.split(">")[0].split("=")[0].split("[")[0].strip().lower() for dep in project["dependencies"]}
    heavy = {"torch", "transformers", "captum", "sentence-transformers", "hdbscan", "streamlit", "faster-whisper", "google-genai"}
    assert not runtime & heavy, f"move {sorted(runtime & heavy)} to an extra"


def test_server_and_bootstrap_never_import_an_optional_extra():
    probe = _run_probe()
    assert probe["attempts"] == [], f"runtime path imported optional extras: {probe['attempts']}"
    assert probe["status"] == 200


def test_the_shipped_backend_scores_without_extras_when_the_model_is_present():
    if not (_REPO / "models" / "linear").exists():
        import pytest

        pytest.skip("linear model not bootstrapped in this checkout")
    assert _run_probe()["backend"] == "linear"
