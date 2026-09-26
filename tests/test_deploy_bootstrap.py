"""Order of the deploy bootstrap steps (`scripts/deploy_bootstrap.py`).

On a fresh clone the model probe, a fallback retrain and the Level-2 seeding all embed
text through the int8 ONNX graph under `site/models/`. The embedder must therefore be in
place before any of them runs -- otherwise a clean self-deploy (and the Docker build)
fails with ONNX `NO_SUCHFILE` even though every download succeeded.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "deploy_bootstrap.py"


def _load_bootstrap():
    spec = importlib.util.spec_from_file_location("deploy_bootstrap", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_embedder_is_provisioned_before_anything_embeds(monkeypatch):
    bootstrap = _load_bootstrap()
    calls: list[str] = []
    for step in ("ensure_corpus", "ensure_dialogue_pool", "ensure_embedder", "ensure_model",
                 "ensure_l2_seeds", "ensure_web_weights", "ensure_asr_models"):
        monkeypatch.setattr(bootstrap, step, lambda step=step: calls.append(step))

    bootstrap.main()

    assert calls.index("ensure_embedder") < calls.index("ensure_model")
    assert calls.index("ensure_embedder") < calls.index("ensure_l2_seeds")
    # The browser's head weights are copied from the bundle, so only after it exists.
    assert calls.index("ensure_model") < calls.index("ensure_web_weights")


def test_corpus_tops_up_missing_splits_without_overwriting_local_ones(monkeypatch, tmp_path):
    bootstrap = _load_bootstrap()
    processed = tmp_path / "processed"
    processed.mkdir()
    (processed / "train.jsonl").write_text("local train\n", encoding="utf-8")  # e.g. rebuilt locally
    hub = tmp_path / "hub"
    hub.mkdir()
    for name in bootstrap.SPLIT_FILES:
        (hub / name).write_text(f"hub {name}\n", encoding="utf-8")
    monkeypatch.setattr(bootstrap, "PROCESSED", processed)
    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda *a, **k: str(hub))

    bootstrap.ensure_corpus()

    # The honest cross-generator split ships with every fresh setup (ADR D35).
    assert "shift.jsonl" in bootstrap.SPLIT_FILES
    assert (processed / "shift.jsonl").read_text(encoding="utf-8") == "hub shift.jsonl\n"
    assert (processed / "train.jsonl").read_text(encoding="utf-8") == "local train\n"
