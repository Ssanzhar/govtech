"""Provision everything the deployed demo needs — idempotent, safe on every start.

Run: `python scripts/deploy_bootstrap.py`. Each step is skipped when its artifact
already exists, so re-running (container restart, local dev) is a fast no-op.

1. Corpus splits  ← Hugging Face dataset `sanzh-ts/govtech_ds` (scrubbed, publishable).
2. Dialogue pool  ← concatenated splits stand in for the raw synthetic corpus (same
   `Dialogue` schema), which is deliberately unpublishable and absent on fresh clones.
3. Linear model   ← Hugging Face `sanzh-ts/govtech`; if the bundle fails its lexicon
   hash-validation (drift), retrain from the corpus — seconds on CPU.
4. Level-2 seeds  ← `demo_seed` + `analytics.pipeline`, deterministic (seed 42). The
   fabricated demo phone numbers live only in the running instance, never in git.
5. Embedder      ← `Xenova/multilingual-e5-base` int8 ONNX (+ tokenizer) self-hosted under
   site/models/ at the dir `QORGAN_EMBED_ONNX_DIR` names — the browser never contacts
   huggingface.co and the server runs the same file (ADR D17); plus the exported head
   weights (`site/models/weights.json`).
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PROCESSED = REPO_ROOT / "data" / "processed"
MODEL_DIR = REPO_ROOT / "models" / "linear"
DATASET_REPO = "sanzh-ts/govtech_ds"
MODEL_REPO = "sanzh-ts/govtech"
SPLIT_FILES = ("train.jsonl", "val.jsonl", "test.jsonl", "authored_heldout.jsonl", "ood.jsonl", "manifest.json")
DIALOGUE_POOL_SPLITS = ("train.jsonl", "val.jsonl", "test.jsonl")
# `real_heldout` was renamed `authored_heldout` (it is hand-written, not real calls --
# PLAN_2026-09 A2); Hub snapshots published before that still use the old name.
LEGACY_SPLIT_NAMES = {"authored_heldout.jsonl": "real_heldout.jsonl"}

# Scored once to prove the linear backend actually loads (also warms the embedder cache).
_PROBE_SNIPPET = (
    "import sys; from qorgan.classifier import predict; "
    "r = predict.score('Алло, это служба безопасности банка, назовите код из смс.', backend='linear'); "
    "sys.exit(0 if r.backend == 'linear' else 1)"
)


def _log(message: str) -> None:
    print(f"[bootstrap] {message}", flush=True)


def _run(args: list[str]) -> None:
    subprocess.run(args, check=True, cwd=REPO_ROOT)


def ensure_corpus() -> None:
    if (PROCESSED / "train.jsonl").exists():
        _log("corpus: present")
        return
    from huggingface_hub import snapshot_download

    _log(f"corpus: downloading {DATASET_REPO}")
    snapshot = Path(snapshot_download(DATASET_REPO, repo_type="dataset"))
    PROCESSED.mkdir(parents=True, exist_ok=True)
    for name in SPLIT_FILES:
        source = snapshot / name
        if not source.exists() and name in LEGACY_SPLIT_NAMES:
            source = snapshot / LEGACY_SPLIT_NAMES[name]  # dataset published before the rename
        if source.exists():
            (PROCESSED / name).write_bytes(source.read_bytes())
    _log("corpus: splits in place")


def ensure_dialogue_pool() -> None:
    from qorgan.data.generate import load_corpus_config

    pool_path = load_corpus_config().output_path
    if pool_path.exists():
        _log("dialogue pool: present")
        return
    lines: list[str] = []
    for name in DIALOGUE_POOL_SPLITS:
        split = PROCESSED / name
        if split.exists():
            lines.extend(line for line in split.read_text(encoding="utf-8").splitlines() if line.strip())
    if not lines:
        raise RuntimeError("no corpus splits available to build the dialogue pool")
    pool_path.parent.mkdir(parents=True, exist_ok=True)
    pool_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    _log(f"dialogue pool: {len(lines)} dialogues -> {pool_path}")


def _linear_backend_works() -> bool:
    # Probed in a subprocess so a retrain is picked up fresh (predict caches per process).
    result = subprocess.run([sys.executable, "-c", _PROBE_SNIPPET], cwd=REPO_ROOT)
    return result.returncode == 0


def ensure_model() -> None:
    if not (MODEL_DIR / "metadata.json").exists():
        from huggingface_hub import snapshot_download

        _log(f"model: downloading {MODEL_REPO}")
        try:
            snapshot_download(MODEL_REPO, local_dir=MODEL_DIR)
        except Exception as exc:  # private repo / offline — retraining covers it
            _log(f"model: download failed ({exc}); falling back to retrain")
    if _linear_backend_works():
        _log("model: linear backend OK")
        return
    _log("model: bundle unusable (hash drift or missing) — retraining from corpus")
    _run([sys.executable, "-m", "qorgan.classifier.linear_train"])
    if not _linear_backend_works():
        raise RuntimeError("linear backend still failing after retrain")
    _log("model: retrained, linear backend OK")


def ensure_l2_seeds() -> None:
    if (PROCESSED / "organizations.jsonl").exists():
        _log("L2 seeds: present")
        return
    if not os.environ.get("QORGAN_NUMBER_HMAC_KEY", "").strip():
        # Seeded numbers are stored as HMAC digests (ADR D14). Without the runtime key
        # (never baked into an image) seeding is deferred to the first start.
        _log("L2 seeds: skipped -- QORGAN_NUMBER_HMAC_KEY not set (seeds on first start with the key)")
        return
    _log("L2 seeds: seeding incidents + clustering (embeds ~500 transcripts on CPU)")
    _run([sys.executable, "scripts/demo_seed.py"])
    _run([sys.executable, "-m", "qorgan.analytics.pipeline"])
    _log("L2 seeds: organizations ready")


# The embedder the browser AND the server run (ADR D17): the Hub's dynamically-quantised
# int8 graph. Static (calibrated) quantisation was measured and rejected -- ADR D18.
EMBED_HUB_REPO = "Xenova/multilingual-e5-base"
EMBED_FILES = ("config.json", "tokenizer.json", "tokenizer_config.json", "onnx/model_quantized.onnx")
SITE_MODELS = REPO_ROOT / "site" / "models"


def _model_dir_complete(model_dir: Path) -> bool:
    return all((model_dir / name).exists() for name in EMBED_FILES)


def _download_embedder(target: Path) -> None:
    from huggingface_hub import hf_hub_download

    for name in EMBED_FILES:
        destination = target / name
        if destination.exists():
            continue
        _log(f"web model: downloading {EMBED_HUB_REPO}/{name}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(Path(hf_hub_download(EMBED_HUB_REPO, name)).read_bytes())


def ensure_web_model() -> None:
    """Self-host the int8 embedder at the dir the config names (server + browser load the
    same files), and copy the exported head weights next to it."""
    from qorgan.config import get_config
    from qorgan.web.client_config import web_model_id

    target = get_config().embed_onnx_dir
    model_id = web_model_id(target)
    if _model_dir_complete(target):
        _log(f"web model: {model_id} present")
    elif model_id == EMBED_HUB_REPO:
        _download_embedder(target)
        _log(f"web model: {model_id} in place")
    else:
        _log(f"web model: WARNING {target} is incomplete and is not the Hub graph -- nothing downloaded")
    weights = MODEL_DIR / "web" / "weights.json"
    if weights.exists():
        (SITE_MODELS / "weights.json").write_bytes(weights.read_bytes())
        _log("web model: head weights in place")
    else:
        _log("web model: no exported head weights (run python -m qorgan.classifier.web_bundle)")


def main() -> None:
    os.environ.setdefault("QORGAN_CLASSIFIER_BACKEND", "linear")
    ensure_corpus()
    ensure_dialogue_pool()
    ensure_model()
    ensure_l2_seeds()
    ensure_web_model()
    _log("done — serve with: uvicorn qorgan.api:app --host 0.0.0.0 --port $PORT")


if __name__ == "__main__":
    main()
