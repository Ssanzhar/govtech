"""Provision everything the deployed demo needs — idempotent, safe on every start.

Run: `python scripts/deploy_bootstrap.py`. Each step is skipped when its artifact
already exists, so re-running (container restart, local dev) is a fast no-op.

1. Corpus splits  ← Hugging Face dataset `sanzh-ts/govtech_ds` (scrubbed, publishable).
2. Dialogue pool  ← concatenated splits stand in for the raw synthetic corpus (same
   `Dialogue` schema), which is deliberately unpublishable and absent on fresh clones.
3. Embedder      ← `Xenova/multilingual-e5-base` int8 ONNX (+ tokenizer) self-hosted under
   site/models/ at the dir `QORGAN_EMBED_ONNX_DIR` names — the browser never contacts
   huggingface.co and the server runs the same file (ADR D17). Provisioned before steps
   4–5 because the model probe, a fallback retrain and the seeding all embed through it.
4. Linear model   ← Hugging Face `sanzh-ts/govtech`; if the bundle fails its lexicon
   hash-validation (drift), retrain from the corpus — seconds on CPU.
5. Level-2 seeds  ← `demo_seed` + `analytics.pipeline`, deterministic (seed 42). The
   fabricated demo phone numbers live only in the running instance, never in git.
   Then the bundle's exported head weights are copied to `site/models/weights.json`.
6. Speech models   ← the small Vosk KK + RU models as USTAR tarballs under site/models/vosk/
   for the on-device microphone mode (PLAN B9, ADR D25); ~106 MB, from ~/.cache/vosk or
   the official alphacephei zips. Then the GigaAM endpoint pass (ADR D64, optional): the
   pinned sherpa-onnx browser runtime under site/vendor/sherpa-onnx/ (from a local
   `scripts/sherpa_wasm/build.sh` output or a mirror of it) and the 225 MB model under
   site/models/gigaam/, every file sha256-checked; without them the page uses the Vosk vote.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PROCESSED = REPO_ROOT / "data" / "processed"
MODEL_DIR = REPO_ROOT / "models" / "linear"
DATASET_REPO = "sanzh-ts/govtech_ds"
MODEL_REPO = "sanzh-ts/govtech"
SPLIT_FILES = ("train.jsonl", "val.jsonl", "test.jsonl", "authored_heldout.jsonl", "ood.jsonl", "adversarial.jsonl", "adversarial_legit.jsonl", "shift.jsonl", "shift.manifest.json", "manifest.json")
DIALOGUE_POOL_SPLITS = ("train.jsonl", "val.jsonl", "test.jsonl")
# `real_heldout` was renamed `authored_heldout` (it is hand-written, not real calls --
# PLAN_2026-09 A2); Hub snapshots published before that still use the old name.
LEGACY_SPLIT_NAMES = {"authored_heldout.jsonl": "real_heldout.jsonl"}

# Scored once to prove the linear backend actually loads (also warms the embedder cache).
# Prints which heads the linear backend serves ("bundle" | "web"); exits non-zero if neither.
_PROBE_SNIPPET = (
    "import sys; from qorgan.classifier import predict; "
    "r = predict.score('Алло, это служба безопасности банка, назовите код из смс.', backend='linear'); "
    "sys.exit(1) if r.backend != 'linear' else print(predict.linear_model_source())"
)


def _log(message: str) -> None:
    print(f"[bootstrap] {message}", flush=True)


def _run(args: list[str]) -> None:
    subprocess.run(args, check=True, cwd=REPO_ROOT)


def ensure_corpus() -> None:
    missing = [name for name in SPLIT_FILES if not (PROCESSED / name).exists()]
    if not missing:
        _log("corpus: present")
        return
    from huggingface_hub import snapshot_download

    _log(f"corpus: downloading {DATASET_REPO} for {len(missing)} missing file(s)")
    snapshot = Path(snapshot_download(DATASET_REPO, repo_type="dataset"))
    PROCESSED.mkdir(parents=True, exist_ok=True)
    # Only missing files are copied: a split rebuilt locally (build_corpus) is never
    # overwritten by the published copy.
    for name in missing:
        source = snapshot / name
        if not source.exists() and name in LEGACY_SPLIT_NAMES:
            source = snapshot / LEGACY_SPLIT_NAMES[name]  # dataset published before the rename
        if source.exists():
            (PROCESSED / name).write_bytes(source.read_bytes())
        else:
            _log(f"corpus: {name} is not in {DATASET_REPO}; skipped")
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


def _linear_source() -> str | None:
    """Which heads the linear backend would serve: "bundle", "web" or None (neither usable).
    Probed in a subprocess so a retrain is picked up fresh (predict caches per process)."""
    result = subprocess.run([sys.executable, "-c", _PROBE_SNIPPET], cwd=REPO_ROOT, capture_output=True, text=True)
    lines = result.stdout.split()
    return lines[-1] if result.returncode == 0 and lines and lines[-1] in ("bundle", "web") else None


def ensure_model() -> str:
    """Make the `linear` backend servable; returns the source: "bundle", "web" or "retrained".

    The committed browser weights (`site/models/weights.json`) are the shipped model. When the
    trained bundle is missing or stale for the current lexicons but those weights are usable,
    the server serves them (`QORGAN_LINEAR_WEIGHTS=auto|web`) and nothing is retrained: a
    retrain here would train a *different* model (other corpus, server embeddings) than the one
    every browser runs. Retraining remains the last resort when neither source is usable."""
    from qorgan.config import get_config

    policy = get_config().linear_weights_source
    if policy != "web" and not (MODEL_DIR / "metadata.json").exists():
        from huggingface_hub import snapshot_download

        _log(f"model: downloading {MODEL_REPO}")
        try:
            snapshot_download(MODEL_REPO, local_dir=MODEL_DIR)
        except Exception as exc:  # private repo / offline — the browser weights or a retrain cover it
            _log(f"model: download failed ({exc})")
    source = _linear_source()
    if source == "bundle":
        _log("model: linear backend OK (trained bundle)")
        return "bundle"
    if source == "web":
        _log("model: linear backend OK (the committed browser weights, site/models/weights.json -- "
             "models/linear is missing or stale for the current lexicons; not retraining)")
        return "web"
    if policy == "web":
        raise RuntimeError("QORGAN_LINEAR_WEIGHTS=web but site/models/weights.json is missing or unusable")
    _log("model: no usable weights (bundle stale or missing, no browser weights) — retraining from corpus")
    _run([sys.executable, "-m", "qorgan.classifier.linear_train"])
    if _linear_source() is None:
        raise RuntimeError("linear backend still failing after retrain")
    _log("model: retrained, linear backend OK")
    return "retrained"


def ensure_l2_seeds() -> None:
    # Through the config, not os.environ / a repo path: a deployment passes the key in an env
    # file the config loader reads, and may move the data dir (compose.yaml, docs/DEPLOY.md).
    from qorgan.config import get_config

    cfg = get_config()
    if (cfg.data_dir / "processed" / "organizations.jsonl").exists():
        _log("L2 seeds: present")
        return
    if cfg.number_hmac_key is None:
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


def ensure_embedder() -> None:
    """Self-host the int8 embedder at the dir the config names (server + browser load the
    same files). Runs before anything embeds: the model probe, a fallback retrain and the
    Level-2 seeding all go through this graph."""
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


def ensure_embed_runtime() -> None:
    """Self-host transformers.js + onnxruntime-web for the browser embedder (ADR D61). Required:
    the citizen page cannot score without it, and a CDN import is blocked by COEP in WebKit."""
    from qorgan.web.embed_runtime import ensure_embed_runtime as install

    target = install(REPO_ROOT / "site")
    _log(f"web runtime: {target.relative_to(REPO_ROOT)} (pinned, hash-verified)")


def _web_weights_usable(path: Path) -> bool:
    from qorgan.classifier.web_bundle import load_linear_from_web

    try:
        load_linear_from_web(path)
    except Exception:  # noqa: BLE001 -- any failure means "cannot be served as is"
        return False
    return True


def ensure_web_weights(source: str = "bundle") -> None:
    """The browser's head weights. The committed `site/models/weights.json` is the shipped
    model and is never overwritten while it is usable (developers publish a retrain explicitly
    with `scripts/export_parity_fixtures.py`). Only a missing or unusable file is replaced, and
    only by the export of a bundle that is valid now (`source` "bundle" or "retrained")."""
    target = SITE_MODELS / "weights.json"
    exported = MODEL_DIR / "web" / "weights.json"
    if target.exists() and _web_weights_usable(target):
        _log("web model: committed head weights kept")
        return
    if source in ("bundle", "retrained") and exported.exists():
        target.write_bytes(exported.read_bytes())
        _log("web model: head weights exported from the trained bundle")
    else:
        _log("web model: no usable head weights (retrain, then scripts/export_parity_fixtures.py)")


def ensure_asr_models() -> None:
    """Self-host the two small Vosk models for on-device recognition (B9)."""
    from qorgan.asr.web_models import ensure_vosk_model_tarball, ensure_vosklet_runtime
    from qorgan.config import get_config

    try:
        vendor = ensure_vosklet_runtime(REPO_ROOT / "site")
        _log(f"asr runtime: {vendor.relative_to(REPO_ROOT)} (pinned, hash-verified)")
    except Exception as exc:  # noqa: BLE001 - offline / tampered download: mic mode reports it
        _log(f"asr runtime: unavailable ({exc}); microphone mode will report it")
    cfg = get_config()
    # English is the optional third recogniser (ADR D62); "" in QORGAN_VOSK_MODEL_EN skips it.
    for name in (cfg.vosk_model_kk, cfg.vosk_model_ru, cfg.vosk_model_en):
        if not name:
            continue
        try:
            target = ensure_vosk_model_tarball(name, site_models_dir=SITE_MODELS)
            _log(f"asr model: {target.relative_to(REPO_ROOT)} ({target.stat().st_size / 1e6:.0f} MB)")
        except Exception as exc:  # offline / upstream down: the mic mode degrades, the demo does not
            _log(f"asr model: {name} unavailable ({exc}); microphone mode will report it")
    ensure_asr_endpoint(cfg)


def ensure_asr_endpoint(cfg) -> None:
    """Self-host the GigaAM endpoint pass (ADR D64): its sherpa-onnx browser runtime (a pinned
    local build or a mirror of it) and the 225 MB model, every file sha256-checked. Optional:
    without them the page falls back to the Vosk vote and says so."""
    from qorgan.asr.gigaam_web import ensure_gigaam_model, ensure_sherpa_runtime

    if not cfg.asr_endpoint_model:
        _log("asr endpoint: off (QORGAN_ASR_ENDPOINT_MODEL is empty)")
        return
    try:
        vendor = ensure_sherpa_runtime(
            REPO_ROOT / "site",
            build_dir=Path(cfg.sherpa_wasm_build_dir) if cfg.sherpa_wasm_build_dir else None,
            base_url=cfg.sherpa_wasm_url or None,
        )
        _log(f"asr endpoint runtime: {vendor.relative_to(REPO_ROOT)} (pinned, hash-verified)")
        target = ensure_gigaam_model(SITE_MODELS)
        _log(f"asr endpoint model: {target.relative_to(REPO_ROOT)} (pinned, hash-verified)")
    except Exception as exc:  # noqa: BLE001 - the page falls back to the Vosk vote
        _log(f"asr endpoint: unavailable ({exc}); the voice mode falls back to the Vosk vote")


# `serve`: everything the server needs before it listens. `seeds`: the Level-2 demo seeding --
# minutes on a shared vCPU, so the container entrypoint runs it in the background after `serve`
# and the health check is answered at once (ADR D60). `all` (default): both, in order.
PHASES = ("all", "serve", "seeds")


def _parse_phase(argv: list[str]) -> str:
    parser = argparse.ArgumentParser(description="Provision the deployed demo (idempotent).")
    parser.add_argument("--phase", choices=PHASES, default="all")
    return parser.parse_args(argv).phase


def main(argv: list[str] | None = None) -> None:
    phase = _parse_phase(argv or [])
    os.environ.setdefault("QORGAN_CLASSIFIER_BACKEND", "linear")
    if phase == "seeds":
        ensure_l2_seeds()
        return
    ensure_corpus()
    ensure_dialogue_pool()
    ensure_embedder()  # first: everything below embeds through it
    ensure_embed_runtime()
    source = ensure_model()
    if phase == "all":
        ensure_l2_seeds()
    ensure_web_weights(source)
    ensure_asr_models()
    _log("done — serve with: uvicorn qorgan.api:app --host 0.0.0.0 --port $PORT")


if __name__ == "__main__":
    main(sys.argv[1:])
