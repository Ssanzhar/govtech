"""Self-host the GigaAM endpoint pass for the browser (ADR D63): sherpa-onnx WASM + the model.

Two gitignored assets, each pinned by sha256 and checked before anything is written:

- the **runtime**, `site/vendor/sherpa-onnx/<version>/` -- Qorğan's own single-threaded browser
  build of sherpa-onnx (offline ASR only; `scripts/sherpa_wasm/build.sh`). Upstream publishes no
  single-threaded browser build (its `vad-asr` web builds use pthreads + SharedArrayBuffer), so
  the files come from a local build (`build_dir`, `QORGAN_SHERPA_WASM_BUILD_DIR`) or a mirror
  of that build (`base_url`, `QORGAN_SHERPA_WASM_URL`); either way the bytes must hash to
  `SHERPA_WASM_FILES`. It sees the raw microphone audio, so it is never a live third-party
  script (the Vosklet rule, ADR D26).
- the **model**, `site/models/gigaam/<id>/{model.int8.onnx,tokens.txt}` -- GigaAM-Multilingual
  220M CTC int8 in sherpa-onnx's NeMo-CTC format (MIT), from the local spike cache
  (`models/asr/gigaam-ml/`) or the Hugging Face export at a pinned revision.

Pure over the paths it is given; the downloader is injectable.
"""

from __future__ import annotations

import hashlib
import shutil
from collections.abc import Callable
from pathlib import Path
from urllib.request import urlopen

SHERPA_WASM_VERSION = "1.13.8"
SHERPA_VENDOR_SUBDIR = Path("vendor") / "sherpa-onnx" / SHERPA_WASM_VERSION
# sha256 of scripts/sherpa_wasm/build.sh's output (emsdk 4.0.23, sherpa-onnx v1.13.8 source
# b0374cc5..., onnxruntime-wasm-static_lib-simd 1.28.2 pinned by sherpa's CMake).
SHERPA_WASM_FILES = {
    "sherpa-onnx-wasm-web.js": "6b5ec5db8dab0eb45c7c7a961253ba444d83120c93ebab90864aec8091c35eab",
    "sherpa-onnx-wasm-web.wasm": "d38edc5aaa21e026e3bb37d6ff315882919741e28bd754ffdc8210a538678a29",
    "sherpa-onnx-asr.js": "d51ae8e8b756ee5e53423ffada0c9702973f154f561aca7984fe0b12f4060178",
}

GIGAAM_MODEL_ID = "gigaam-multilingual-220m-ctc-int8"
GIGAAM_REPO = "fussraider/GigaAM-Multilingual-sherpa-onnx-ctc"
GIGAAM_REVISION = "9f5a77e8975211abe8511693accd3a63ee1e9f43"
GIGAAM_FILES = {
    "model.int8.onnx": "2d94f93ffd4ef58e7899c9de885c25bbbc8c9f1073618868d118a674450ba5f7",
    "tokens.txt": "9b5df7987cb4ca52c1a468649ce897fab1cd182067416e29fef49dfaa7a856c2",
}
GIGAAM_SUBDIR = Path("gigaam") / GIGAAM_MODEL_ID
GIGAAM_MB = 225  # the download the page discloses before the first use
DEFAULT_GIGAAM_CACHE = Path(__file__).resolve().parents[3] / "models" / "asr" / "gigaam-ml"

Downloader = Callable[[str], bytes]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _all_present(directory: Path, pins: dict[str, str]) -> bool:
    return all((directory / name).is_file() and _sha256_file(directory / name) == sha for name, sha in pins.items())


def _install_verified(target_dir: Path, name: str, payload: bytes, expected: str, origin: str) -> None:
    actual = hashlib.sha256(payload).hexdigest()
    if actual != expected:
        raise RuntimeError(f"{name} from {origin} has sha256 {actual}, expected {expected}; refusing to install")
    target_dir.mkdir(parents=True, exist_ok=True)
    (target_dir / name).write_bytes(payload)


def ensure_sherpa_runtime(
    site_dir: Path,
    *,
    build_dir: Path | None = None,
    base_url: str | None = None,
    downloader: Downloader | None = None,
) -> Path:
    """`site/vendor/sherpa-onnx/<v>/` with every file of `SHERPA_WASM_FILES`, verified.

    Kept when already installed with the right hashes; else copied from `build_dir` or
    downloaded from `base_url`. Nothing is written unless every file verifies."""
    target = site_dir / SHERPA_VENDOR_SUBDIR
    if _all_present(target, SHERPA_WASM_FILES):
        return target
    if build_dir is not None:
        payloads = {name: (build_dir / name).read_bytes() for name in SHERPA_WASM_FILES if (build_dir / name).is_file()}
        origin = str(build_dir)
    elif base_url:
        fetch = downloader or _download
        payloads = {name: fetch(base_url.rstrip("/") + "/" + name) for name in SHERPA_WASM_FILES}
        origin = base_url
    else:
        raise RuntimeError(
            "no sherpa-onnx browser runtime: build it with scripts/sherpa_wasm/build.sh and set "
            "QORGAN_SHERPA_WASM_BUILD_DIR (or QORGAN_SHERPA_WASM_URL to a mirror of that build)"
        )
    missing = set(SHERPA_WASM_FILES) - set(payloads)
    if missing:
        raise RuntimeError(f"{origin} lacks {sorted(missing)}; refusing to install a partial runtime")
    for name, payload in payloads.items():  # verify everything before writing anything
        actual = hashlib.sha256(payload).hexdigest()
        if actual != SHERPA_WASM_FILES[name]:
            raise RuntimeError(f"{name} from {origin} has sha256 {actual}, expected {SHERPA_WASM_FILES[name]}; refusing to install")
    for name, payload in payloads.items():
        _install_verified(target, name, payload, SHERPA_WASM_FILES[name], origin)
    return target


def ensure_gigaam_model(
    site_models_dir: Path,
    *,
    cache_dir: Path = DEFAULT_GIGAAM_CACHE,
    downloader: Downloader | None = None,
) -> Path:
    """`site/models/gigaam/<id>/` with the pinned model + tokens, from `cache_dir` when its
    copies verify, else from the Hugging Face export at `GIGAAM_REVISION`."""
    target = site_models_dir / GIGAAM_SUBDIR
    for name, expected in GIGAAM_FILES.items():
        dest = target / name
        if dest.is_file() and _sha256_file(dest) == expected:
            continue
        cached = cache_dir / name
        if cached.is_file() and _sha256_file(cached) == expected:
            target.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(cached, dest)
            continue
        url = f"https://huggingface.co/{GIGAAM_REPO}/resolve/{GIGAAM_REVISION}/{name}"
        _install_verified(target, name, (downloader or _download)(url), expected, url)
    return target


def _download(url: str) -> bytes:  # pragma: no cover - network
    with urlopen(url, timeout=600) as response:
        return response.read()
