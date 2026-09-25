"""Package the small Vosk models — and pin the Vosklet runtime — for the browser (PLAN_2026-09 B9).

Vosklet loads a model from a `.tar.gz` **in USTAR format** with the model directory at
the archive root; macOS `tar` defaults (PAX headers, AppleDouble `._*` entries) make it
refuse the archive. This module builds a clean archive from the model directory the
`vosk` package downloads to `~/.cache/vosk`, fetching the official zip from alphacephei
when the cache is empty. Pure over the paths it is given; the downloader is injectable.
"""

from __future__ import annotations

import hashlib
import io
import tarfile
import zipfile
from collections.abc import Callable
from pathlib import Path
from urllib.request import urlopen

VOSK_MODELS_BASE_URL = "https://alphacephei.com/vosk/models"
# The recogniser runtime is self-hosted (it sees the raw microphone audio, so it must not be
# a live third-party script) and pinned: a release tag plus the sha256 of each file, checked
# on download. Bump all three together when upgrading.
VOSKLET_VERSION = "1.2.1"
VOSKLET_BASE_URL = f"https://cdn.jsdelivr.net/gh/msqr1/Vosklet@{VOSKLET_VERSION}/Examples"
VOSKLET_FILES = {
    "Vosklet.js": "c60c88e97923573860eea3958f9dbdc4e309567d6f1bc911828ecb4bb4f35cbd",
    "Vosklet.wasm": "35b6c484c7cf35e09fd76c3460a5c5816706f97fc2503364eb241ad97ecbe73b",
}
VOSKLET_VENDOR_SUBDIR = Path("vendor") / "vosklet"
DEFAULT_VOSK_CACHE = Path.home() / ".cache" / "vosk"
_SKIP_NAMES = {".DS_Store"}
_SKIP_PREFIXES = ("._",)

Downloader = Callable[[str], bytes]


def package_vosk_model(model_dir: Path, target: Path) -> Path:
    """Write `target` (USTAR tar.gz) with `model_dir.name/...` at the root, skipping
    macOS metadata entries; deterministic file order."""
    if not (model_dir / "conf").is_dir() or not (model_dir / "am").is_dir():
        raise ValueError(f"{model_dir} does not look like a Vosk model (no conf/ and am/)")
    target.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(target, "w:gz", format=tarfile.USTAR_FORMAT) as archive:
        # The root directory entry must come first: Vosklet's untar creates directories only
        # from their own entries and fails to open a file whose parent it has not seen.
        archive.add(model_dir, arcname=model_dir.name, recursive=False)
        for path in sorted(p for p in model_dir.rglob("*") if _keep(p)):
            archive.add(path, arcname=f"{model_dir.name}/{path.relative_to(model_dir)}", recursive=False)
    return target


def ensure_vosk_model_tarball(
    name: str,
    *,
    site_models_dir: Path,
    cache_dir: Path = DEFAULT_VOSK_CACHE,
    downloader: Downloader | None = None,
) -> Path:
    """`site_models_dir/vosk/<name>.tar.gz`, built from the cached model directory or a
    fresh download of the official zip (extracted into `cache_dir`)."""
    target = site_models_dir / "vosk" / f"{name}.tar.gz"
    if target.exists():
        return target
    model_dir = cache_dir / name
    if not model_dir.is_dir():
        payload = (downloader or _download)(f"{VOSK_MODELS_BASE_URL}/{name}.zip")
        cache_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(io.BytesIO(payload)) as zipped:
            _safe_extract(zipped, cache_dir)
        if not model_dir.is_dir():
            raise RuntimeError(f"the {name} archive did not contain a {name}/ directory")
    return package_vosk_model(model_dir, target)


def ensure_vosklet_runtime(site_dir: Path, *, downloader: Downloader | None = None) -> Path:
    """`site/vendor/vosklet/{Vosklet.js,Vosklet.wasm}` from the pinned release, each verified
    against its sha256 before being written; an existing file with the right hash is kept."""
    target_dir = site_dir / VOSKLET_VENDOR_SUBDIR
    for name, expected in VOSKLET_FILES.items():
        target = target_dir / name
        if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest() == expected:
            continue
        payload = (downloader or _download)(f"{VOSKLET_BASE_URL}/{name}")
        actual = hashlib.sha256(payload).hexdigest()
        if actual != expected:
            raise RuntimeError(f"{name} from {VOSKLET_BASE_URL} has sha256 {actual}, expected {expected}; refusing to install")
        target_dir.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
    return target_dir


def _keep(path: Path) -> bool:
    return path.name not in _SKIP_NAMES and not path.name.startswith(_SKIP_PREFIXES) and not path.is_symlink()


def _safe_extract(zipped: zipfile.ZipFile, destination: Path) -> None:
    root = destination.resolve()
    for member in zipped.infolist():
        resolved = (root / member.filename).resolve()
        if root not in resolved.parents and resolved != root:
            raise RuntimeError(f"refusing to extract {member.filename!r} outside {root}")
    zipped.extractall(root)


def _download(url: str) -> bytes:  # pragma: no cover - network
    with urlopen(url, timeout=120) as response:
        return response.read()
