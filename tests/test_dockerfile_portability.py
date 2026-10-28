"""The production image must build and start on managed builders, not only a local BuildKit.

Railway (the hosted demo) refuses a BuildKit cache mount whose id lacks its service prefix
("Cache mount ID is not prefixed with cache key"), older Dockerfile frontends reject
`COPY --exclude` ("unknown flag: exclude") and `HEALTHCHECK --start-interval`, and a platform
volume is mounted root-owned, which the image's non-root service account cannot write.
docs/DEPLOY.md §8.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DOCKERFILE = REPO_ROOT / "Dockerfile"
ENTRYPOINT = REPO_ROOT / "scripts" / "deploy_entrypoint.sh"


def _instructions() -> list[str]:
    """Dockerfile instructions with comments dropped and continuation lines joined."""
    lines = [line for line in DOCKERFILE.read_text(encoding="utf-8").splitlines() if not line.lstrip().startswith("#")]
    return [part.strip() for part in re.sub(r"\\\n", " ", "\n".join(lines)).splitlines() if part.strip()]


def test_no_buildkit_cache_mounts() -> None:
    offending = [line for line in _instructions() if "--mount=type=cache" in line]
    assert offending == [], offending


def test_no_copy_exclude() -> None:
    offending = [line for line in _instructions() if line.startswith(("COPY", "ADD")) and "--exclude" in line]
    assert offending == [], offending


def test_no_healthcheck_start_interval() -> None:
    offending = [line for line in _instructions() if line.startswith("HEALTHCHECK") and "--start-interval" in line]
    assert offending == [], offending


def test_image_runs_as_the_service_account() -> None:
    users = [line for line in _instructions() if line.startswith("USER")]
    assert users == ["USER 10001:10001"]


def test_entrypoint_drops_root_after_taking_the_state_dir() -> None:
    script = ENTRYPOINT.read_text(encoding="utf-8")
    root_branch = script.index('if [ "$(id -u)" = 0 ]')
    assert script.index("chown -R", root_branch) < script.index("exec setpriv --reuid=", root_branch)
    # The server is started only after the privilege drop.
    assert script.index("exec setpriv", root_branch) < script.index("exec uvicorn")


def test_bootstrap_build_step_gets_the_pinned_gigaam_runtime_mirror() -> None:
    """The GigaAM runtime is fetched at build time, so the mirror URL must be a build ARG; a
    Railway service variable alone never reached the RUN step and the page fell back (ADR D79)."""
    lines = _instructions()
    bootstrap = next(i for i, line in enumerate(lines) if line.startswith("RUN python scripts/deploy_bootstrap.py"))
    last_from = max(i for i, line in enumerate(lines[:bootstrap]) if line.startswith("FROM"))
    args = [line for line in lines[last_from:bootstrap] if line.startswith("ARG QORGAN_SHERPA_WASM_URL=")]
    assert len(args) == 1, "declare ARG QORGAN_SHERPA_WASM_URL in the runtime stage, before the bootstrap"
    url = args[0].split("=", 1)[1]
    # Pinned to a commit (40 hex), never a moving branch; the files are also sha256-checked.
    assert re.fullmatch(r"https://huggingface\.co/sanzh-ts/govtech/resolve/[0-9a-f]{40}/sherpa-onnx/1\.13\.8", url), url
