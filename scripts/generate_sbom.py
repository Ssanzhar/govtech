"""Regenerate the CycloneDX SBOMs under `docs/egov/security/` (ADR D66; ГТС source-code analysis).

Three documents, because three things ship:
- `sbom-python.cdx.json` -- the server image: the exact pins in `scripts/deploy_constraints.txt`
  (what the Dockerfile installs), via `cyclonedx-py requirements`;
- `sbom-npm.cdx.json` -- the Node toolchain for the parity tests and the device-embedding bridge
  (`package-lock.json`, via `npm sbom`); it is NOT shipped to browsers or into the image;
- `sbom-browser.cdx.json` -- what the citizen's browser runs that we did not write: the
  self-hosted transformers.js / ONNX Runtime Web and Vosklet files, with the sha256 that
  `deploy_bootstrap` verifies (`qorgan.web.embed_runtime`, `qorgan.asr.web_models`).

Run (needs the CycloneDX tool, kept out of the project's dependencies on purpose):
    pip install "cyclonedx-bom>=6"        # in any venv; or pass --cyclonedx-py PATH
    python scripts/generate_sbom.py
"""

from __future__ import annotations

import argparse
import json
import subprocess
import uuid
from datetime import UTC, datetime
from pathlib import Path

from qorgan.asr.web_models import VOSKLET_BASE_URL, VOSKLET_FILES, VOSKLET_VERSION
from qorgan.web.embed_runtime import TRANSFORMERS_BASE_URL, TRANSFORMERS_FILES, TRANSFORMERS_VERSION

_REPO = Path(__file__).resolve().parents[1]
_SPEC_VERSION = "1.5"


def python_sbom(cyclonedx_py: str, out: Path) -> None:
    subprocess.run(
        [cyclonedx_py, "requirements", str(_REPO / "scripts" / "deploy_constraints.txt"),
         "--of", "JSON", "--sv", _SPEC_VERSION, "-o", str(out)],
        check=True,
    )


def npm_sbom(out: Path) -> None:
    result = subprocess.run(
        ["npm", "sbom", "--sbom-format", "cyclonedx", "--package-lock-only"],
        cwd=_REPO, check=True, capture_output=True, text=True,
    )
    out.write_text(result.stdout, encoding="utf-8")


def _component(name: str, version: str, base_url: str, filename: str, sha256: str) -> dict:
    return {
        "type": "library",
        "name": f"{name}/{filename}",
        "version": version,
        "hashes": [{"alg": "SHA-256", "content": sha256}],
        "externalReferences": [{"type": "distribution", "url": f"{base_url}/{filename}"}],
    }


def browser_sbom(out: Path) -> None:
    components = [
        _component("@huggingface/transformers", TRANSFORMERS_VERSION, TRANSFORMERS_BASE_URL, name, digest)
        for name, digest in sorted(TRANSFORMERS_FILES.items())
    ] + [
        _component("vosklet", VOSKLET_VERSION, VOSKLET_BASE_URL, name, digest)
        for name, digest in sorted(VOSKLET_FILES.items())
    ]
    document = {
        "bomFormat": "CycloneDX",
        "specVersion": _SPEC_VERSION,
        "serialNumber": f"urn:uuid:{uuid.uuid4()}",
        "version": 1,
        "metadata": {
            "timestamp": datetime.now(UTC).replace(microsecond=0).isoformat(),
            "component": {"type": "application", "name": "qorgan-site (self-hosted third-party runtime)"},
        },
        "components": components,
    }
    out.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out-dir", type=Path, default=_REPO / "docs" / "egov" / "security")
    parser.add_argument("--cyclonedx-py", default="cyclonedx-py", help="path to the cyclonedx-py executable")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    python_sbom(args.cyclonedx_py, args.out_dir / "sbom-python.cdx.json")
    npm_sbom(args.out_dir / "sbom-npm.cdx.json")
    browser_sbom(args.out_dir / "sbom-browser.cdx.json")
    print(f"wrote 3 CycloneDX SBOMs to {args.out_dir}")


if __name__ == "__main__":
    main()
