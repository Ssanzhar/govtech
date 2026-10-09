# Security evidence: SBOMs, dependency audits, static analysis (ADR D66, 2026-10-09)

Generated on branch `sanzh-ts` for the ГТС source-code test (`../SECURITY_CHECKLIST.md` §1).
No file here contains secrets or local paths. Regenerate the SBOMs with
`python scripts/generate_sbom.py` (needs `pip install "cyclonedx-bom>=6"` in any venv).

| File | What | Result |
|---|---|---|
| `sbom-python.cdx.json` | CycloneDX 1.5, the server image: exact pins of `scripts/deploy_constraints.txt` | 45 components |
| `sbom-npm.cdx.json` | CycloneDX, the Node toolchain (`package-lock.json`): parity tests + device-embedding bridge. Not in the image, not served to browsers | 82 components |
| `sbom-browser.cdx.json` | Third-party code the citizen's browser runs: self-hosted transformers.js 3.8.1 / ONNX Runtime Web and Vosklet 1.2.1, with the SHA-256 `deploy_bootstrap` verifies | 5 files |
| `pip-audit-runtime.json` | pip-audit 2.10.1 over the image pins (`--no-deps --disable-pip`) | **0 known vulnerabilities** |
| `pip-audit-dev-env.json` | pip-audit over the local development venv (dev, research, harness extras) | 61 advisories in 9 packages, none in the image (below) |
| `npm-audit.json` | `npm audit` | 2 high, 4 moderate, all dev-only (below) |
| `bandit-src.{json,txt}` | bandit 1.9.4 over `src/` | 0 high (2 fixed), 10 medium (below), 2 low |
| `semgrep-site.json` | semgrep over `site/` (own JS/HTML, `vendor/` and models excluded), if present | see the ADR D66 entry |

## Triage

**Fixed (bandit HIGH).**
- B324 `analytics/intake.py`: SHA-1 builds a deterministic incident id, not a security function.
  Now `hashlib.sha1(..., usedforsecurity=False)`: the same digest, so no id changes.
- B202 `asr/web_models.py`: `ZipFile.extractall` runs after `_safe_extract` has checked every
  member against the destination (path traversal refused). False positive, annotated `nosec B202`.

**Documented, not changed.**
- npm (2 high, 4 moderate): `sharp` (libvips / libheif / librsvg CVEs), `sprintf-js`, `roarr`,
  `global-agent`, `onnxruntime-node`, all pulled in by `@huggingface/transformers@3.8.1`, a
  devDependency. The fix is transformers 4.x, which `CLAUDE.md` forbids casually: the device
  embedding must stay byte-compatible with the server's ONNX runtime (ADR D32/D33). None of
  this runs in production: the image has no Node, and the browser loads the separately vendored
  `transformers.min.js` (no `sharp`, no `onnxruntime-node`). `sharp` processes images, which
  Qorğan never does.
- pip dev env: `aiohttp`, `anyio` 4.14.1, `cryptography`, `datasets`, `fsspec` 2026.4.0,
  `gitpython`, `multidict`, `pip`, `urllib3`. None is in the image closure. Where the image has
  the same package (`anyio` 4.15.1, `fsspec` 2026.9.0), its pin is already a fixed version.
  Action for developers: upgrade the local venv (`pip install -U ...`); nothing ships.
- bandit MEDIUM: B310 `urlopen` (5x) on constant `https://` URLs of pinned, hash-checked
  downloads at bootstrap / dev time; B615 (4x) and B614 (1x) in the abandoned `xlmr` research
  path (`classifier/model.py`, `predict.py`, `train.py`), never imported by the server.
- The parity-pinned runtimes (`onnxruntime`, `onnxruntime-node`, transformers.js) were not
  bumped (`CLAUDE.md`); pip-audit reports no advisory for the pinned `onnxruntime`.

**Not done (organisational / CI).** Running these in CI on every change, Trivy on the built image,
pinning the base image by digest, hash-checked requirements (`docs/DEPLOY.md` §7(8)).
