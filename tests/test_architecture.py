"""Architecture invariants (PLAN_2026-09 §2, C1) -- enforced, not just documented.

1. Level 2 has a single ingress: consented reports. The live pipeline and the stateless
   `/api/analyze` endpoint must not be able to reach the analytics write paths.
2. `/api/analyze` persists nothing.
3. No route accepts audio (no WebSocket, no audio/multipart request bodies) -- raw call
   audio never reaches a server; ASR belongs on the device.

These are AST/route-table checks, so they stay green regardless of which backend or model
is installed.
"""

from __future__ import annotations

import ast
import os
from pathlib import Path

import pytest
from fastapi.routing import APIWebSocketRoute
from fastapi.testclient import TestClient

from qorgan.api import app

_SRC = Path(__file__).resolve().parents[1] / "src" / "qorgan"
_DATA = Path(__file__).resolve().parents[1] / "data"

# Modules that sit on the citizen-facing path and must never touch the analytics store.
_GUARDED_MODULES = sorted(
    [*(_SRC / "live").glob("*.py"), *_SRC.glob("api_live*.py"), _SRC / "api.py", *_SRC.glob("api_reports*.py")]
)
_FORBIDDEN_MODULES = ("qorgan.analytics.store", "qorgan.analytics.pipeline", "qorgan.analytics.cluster")
_FORBIDDEN_INTAKE_NAMES = ("ingest_pending",)


def _imports(path: Path) -> list[tuple[str, str | None]]:
    """`(module, name)` pairs for every import statement in `path` (name is None for `import x`)."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found: list[tuple[str, str | None]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.extend((alias.name, None) for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            found.extend((node.module, alias.name) for alias in node.names)
    return found


def _all_routes(routes):
    for route in routes:
        inner = getattr(route, "original_router", None)  # FastAPI >= 0.13x lazily-included routers
        if inner is not None:
            yield from _all_routes(inner.routes)
        elif hasattr(route, "routes"):
            yield from _all_routes(route.routes)
        else:
            yield route


def _tree_snapshot(root: Path) -> dict[str, tuple[int, int]]:
    return {
        str(p.relative_to(root)): (p.stat().st_size, p.stat().st_mtime_ns)
        for p in root.rglob("*")
        if p.is_file()
    }


# --- 1. single ingress ------------------------------------------------------------------------


@pytest.mark.parametrize("module_path", _GUARDED_MODULES, ids=lambda p: p.name)
def test_citizen_path_modules_cannot_reach_analytics_write_paths(module_path: Path):
    offending = [
        (module, name)
        for module, name in _imports(module_path)
        if module.startswith(_FORBIDDEN_MODULES)
        or (module == "qorgan.analytics.intake" and name in _FORBIDDEN_INTAKE_NAMES)
        or (module == "qorgan.analytics" and name in ("store", "pipeline", "cluster"))
    ]
    assert not offending, f"{module_path.name} imports analytics write paths: {offending}"


# --- 2. /api/analyze is stateless -------------------------------------------------------------


def test_analyze_endpoint_writes_nothing_under_data():
    before = _tree_snapshot(_DATA)
    client = TestClient(app)
    res = client.post(
        "/api/analyze",
        json={"transcript": "Алло, это служба безопасности банка. Продиктуйте код из SMS.", "backend": "mock"},
    )
    assert res.status_code == 200
    assert _tree_snapshot(_DATA) == before


# --- 3. no audio reaches the server -----------------------------------------------------------


def test_no_websocket_routes():
    sockets = [r.path for r in _all_routes(app.routes) if isinstance(r, APIWebSocketRoute)]
    assert sockets == [], f"audio-capable WebSocket routes present: {sockets}"


def test_no_route_accepts_audio_or_multipart_bodies():
    spec = app.openapi()
    offending: list[str] = []
    for path, operations in spec["paths"].items():
        for method, op in operations.items():
            content = (op.get("requestBody") or {}).get("content", {})
            if any(ctype.startswith("audio/") or ctype.startswith("multipart/") for ctype in content):
                offending.append(f"{method.upper()} {path}")
    assert offending == [], offending


def test_no_server_side_streaming_asr_module_is_wired():
    """The dual-Vosk server recognizer may exist for batch/offline use, but nothing in the
    served API package may import it (`asr.vosk_stream` was the WebSocket's engine)."""
    served = [_SRC / "api.py", *_SRC.glob("api_*.py")]
    offending = [
        (p.name, module)
        for p in served
        for module, _ in _imports(p)
        if module.startswith("qorgan.asr.vosk_stream") or module.startswith("qorgan.asr.capture")
    ]
    assert offending == [], offending


def test_site_ships_no_server_streaming_microphone_client():
    site = Path(__file__).resolve().parents[1] / "site"
    assert not (site / "live_mic.js").exists(), "site/live_mic.js streams raw PCM to the server"
    for js in site.glob("*.js"):
        assert "WebSocket" not in js.read_text(encoding="utf-8"), f"{js.name} opens a WebSocket"


# sanity: the guard list is not silently empty
def test_guarded_module_list_is_populated():
    names = {p.name for p in _GUARDED_MODULES}
    assert {"api.py", "api_live.py", "session.py", "summary.py"} <= names
    assert os.environ.get("QORGAN_CLASSIFIER_BACKEND", "linear") in {"linear", "mock", "llm", "xlmr"}
