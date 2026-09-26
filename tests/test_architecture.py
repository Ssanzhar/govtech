"""Architecture invariants (PLAN_2026-09 §2, C1) -- enforced, not just documented.

1. Level 2 has a single ingress: consented reports. The live pipeline and the stateless
   `/api/analyze` endpoint must not be able to reach the analytics write paths.
2. `/api/analyze` persists nothing.
3. No route accepts audio (no WebSocket, no audio/multipart request bodies) -- raw call
   audio never reaches a server; ASR belongs on the device.
4. The partner ingress (`/api/v1`, PLAN C5) is a consented ingress, not a bulk feed: it
   is authenticated, requires a `consent_basis`, refuses unscrubbed transcripts and is
   quota-bound -- and, like the citizen ingress, cannot reach the analytics write paths.
5. The analyst console (`/api/admin`) is not a browsing tool: every route requires an
   authenticated analyst (identity from the credential only), it is closed when no analyst
   credentials or no audit-chain key are configured, and any route whose response carries a
   full transcript requires the investigator role.

These are AST/route-table checks, so they stay green regardless of which backend or model
is installed.
"""

from __future__ import annotations

import ast
import os
from pathlib import Path

import pytest
from fastapi.routing import APIRoute, APIWebSocketRoute
from fastapi.testclient import TestClient

from qorgan.api import app

_SRC = Path(__file__).resolve().parents[1] / "src" / "qorgan"
_DATA = Path(__file__).resolve().parents[1] / "data"

# Modules that sit on the citizen-facing path and must never touch the analytics store.
_GUARDED_MODULES = sorted(
    [
        *(_SRC / "live").glob("*.py"),
        *_SRC.glob("api_live*.py"),
        _SRC / "api.py",
        *_SRC.glob("api_reports*.py"),
        _SRC / "api_partner.py",  # the partner ingress; the export module is read-only by design
    ]
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


# Every route that accepts a request body or changes state, by name. Adding one is an
# architecture change: it must be argued here, not slipped in (2026-09-26: the retired
# `/api/live/session/*` held call content in server memory and its `/report` was a second,
# consent-free citizen ingress; nothing noticed because no test listed the write routes).
_WRITE_ROUTES = {
    ("POST", "/api/analyze"),                                    # stateless (invariant 2)
    ("POST", "/api/reports"),                                    # THE citizen ingress
    ("DELETE", "/api/reports/{receipt_id}"),
    ("POST", "/api/v1/reports"),                                 # the consented partner ingress
    ("DELETE", "/api/v1/reports/{receipt_id}"),
    ("POST", "/api/admin/organizations/{org_id}/feedback"),      # analyst, audited
    ("POST", "/api/admin/incidents/{incident_id}/open"),         # investigator, audited
    ("POST", "/api/admin/ingest"),                               # analyst, audited
}
_REPORT_CREATING_ROUTES = {("POST", "/api/reports"), ("POST", "/api/v1/reports")}


def _write_routes() -> set[tuple[str, str]]:
    return {
        (method, route.path)
        for route in _all_routes(app.routes)
        if isinstance(route, APIRoute)
        for method in route.methods
        if method not in ("GET", "HEAD", "OPTIONS")
    }


def test_every_write_route_is_named_in_the_architecture():
    assert _write_routes() == _WRITE_ROUTES


def test_only_the_two_consented_ingresses_create_reports():
    creators = {
        (method, path) for method, path in _write_routes()
        if method == "POST" and path.rstrip("/").endswith("/reports")
    }
    assert creators == _REPORT_CREATING_ROUTES
    assert not any(path.startswith("/api/live") for _, path in _write_routes()), "live call content stays on the device"


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


_NETWORK_CALLS = ("fetch(", "XMLHttpRequest", "WebSocket", "sendBeacon", "EventSource", "RTCPeerConnection")


def test_on_device_asr_module_makes_no_network_calls():
    """The only module that holds microphone audio (site/core/asr.js, PLAN B9) must not be
    able to send anything: no fetch, XHR, sockets or beacons. Model files are fetched by the
    recogniser runtime from this origin; the audio itself never leaves the AudioWorklet."""
    source = (Path(__file__).resolve().parents[1] / "site" / "core" / "asr.js").read_text(encoding="utf-8")
    offending = [call for call in _NETWORK_CALLS if call in source]
    assert not offending, f"site/core/asr.js contains network primitives: {offending}"
    assert "getUserMedia(" not in source  # capture is the page's explicit action, not the module's (a support check may name it)


# sanity: the guard list is not silently empty
def test_guarded_module_list_is_populated():
    names = {p.name for p in _GUARDED_MODULES}
    assert {"api.py", "api_live.py", "session.py", "summary.py", "api_reports.py", "api_partner.py"} <= names
    assert os.environ.get("QORGAN_CLASSIFIER_BACKEND", "linear") in {"linear", "mock", "llm", "xlmr"}


# --- 4. the partner ingress is consented, not a bulk feed -------------------------------------

_PARTNER_SECRET = "arch-test-secret-0123456789abcdef"


def test_partner_ingress_is_closed_by_default_and_never_a_bulk_feed(tmp_path, monkeypatch):
    monkeypatch.setenv("QORGAN_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("QORGAN_TAXONOMY_PATH", str(_DATA / "taxonomy" / "tactics.yaml"))
    monkeypatch.setenv("QORGAN_PARTNER_API_KEYS", "")
    client = TestClient(app)
    report = {"consent_basis": "customer_consent", "tactic_ids": ["otp_request"]}
    assert client.post("/api/v1/reports", json=report).status_code == 401  # closed without a registry

    monkeypatch.setenv("QORGAN_PARTNER_API_KEYS", f"arch_partner:{_PARTNER_SECRET}:1")
    headers = {"X-API-Key": _PARTNER_SECRET}
    assert client.post("/api/v1/reports", json=[report, report], headers=headers).status_code == 422  # one per request
    assert client.post("/api/v1/reports", json={"tactic_ids": ["otp_request"]}, headers=headers).status_code == 422  # consent_basis
    unscrubbed = {**report, "tactic_ids": [], "transcript": "перезвоните на +7 700 101 20 30"}
    assert client.post("/api/v1/reports", json=unscrubbed, headers=headers).status_code == 422  # pre-scrubbed only
    assert client.post("/api/v1/reports", json={**report, "partner_reference": "a"}, headers=headers).status_code == 201
    assert client.post("/api/v1/reports", json={**report, "partner_reference": "b"}, headers=headers).status_code == 429  # quota
    assert "101 20 30" not in (tmp_path / "processed" / "citizen_reports.jsonl").read_text(encoding="utf-8")


# --- 5. the analyst console is authenticated; a whole call needs the investigator role -------

from qorgan.api_admin_auth import require_analyst, require_investigator  # noqa: E402


def _admin_routes() -> list[APIRoute]:
    return [r for r in _all_routes(app.routes) if isinstance(r, APIRoute) and r.path.startswith("/api/admin")]


def _dependency_calls(dependant) -> set:
    calls = set()
    for sub in dependant.dependencies:
        calls.add(sub.call)
        calls |= _dependency_calls(sub)
    return calls


def _carries_a_transcript(route: APIRoute) -> bool:
    fields = getattr(route.response_model, "model_fields", {}) or {}
    return "transcript" in fields


def test_every_admin_route_requires_an_authenticated_analyst():
    routes = _admin_routes()
    assert len(routes) >= 8, [r.path for r in routes]  # guard against the check silently seeing nothing
    unguarded = [f"{sorted(r.methods)} {r.path}" for r in routes if require_analyst not in _dependency_calls(r.dependant)]
    assert unguarded == [], f"admin routes without the analyst-auth dependency: {unguarded}"


def test_a_full_transcript_is_only_served_to_the_investigator_role():
    revealing = [r for r in _admin_routes() if _carries_a_transcript(r)]
    assert [r.path for r in revealing] == ["/api/admin/incidents/{incident_id}/open"]
    for route in revealing:
        assert require_investigator in _dependency_calls(route.dependant), route.path


def test_admin_routes_fail_closed_at_runtime(tmp_path, monkeypatch):
    """Behavioural twin of the structural check: every admin route, with its path parameters
    filled in, answers 401 without a key and 503 when the console is not configured."""
    monkeypatch.setenv("QORGAN_DATA_DIR", str(tmp_path))
    client = TestClient(app)
    for route in _admin_routes():
        path = route.path.replace("{org_id}", "org_0").replace("{incident_id}", "i1")
        for method in route.methods:
            assert client.request(method, path).status_code == 401, f"{method} {path}"
            assert client.request(method, path, headers={"X-Analyst-Id": "someone"}).status_code == 401
    monkeypatch.setenv("QORGAN_ANALYST_KEYS", "")
    for route in _admin_routes():
        path = route.path.replace("{org_id}", "org_0").replace("{incident_id}", "i1")
        for method in route.methods:
            assert client.request(method, path).status_code == 503, f"{method} {path}"
