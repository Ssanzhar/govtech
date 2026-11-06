"""Tests for `qorgan.api_limits`: 422 bodies never echo what the client sent; oversized or
length-less bodies are refused before they are read."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from qorgan.api import app
from qorgan.api_limits import MAX_BODY_BYTES
from qorgan.api_reports import _LIMITER

NUMBER = "+7 700 101 20 30"


@pytest.fixture()
def client(tmp_path, monkeypatch) -> TestClient:
    monkeypatch.setenv("QORGAN_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("QORGAN_TAXONOMY_PATH", "data/taxonomy/tactics.yaml")
    _LIMITER.reset()
    return TestClient(app)


def test_validation_errors_do_not_echo_the_offending_value(client):
    too_long = f"перезвоните на {NUMBER} " * 2000  # > 20 000 chars, contains a number
    res = client.post("/api/reports", json={"transcript": too_long, "risk_score": 10, "consent": True})

    assert res.status_code == 422
    assert NUMBER not in res.text and "input" not in res.json()["detail"][0]
    assert {"loc", "msg", "type"} <= set(res.json()["detail"][0])


def test_oversized_bodies_are_refused_before_validation(client):
    res = client.post("/api/analyze", content=b"x" * (MAX_BODY_BYTES + 1), headers={"Content-Type": "application/json"})
    assert res.status_code == 413


def test_bodies_without_a_declared_length_are_refused(client):
    res = client.post("/api/analyze", content=iter([b'{"transcript": "a"}']), headers={"Content-Type": "application/json"})
    assert res.status_code == 411


def test_normal_requests_pass_through(client):
    assert client.get("/api/health").status_code == 200
    res = client.post("/api/analyze", json={"transcript": "Алло, это банк.", "backend": "mock"})
    assert res.status_code == 200


def test_only_the_live_page_and_its_worker_scripts_are_cross_origin_isolated(client):
    for path in ("/live.html", "/core/embed-worker.js", "/core/asr.js"):
        res = client.get(path)
        assert res.status_code == 200, path
        assert res.headers["cross-origin-opener-policy"] == "same-origin"
        assert res.headers["cross-origin-embedder-policy"] == "require-corp"  # workers inherit the document's COEP
        assert res.headers["cache-control"] == "no-cache"  # stale cached headers would block the worker
    for path in ("/", "/admin.html", "/api/health"):
        res = client.get(path)
        assert res.status_code == 200 and "cross-origin-embedder-policy" not in res.headers, path


# --- security headers on every response (QA 2026-09-29) ----------------------------------------


@pytest.mark.parametrize("path", ["/", "/index.html", "/live.html", "/admin.html", "/api/health", "/core/qorgan-config.json"])
def test_every_response_refuses_framing_sniffing_and_referrers(path):
    res = TestClient(app).get(path)
    assert res.status_code == 200, path
    assert res.headers["x-content-type-options"] == "nosniff"
    assert res.headers["referrer-policy"] == "no-referrer"
    # The analyst console must not be framed (clickjacking); frame-ancestors only works as a header.
    assert res.headers["x-frame-options"] == "DENY"
    assert res.headers["content-security-policy"] == "frame-ancestors 'none'"
    assert "microphone" not in res.headers["permissions-policy"]  # the live page needs the mic
    assert "camera=()" in res.headers["permissions-policy"]


def test_live_page_keeps_its_cross_origin_isolation_next_to_the_security_headers():
    res = TestClient(app).get("/live.html")
    assert res.headers["cross-origin-embedder-policy"] == "require-corp"
    assert res.headers["x-frame-options"] == "DENY"


# --- the site shell always revalidates (UI bug 2026-09-30) ---------------------------------------
# Without Cache-Control the browser caches JS/CSS heuristically, so after a deploy a fresh
# index.html ran against an old i18n.js / styles.css: raw "landing.title_html" keys and bare radios.


@pytest.mark.parametrize("path", ["/", "/index.html", "/admin.html", "/i18n.js", "/i18n-dom.js", "/styles.css", "/landing.js"])
def test_site_shell_is_revalidated_on_every_load(path):
    res = TestClient(app).get(path)
    assert res.status_code == 200, path
    assert res.headers["cache-control"] == "no-cache"
    assert "etag" in res.headers  # so revalidation is a cheap 304



def test_the_self_hosted_worker_runtime_is_cross_origin_isolated(client):
    # ADR D61: embed-worker.js imports transformers.js from /vendor/, and onnxruntime-web starts
    # its own workers from the .mjs there. Without COEP on those responses WebKit refuses the
    # import ("worker because of Cross-Origin-Embedder-Policy") and Chromium blocks the .mjs
    # (net::ERR_BLOCKED_BY_RESPONSE). Asserted on the headers alone: site/vendor is provisioned
    # by deploy_bootstrap and absent from a bare checkout.
    for path in ("/vendor/transformers/3.8.1/transformers.min.js", "/vendor/transformers/3.8.1/ort-wasm-simd-threaded.jsep.mjs"):
        res = client.get(path)
        assert res.headers["cross-origin-embedder-policy"] == "require-corp", path
        assert res.headers["cross-origin-opener-policy"] == "same-origin", path


def test_the_weights_are_revalidated_like_the_code_that_reads_them() -> None:
    """A browser that kept old weights next to new code failed every run (ADR D86)."""
    from fastapi.testclient import TestClient

    from qorgan.api import app

    res = TestClient(app).get("/models/weights.json")
    assert res.status_code == 200
    assert res.headers["cache-control"] == "no-cache"
