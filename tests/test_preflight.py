"""scripts/preflight.py against a stubbed deployment: a good one passes, each breakage is named."""

from __future__ import annotations

import importlib.util
import json
import sys
from datetime import date
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location("preflight", REPO / "scripts" / "preflight.py")
preflight = importlib.util.module_from_spec(_SPEC)
sys.modules["preflight"] = preflight  # its dataclasses look their module up while they are built
_SPEC.loader.exec_module(preflight)

BASE = "https://demo.example"
ISOLATED = {"cross-origin-opener-policy": "same-origin", "cross-origin-embedder-policy": "require-corp"}
TODAY = date(2026, 11, 8)


def healthy_routes() -> dict[str, preflight.Response]:
    routes: dict[str, preflight.Response] = {}
    for path in preflight.PAGES:
        routes[path] = preflight.Response(200, dict(ISOLATED), b"<html>")
    routes["/api/health"] = preflight.Response(
        200, {}, json.dumps({"status": "ok", "configured_backend": "linear", "threshold": 0.59, "linear_model_source": "web"}).encode()
    )
    for path in preflight.ISOLATED_PATHS:
        routes[path] = preflight.Response(200, dict(ISOLATED), b"")
    for path, _ in preflight.ASSETS:
        routes[path] = preflight.Response(200, {"content-length": "1000000"})
    for path in preflight.SAME_AS_CHECKOUT:
        routes[path] = preflight.Response(200, {}, (REPO / "site" / path.lstrip("/")).read_bytes())
    routes["/api/admin/session"] = preflight.Response(200, {}, b'{"open_access": true}')
    routes["/api/admin/overview?locale=ru"] = preflight.Response(
        200, {}, json.dumps({"available": True, "kpis": {"incidents": 500}, "organizations": [{"last_activity": "2026-11-06"}]}).encode()
    )
    return routes


def fetcher(routes: dict[str, preflight.Response]):
    def fetch(url: str, method: str = "GET") -> preflight.Response:
        return routes.get(url.removeprefix(BASE), preflight.Response(404, {}))

    return fetch


def levels(checks: list) -> dict[str, str]:
    return {check.name: check.level for check in checks}


def run(routes: dict[str, preflight.Response]) -> list:
    return [
        *preflight.check_pages(BASE, fetcher(routes)),
        *preflight.check_health(BASE, fetcher(routes)),
        *preflight.check_isolation(BASE, fetcher(routes)),
        *preflight.check_assets(BASE, fetcher(routes)),
        *preflight.check_same_as_checkout(BASE, fetcher(routes)),
        *preflight.check_demo_data(BASE, fetcher(routes), today=TODAY),
    ]


def test_a_healthy_deployment_passes_every_check():
    checks = run(healthy_routes())
    assert {check.level for check in checks} == {"ok"}, [c for c in checks if c.level != "ok"]


@pytest.mark.parametrize(
    "path", ["/models/gigaam/gigaam-multilingual-220m-ctc-int8/model.int8.onnx", "/vendor/sherpa-onnx/1.13.8/sherpa-onnx-asr.js", "/live.html"]
)
def test_a_missing_file_fails(path):
    routes = healthy_routes()
    routes[path] = preflight.Response(404, {})
    assert "fail" in {check.level for check in run(routes)}


def test_a_network_failure_is_a_failure_not_a_crash():
    checks = preflight.check_pages(BASE, lambda url, method="GET": preflight.Response(0, {"x-error": "timed out"}))
    assert {check.level for check in checks} == {"fail"}
    assert "timed out" in checks[0].detail


def test_another_commit_is_named():
    routes = healthy_routes()
    routes["/models/weights.json"] = preflight.Response(200, {}, b'{"other": "weights"}')
    assert levels(run(routes))["same as this checkout: /models/weights.json"] == "fail"


def test_missing_isolation_headers_fail():
    routes = healthy_routes()
    routes["/live.html"] = preflight.Response(200, {}, b"<html>")
    assert levels(run(routes))["isolation headers on /live.html"] == "fail"


def test_the_mock_backend_fails_the_model_check():
    routes = healthy_routes()
    routes["/api/health"] = preflight.Response(200, {}, b'{"status": "ok", "configured_backend": "mock"}')
    assert levels(run(routes))["health: serving the linear model"] == "fail"


def test_stale_demo_data_warns_and_does_not_fail():
    routes = healthy_routes()
    routes["/api/admin/overview?locale=ru"] = preflight.Response(
        200, {}, json.dumps({"available": True, "kpis": {"incidents": 500}, "organizations": [{"last_activity": "2026-10-29"}]}).encode()
    )
    checks = preflight.check_demo_data(BASE, fetcher(routes), today=TODAY)
    assert levels(checks)["demo data is fresh"] == "warn"
    assert "fail" not in {check.level for check in checks}


def test_data_not_seeded_yet_fails():
    routes = healthy_routes()
    routes["/api/admin/overview?locale=ru"] = preflight.Response(200, {}, b'{"available": false}')
    assert levels(preflight.check_demo_data(BASE, fetcher(routes), today=TODAY))["demo data"] == "fail"


def test_a_closed_console_is_not_a_failure():
    routes = healthy_routes()
    routes["/api/admin/session"] = preflight.Response(401, {})
    checks = preflight.check_demo_data(BASE, fetcher(routes), today=TODAY)
    assert {check.level for check in checks} == {"ok"}


def test_the_named_paths_exist_or_are_built_at_deploy():
    """A renamed page or module must not leave the preflight checking a path nobody serves."""
    for path in (*preflight.PAGES, *preflight.ISOLATED_PATHS, *preflight.SAME_AS_CHECKOUT):
        target = "index.html" if path == "/" else path.lstrip("/")
        assert (REPO / "site" / target).exists(), path
    config = json.loads((REPO / "site/core/qorgan-config.json").read_text(encoding="utf-8"))
    served = {path for path, _ in preflight.ASSETS}
    for model in config["asr"]["models"].values():
        assert f"/models/{model['url']}" in served, model["url"]
    endpoint = config["asr"]["endpoint"]
    assert f"/models/{endpoint['model']['url']}" in served
    assert f"/models/{endpoint['model']['tokens']}" in served
