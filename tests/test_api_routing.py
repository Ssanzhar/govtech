"""The admin/live routers must be registered before the `site/` static mount in
`qorgan.api`, or the static host would shadow them (FastAPI/Starlette check routes in
registration order, so this only passes if `app.include_router(...)` runs before
`app.mount("/", StaticFiles(...))`)."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from qorgan.api import app
from support.analysts import as_analyst


@pytest.fixture()
def client() -> TestClient:
    return TestClient(app)


def test_admin_and_live_routes_are_not_shadowed_by_the_static_mount(client: TestClient) -> None:
    admin_res = client.get("/api/admin/overview", headers=as_analyst())
    live_res = client.get("/api/live/scenarios")

    assert admin_res.status_code == 200
    assert "available" in admin_res.json()
    assert live_res.status_code == 200
    assert "scenarios" in live_res.json()


def test_an_unknown_page_gets_the_site_404(client: TestClient) -> None:
    res = client.get("/no-such-page")

    assert res.status_code == 404
    assert res.headers["content-type"].startswith("text/html")
    assert "Qorğan" in res.text and 'href="/"' in res.text


def test_an_unknown_api_path_still_answers_json(client: TestClient) -> None:
    """The site's 404 page is for people: an API client must not have to parse HTML."""
    res = client.get("/api/no-such-route")

    assert res.status_code == 404
    assert res.json() == {"detail": "Not Found"}


def test_health_names_the_deployed_commit_when_the_platform_does(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("RAILWAY_GIT_COMMIT_SHA", raising=False)
    assert "commit" not in client.get("/api/health").json()

    monkeypatch.setenv("RAILWAY_GIT_COMMIT_SHA", "0123456789abcdef0123456789abcdef01234567")
    assert client.get("/api/health").json()["commit"] == "0123456789ab"
