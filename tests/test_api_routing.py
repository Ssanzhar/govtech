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
