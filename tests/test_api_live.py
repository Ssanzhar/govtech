"""`qorgan.api_live` -- read-only facts for the live page. The server-side session API is
retired (it kept call content in server memory and had a consent-free report route); the
meter/session logic it wrapped is tested in `tests/live/`."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from qorgan.api import app


@pytest.fixture()
def client() -> TestClient:
    return TestClient(app)


def test_scenarios_lists_both_demo_calls_with_nonempty_lines(client: TestClient) -> None:
    body = client.get("/api/live/scenarios").json()

    ids = {s["id"] for s in body["scenarios"]}
    assert {"live_scam_bank_ru", "live_hard_negative_bank_ru"} <= ids
    for scenario in body["scenarios"]:
        assert scenario["label"]
        assert scenario["lines"], f"{scenario['id']} must have non-empty lines"


def test_capabilities_never_offer_server_audio(client: TestClient) -> None:
    body = client.get("/api/live/capabilities").json()
    assert body["microphone"] is False and "does not accept audio" in body["reason"]


@pytest.mark.parametrize(
    "path",
    ["/api/live/session", "/api/live/session/x/utterance", "/api/live/session/x/end", "/api/live/session/x/report"],
)
def test_the_server_session_api_is_gone(client: TestClient, path: str) -> None:
    assert client.post(path, json={}).status_code in (404, 405)
