"""The cloud second opinion (`llm` backend, Gemini) sends call text to Google, outside Kazakhstan.

Before 2026-09-26 any caller could pass `backend=llm` to `/api/analyze` or the admin analysis
routes, with no notice and no consent, and the classifier cached the verbatim trigger phrases on
disk -- so "/api/analyze persists nothing" was false on that backend. Now: the tier is off unless
the operator enables it (`QORGAN_CLOUD_TIER=on`), a request needs the citizen's explicit
`cloud_consent`, request-time scoring never writes the cache, and analyst routes never use it.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from qorgan.api import app
from qorgan.classifier import llm_classifier, predict
from qorgan.data.schema import ScoreResult

TRANSCRIPT = "Это служба безопасности банка. Назовите код из SMS, никому не говорите."


@pytest.fixture()
def cloud_calls(monkeypatch) -> list[dict]:
    """Stand-in for Gemini: records every call instead of sending anything anywhere."""
    calls: list[dict] = []

    def fake_classify(transcript: str, **kwargs) -> ScoreResult:
        calls.append(kwargs)
        return ScoreResult(risk=0.95, tags=(), attributions=(), backend="llm")

    monkeypatch.setattr(llm_classifier, "classify", fake_classify)
    return calls


def _analyze(**extra):
    return TestClient(app).post("/api/analyze", json={"transcript": TRANSCRIPT, **extra})


def test_cloud_tier_is_off_by_default_and_a_request_for_it_is_refused(monkeypatch, cloud_calls):
    monkeypatch.delenv("QORGAN_CLOUD_TIER", raising=False)
    res = _analyze(backend="llm", cloud_consent=True)
    assert res.status_code == 403
    assert cloud_calls == []


def test_a_configured_llm_default_does_not_bypass_the_switch(monkeypatch, cloud_calls):
    monkeypatch.setenv("QORGAN_CLASSIFIER_BACKEND", "llm")
    monkeypatch.setenv("QORGAN_CLOUD_TIER", "off")
    assert _analyze().status_code == 403
    assert cloud_calls == []


def test_enabled_cloud_tier_still_needs_the_callers_explicit_consent(monkeypatch, cloud_calls):
    monkeypatch.setenv("QORGAN_CLOUD_TIER", "on")
    res = _analyze(backend="llm")
    assert res.status_code == 422
    assert cloud_calls == []


def test_consented_cloud_request_is_scored_without_writing_the_cache(monkeypatch, cloud_calls):
    monkeypatch.setenv("QORGAN_CLOUD_TIER", "on")
    res = _analyze(backend="llm", cloud_consent=True)
    assert res.status_code == 200
    assert res.json()["backend"] == "llm"
    assert cloud_calls == [{"use_cache": False}]


def test_local_backends_need_no_consent(monkeypatch, cloud_calls):
    monkeypatch.delenv("QORGAN_CLOUD_TIER", raising=False)
    assert _analyze(backend="mock").status_code == 200
    assert cloud_calls == []


def test_predict_passes_the_cache_switch_through(cloud_calls):
    predict.score(TRANSCRIPT, backend="llm", use_cache=False)
    assert cloud_calls == [{"use_cache": False}]


def test_bad_switch_value_is_a_config_error(monkeypatch):
    from qorgan.config import ConfigError, load_config

    monkeypatch.setenv("QORGAN_CLOUD_TIER", "maybe")
    with pytest.raises(ConfigError):
        load_config()


def test_analyst_routes_never_reach_the_cloud(monkeypatch, cloud_calls):
    from support.analysts import INVESTIGATOR_KEY

    monkeypatch.setenv("QORGAN_CLOUD_TIER", "on")
    client = TestClient(app)
    headers = {"X-Analyst-Key": INVESTIGATOR_KEY}
    res = client.get("/api/admin/incidents/any/analysis", params={"backend": "llm"}, headers=headers)
    assert res.status_code == 422
    res = client.post(
        "/api/admin/incidents/any/open", params={"backend": "llm"},
        json={"purpose": "pattern_review"}, headers=headers,
    )
    assert res.status_code == 422
    assert cloud_calls == []
