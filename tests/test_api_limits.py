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
