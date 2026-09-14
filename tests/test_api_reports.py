"""Tests for `POST/DELETE /api/reports` -- the consented-report ingress (PLAN_2026-09 B5/C3).

Contract: submitting is the explicit action; the response shows exactly what was stored
(scrubbed transcript, coarse number prefix), and the receipt deletes it everywhere.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from qorgan.api import app
from qorgan.api_reports import _LIMITER
from qorgan.reports.store import load_reports

TRANSCRIPT = "Алло, это служба безопасности банка.\nПродиктуйте код из SMS.\nПеревод на +7 700 101 20 30."


@pytest.fixture()
def client(tmp_path, monkeypatch) -> TestClient:
    monkeypatch.setenv("QORGAN_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("QORGAN_TAXONOMY_PATH", "data/taxonomy/tactics.yaml")
    monkeypatch.setenv("QORGAN_CUE_LEXICON_PATH", "data/lexicon/hard_signal_cues.yaml")
    monkeypatch.setenv("QORGAN_REASSURANCE_PATTERNS_PATH", "data/lexicon/reassurance_patterns.yaml")
    monkeypatch.setenv("QORGAN_NUMBER_HMAC_KEY", "tests-only-key")
    _LIMITER.reset()
    return TestClient(app)


def _submission(**overrides):
    base = {
        "transcript": TRANSCRIPT,
        "phone_number": "+7 700 555 66 77",
        "flagged_phrases": ["Продиктуйте код из SMS", "+7 700 101 20 30"],
        "tactic_ids": ["otp_request"],
        "risk_score": 86.0,
        "consent": True,
    }
    return {**base, **overrides}


def _reports_file(tmp_path):
    return tmp_path / "processed" / "citizen_reports.jsonl"


def test_submit_stores_minimised_report_and_returns_what_was_stored(client, tmp_path):
    res = client.post("/api/reports", json=_submission())

    assert res.status_code == 201, res.text
    body = res.json()
    assert len(body["receipt_id"]) == 24 and body["report_id"].startswith("report-")
    assert body["number_prefix"] == "+7 700 ***"
    assert "[PHONE]" in body["stored_transcript"] and "101 20 30" not in body["stored_transcript"]
    assert body["flagged_phrases"] == ["Продиктуйте код из SMS"]  # the number phrase was scrubbed away
    assert body["status"] == "stored"
    raw = _reports_file(tmp_path).read_text(encoding="utf-8")
    assert "555 66 77" not in raw and "5556677" not in raw and "101 20 30" not in raw


def test_consent_must_be_explicitly_true(client, tmp_path):
    assert client.post("/api/reports", json=_submission(consent=False)).status_code == 422
    payload = _submission()
    del payload["consent"]
    assert client.post("/api/reports", json=payload).status_code == 422
    assert not _reports_file(tmp_path).exists()


def test_unknown_tactic_ids_are_rejected(client):
    assert client.post("/api/reports", json=_submission(tactic_ids=["not_a_tactic"])).status_code == 422


def test_number_without_hashing_key_is_refused_but_numberless_is_fine(client, monkeypatch):
    monkeypatch.setenv("QORGAN_NUMBER_HMAC_KEY", "")
    assert client.post("/api/reports", json=_submission()).status_code == 503
    res = client.post("/api/reports", json=_submission(phone_number=None))
    assert res.status_code == 201 and res.json()["number_prefix"] is None


def test_unparseable_number_is_a_client_error(client):
    assert client.post("/api/reports", json=_submission(phone_number="call me")).status_code == 422


def test_delete_by_receipt_removes_the_report(client, tmp_path):
    receipt = client.post("/api/reports", json=_submission()).json()["receipt_id"]
    assert len(load_reports(_reports_file(tmp_path))) == 1

    res = client.delete(f"/api/reports/{receipt}")

    assert res.status_code == 204
    assert load_reports(_reports_file(tmp_path)) == []
    assert client.delete(f"/api/reports/{receipt}").status_code == 404


def test_delete_rejects_malformed_receipts(client):
    assert client.delete("/api/reports/not-a-receipt").status_code == 422


def test_submissions_are_rate_limited_per_client(client):
    for _ in range(_LIMITER.max_requests):
        assert client.post("/api/reports", json=_submission(phone_number=None)).status_code == 201
    assert client.post("/api/reports", json=_submission(phone_number=None)).status_code == 429


def test_blank_transcript_is_rejected(client):
    assert client.post("/api/reports", json=_submission(transcript="   ")).status_code == 422
