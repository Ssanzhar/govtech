"""Tests for the partner intake/export API (`/api/v1`, PLAN_2026-09 C5, ADR D19).

Contract: API-key auth, one report per request, structured tactic hits preferred, a
transcript only if already scrubbed, `consent_basis` required, numbers hashed on receipt,
per-partner rate limit + rolling daily quota, idempotent retries by `partner_reference`,
content-free audit lines, aggregates-only export. Partners can delete only their own reports.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from fastapi.testclient import TestClient

from qorgan.analytics.pipeline import write_organizations_jsonl
from qorgan.api import app
from qorgan.api_partner import _LIMITER
from qorgan.audit import load_audit
from qorgan.data.incident_seed import write_incidents_jsonl
from qorgan.data.schema import Incident, Label, Organization, TacticTag
from qorgan.reports.store import load_reports
from support.numbers import TEST_HMAC_KEY, hashed, prefix

SECRET_A = "bank-a-secret-0123456789abcdef"
SECRET_B = "telecom-b-secret-0123456789abcdef"
QUOTA_A = 3
SCRUBBED = "Алло, это служба безопасности банка.\nПродиктуйте код из SMS."
UNSCRUBBED = "Перезвоните на +7 700 101 20 30 и назовите код."


@pytest.fixture()
def client(tmp_path, monkeypatch) -> TestClient:
    monkeypatch.setenv("QORGAN_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("QORGAN_TAXONOMY_PATH", "data/taxonomy/tactics.yaml")
    monkeypatch.setenv("QORGAN_CUE_LEXICON_PATH", "data/lexicon/hard_signal_cues.yaml")
    monkeypatch.setenv("QORGAN_REASSURANCE_PATTERNS_PATH", "data/lexicon/reassurance_patterns.yaml")
    monkeypatch.setenv("QORGAN_NUMBER_HMAC_KEY", TEST_HMAC_KEY.decode())
    monkeypatch.setenv("QORGAN_PARTNER_API_KEYS", f"bank_a:{SECRET_A}:{QUOTA_A},telecom_b:{SECRET_B}")
    _LIMITER.reset()
    return TestClient(app)


def _auth(secret: str = SECRET_A) -> dict[str, str]:
    return {"X-API-Key": secret}


def _signals(**overrides):
    base = {
        "consent_basis": "customer_consent",
        "tactic_ids": ["otp_request", "safe_account"],
        "phone_number": "+7 700 555 66 77",
        "partner_reference": "CASE-1",
    }
    return {**base, **overrides}


def _reports_file(tmp_path):
    return tmp_path / "processed" / "citizen_reports.jsonl"


def _audit_file(tmp_path):
    return tmp_path / "processed" / "audit_log.jsonl"


# --- auth ----------------------------------------------------------------------------------


def test_missing_or_wrong_key_is_401_and_stores_nothing(client, tmp_path):
    assert client.post("/api/v1/reports", json=_signals()).status_code == 401
    res = client.post("/api/v1/reports", json=_signals(), headers=_auth("not-a-key-0123456789abcdef"))
    assert res.status_code == 401
    assert res.headers["WWW-Authenticate"] == "ApiKey"
    assert not _reports_file(tmp_path).exists()
    assert client.get("/api/v1/organizations").status_code == 401


def test_no_registry_configured_means_partner_api_is_closed(client, monkeypatch):
    monkeypatch.setenv("QORGAN_PARTNER_API_KEYS", "")
    assert client.post("/api/v1/reports", json=_signals(), headers=_auth()).status_code == 401


def test_openapi_declares_the_api_key_scheme(client):
    spec = client.get("/openapi.json").json()
    scheme = spec["components"]["securitySchemes"]["PartnerApiKey"]
    assert scheme == {"type": "apiKey", "in": "header", "name": "X-API-Key", "description": scheme["description"]}
    assert spec["paths"]["/api/v1/reports"]["post"]["security"] == [{"PartnerApiKey": []}]


# --- ingress -------------------------------------------------------------------------------


def test_structured_signals_report_is_stored_minimised_with_provenance(client, tmp_path):
    res = client.post("/api/v1/reports", json=_signals(), headers=_auth())

    assert res.status_code == 201, res.text
    body = res.json()
    assert body["status"] == "stored" and body["source"] == "partner" and body["partner_id"] == "bank_a"
    assert body["consent_basis"] == "customer_consent"
    assert body["number_prefix"] == "+7 700 ***"
    assert body["stored_transcript"] is None
    assert body["tactic_ids"] == ["otp_request", "safe_account"]
    assert len(body["receipt_id"]) == 24 and body["report_id"].startswith("report-")
    assert body["quota"] == {"limit": QUOTA_A, "used": 1, "remaining": QUOTA_A - 1, "window_hours": 24}
    assert res.headers["X-Quota-Limit"] == str(QUOTA_A) and res.headers["X-Quota-Remaining"] == str(QUOTA_A - 1)

    [stored] = load_reports(_reports_file(tmp_path))
    assert stored.source == "partner" and stored.partner_id == "bank_a" and stored.partner_reference == "CASE-1"
    assert stored.number_hash == hashed("+7 700 555 66 77") and stored.number_prefix == prefix("+7 700 555 66 77")
    raw = _reports_file(tmp_path).read_text(encoding="utf-8")
    assert "555 66 77" not in raw and "5556677" not in raw and SECRET_A not in raw


def test_pre_scrubbed_transcript_is_accepted(client):
    res = client.post(
        "/api/v1/reports",
        json=_signals(tactic_ids=[], transcript=SCRUBBED, flagged_phrases=["Продиктуйте код из SMS"], risk_score=91.0),
        headers=_auth(),
    )
    assert res.status_code == 201, res.text
    assert res.json()["stored_transcript"] == SCRUBBED
    assert res.json()["flagged_phrases"] == ["Продиктуйте код из SMS"]


def test_unscrubbed_transcript_is_refused_without_echoing_it(client, tmp_path):
    res = client.post("/api/v1/reports", json=_signals(tactic_ids=[], transcript=UNSCRUBBED), headers=_auth())

    assert res.status_code == 422
    assert "scrub" in res.json()["detail"].lower()
    assert "101 20 30" not in res.text
    assert not _reports_file(tmp_path).exists()


def test_consent_basis_is_required_and_machine_readable(client, tmp_path):
    payload = _signals()
    del payload["consent_basis"]
    assert client.post("/api/v1/reports", json=payload, headers=_auth()).status_code == 422
    assert client.post("/api/v1/reports", json=_signals(consent_basis=""), headers=_auth()).status_code == 422
    assert client.post("/api/v1/reports", json=_signals(consent_basis="Customer said OK"), headers=_auth()).status_code == 422
    assert not _reports_file(tmp_path).exists()


def test_either_tactics_or_transcript_is_required(client):
    assert client.post("/api/v1/reports", json=_signals(tactic_ids=[]), headers=_auth()).status_code == 422
    assert client.post("/api/v1/reports", json=_signals(tactic_ids=["not_a_tactic"]), headers=_auth()).status_code == 422


def test_bulk_bodies_are_rejected(client):
    assert client.post("/api/v1/reports", json=[_signals(), _signals()], headers=_auth()).status_code == 422


def test_number_without_hashing_key_is_refused(client, monkeypatch):
    monkeypatch.setenv("QORGAN_NUMBER_HMAC_KEY", "")
    assert client.post("/api/v1/reports", json=_signals(), headers=_auth()).status_code == 503
    assert client.post("/api/v1/reports", json=_signals(phone_number=None), headers=_auth()).status_code == 201


def test_retries_with_the_same_reference_are_idempotent(client, tmp_path):
    first = client.post("/api/v1/reports", json=_signals(), headers=_auth())
    again = client.post("/api/v1/reports", json=_signals(), headers=_auth())

    assert first.status_code == 201 and again.status_code == 200
    assert again.json()["status"] == "duplicate"
    assert again.json()["receipt_id"] == first.json()["receipt_id"]
    assert len(load_reports(_reports_file(tmp_path))) == 1
    assert again.json()["quota"]["used"] == 1  # a retry does not burn quota


def test_daily_quota_is_enforced_per_partner_and_rolls(client, tmp_path):
    for i in range(QUOTA_A):
        assert client.post("/api/v1/reports", json=_signals(partner_reference=f"C-{i}"), headers=_auth()).status_code == 201
    res = client.post("/api/v1/reports", json=_signals(partner_reference="C-over"), headers=_auth())
    assert res.status_code == 429 and "quota" in res.json()["detail"].lower()
    assert res.headers["X-Quota-Remaining"] == "0"
    # Another partner has its own budget.
    assert client.post("/api/v1/reports", json=_signals(partner_reference="T-1"), headers=_auth(SECRET_B)).status_code == 201
    # A report RECEIVED before the window no longer counts.
    reports = load_reports(_reports_file(tmp_path))
    aged = reports[0].model_copy(update={"received_at": datetime.now(UTC) - timedelta(hours=25)})
    _reports_file(tmp_path).write_text(
        "".join(r.model_dump_json() + "\n" for r in [aged, *reports[1:]]), encoding="utf-8"
    )
    assert client.post("/api/v1/reports", json=_signals(partner_reference="C-after"), headers=_auth()).status_code == 201


def test_backdated_occurred_at_cannot_escape_the_quota(client, tmp_path):
    """Regression (security review): the quota counts server receipt time, not `occurred_at`."""
    old = (datetime.now(UTC) - timedelta(days=30)).isoformat()
    for i in range(QUOTA_A):
        res = client.post("/api/v1/reports", json=_signals(partner_reference=f"B-{i}", occurred_at=old), headers=_auth())
        assert res.status_code == 201 and res.json()["quota"]["used"] == i + 1
    assert client.post("/api/v1/reports", json=_signals(partner_reference="B-over", occurred_at=old), headers=_auth()).status_code == 429
    stored = load_reports(_reports_file(tmp_path))
    assert all(r.received_at is not None and r.received_at > datetime.now(UTC) - timedelta(minutes=1) for r in stored)


def test_occurred_at_is_bounded_by_retention_and_the_clock(client):
    future = (datetime.now(UTC) + timedelta(hours=1)).isoformat()
    ancient = (datetime.now(UTC) - timedelta(days=400)).isoformat()
    assert client.post("/api/v1/reports", json=_signals(occurred_at=future), headers=_auth()).status_code == 422
    assert client.post("/api/v1/reports", json=_signals(occurred_at=ancient), headers=_auth()).status_code == 422
    recent = (datetime.now(UTC) - timedelta(days=3)).isoformat()
    assert client.post("/api/v1/reports", json=_signals(occurred_at=recent), headers=_auth()).status_code == 201


def test_tactic_ids_are_bounded_and_deduplicated(client):
    res = client.post("/api/v1/reports", json=_signals(tactic_ids=["otp_request"] * 3), headers=_auth())
    assert res.status_code == 201 and res.json()["tactic_ids"] == ["otp_request"]
    assert client.post("/api/v1/reports", json=_signals(tactic_ids=["otp_request"] * 33, partner_reference="X"), headers=_auth()).status_code == 422


def test_requests_are_rate_limited_per_partner(client):
    for i in range(_LIMITER.max_requests):
        assert client.get("/api/v1/organizations", headers=_auth()).status_code == 200
    assert client.get("/api/v1/organizations", headers=_auth()).status_code == 429
    assert client.get("/api/v1/organizations", headers=_auth(SECRET_B)).status_code == 200


# --- deletion ------------------------------------------------------------------------------


def test_partner_can_delete_only_its_own_reports(client, tmp_path):
    receipt = client.post("/api/v1/reports", json=_signals(), headers=_auth()).json()["receipt_id"]

    assert client.delete(f"/api/v1/reports/{receipt}", headers=_auth(SECRET_B)).status_code == 404
    assert len(load_reports(_reports_file(tmp_path))) == 1
    assert client.delete(f"/api/v1/reports/{receipt}", headers=_auth()).status_code == 204
    assert load_reports(_reports_file(tmp_path)) == []
    assert client.delete(f"/api/v1/reports/{receipt}", headers=_auth()).status_code == 404
    assert client.delete("/api/v1/reports/not-a-receipt", headers=_auth()).status_code == 422


# --- audit ---------------------------------------------------------------------------------


def test_every_partner_action_leaves_a_content_free_audit_line(client, tmp_path):
    receipt = client.post("/api/v1/reports", json=_signals(transcript=SCRUBBED), headers=_auth()).json()["receipt_id"]
    client.post("/api/v1/reports", json=_signals(tactic_ids=[], transcript=UNSCRUBBED, partner_reference="X"), headers=_auth())
    client.get("/api/v1/organizations", headers=_auth())
    client.delete(f"/api/v1/reports/{receipt}", headers=_auth())

    entries = load_audit(_audit_file(tmp_path))
    assert [(e.actor_id, e.action, e.outcome) for e in entries] == [
        ("bank_a", "report.submit", "stored"),
        ("bank_a", "report.submit", "rejected:unscrubbed_transcript"),
        ("bank_a", "organizations.export", "unavailable"),  # nothing seeded in this fixture
        ("bank_a", "report.delete", "deleted"),
    ]
    assert entries[0].subject == f"receipt:{receipt}"
    raw = _audit_file(tmp_path).read_text(encoding="utf-8")
    assert "Продиктуйте" not in raw and "101 20 30" not in raw and SECRET_A not in raw


# --- export --------------------------------------------------------------------------------


def _incident(iid: str, *, tags: tuple[str, ...], number: str | None, ts: datetime) -> Incident:
    return Incident(
        id=iid, dialogue_id=iid, transcript="это служба безопасности банка, продиктуйте код",
        label=Label(risk=0.9, tactic_tags=tuple(TacticTag(id=t) for t in tags)),
        number_hash=hashed(number), number_prefix=prefix(number), timestamp=ts,
    )


def test_export_is_aggregates_only(client, tmp_path):
    processed = tmp_path / "processed"
    ts = datetime(2026, 9, 10, 9, 0)
    incidents = [
        _incident("i1", tags=("impersonation_bank", "otp_request"), number="+7 700 111 22 33", ts=ts),
        _incident("i2", tags=("otp_request",), number="+7 700 111 22 33", ts=ts + timedelta(days=1)),
        _incident("i3", tags=("investment_scam",), number=None, ts=ts),
    ]
    write_incidents_jsonl(incidents, processed / "incidents.jsonl")
    write_organizations_jsonl(
        [
            Organization(id="org-a", members=("i1", "i2"), numbers=(hashed("+7 700 111 22 33"),), priority=0.8,
                         representative_script="это служба безопасности банка, продиктуйте код"),
            Organization(id="org-b", members=("i3",), priority=0.2, is_novel=True),
        ],
        processed / "organizations.jsonl",
    )

    res = client.get("/api/v1/organizations?locale=ru", headers=_auth())

    assert res.status_code == 200, res.text
    body = res.json()
    assert body["available"] is True and body["generated_at"]
    orgs = {o["id"]: o for o in body["organizations"]}
    assert orgs["org-a"]["incidents"] == 2 and orgs["org-a"]["is_novel"] is False
    assert orgs["org-a"]["tactics"] == [{"id": "otp_request", "count": 2}, {"id": "impersonation_bank", "count": 1}]
    assert orgs["org-a"]["last_activity"] == "2026-09-11"
    assert orgs["org-b"]["is_novel"] is True and orgs["org-b"]["name"]
    for org in body["organizations"]:
        assert set(org) == {"id", "name", "incidents", "tactics", "last_activity", "is_novel", "priority"}
    assert "111 22 33" not in res.text and "продиктуйте" not in res.text and hashed("+7 700 111 22 33") not in res.text


def test_export_degrades_when_no_analysis(client):
    body = client.get("/api/v1/organizations", headers=_auth()).json()
    assert body == {"available": False, "generated_at": body["generated_at"], "organizations": []}


# --- an action that cannot be audited does not happen (review finding, 2026-09-26) ------------


def test_a_submission_that_cannot_be_audited_is_refused_and_stores_nothing(client, tmp_path, monkeypatch):
    from qorgan import api_partner
    from qorgan.audit import AuditIntegrityError

    def broken_audit(*args, **kwargs):
        raise AuditIntegrityError("torn last line")

    monkeypatch.setattr(api_partner, "append_audit", broken_audit)
    res = client.post("/api/v1/reports", json=_signals(), headers=_auth())
    assert res.status_code == 503
    assert not _reports_file(tmp_path).exists() or load_reports(_reports_file(tmp_path)) == []


def test_a_deletion_that_cannot_be_audited_is_refused_and_keeps_the_report(client, tmp_path, monkeypatch):
    from qorgan import api_partner
    from qorgan.audit import AuditIntegrityError

    receipt = client.post("/api/v1/reports", json=_signals(), headers=_auth()).json()["receipt_id"]

    def broken_audit(*args, **kwargs):
        raise AuditIntegrityError("torn last line")

    monkeypatch.setattr(api_partner, "append_audit", broken_audit)
    assert client.delete(f"/api/v1/reports/{receipt}", headers=_auth()).status_code == 503
    assert [r.receipt_id for r in load_reports(_reports_file(tmp_path))] == [receipt]
