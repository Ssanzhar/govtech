"""Analyst authentication, roles, purpose limitation and access auditing on `/api/admin`.

The council's deal-breaker was that Level 2 is "architecturally indistinguishable from
surveillance infrastructure". The answer has to be in code: every console route needs an
analyst credential (the identity comes only from it), reading a whole call needs the
investigator role and a declared purpose, and every such act -- including refused ones --
leaves a content-free line in a tamper-evident audit log.
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

import qorgan.api_admin_auth as auth
from qorgan.analytics.feedback import load_feedback
from qorgan.analytics.pipeline import write_organizations_jsonl
from qorgan.api import app
from qorgan.api_ratelimit import SlidingWindowLimiter
from qorgan.audit import load_audit, verify_audit
from qorgan.data.incident_seed import write_incidents_jsonl
from qorgan.data.schema import Incident, Label, Organization, TacticTag
from support.analysts import (
    ANALYST_ID,
    AUDIT_CHAIN_KEY,
    INVESTIGATOR_ID,
    KEY_HEADER,
    as_analyst,
    as_investigator,
)
from support.numbers import hashed, prefix

TRANSCRIPT = "это служба безопасности банка, переведите деньги на безопасный счёт и продиктуйте код из смс"
WRONG_KEY = "not-an-analyst-key-0123456789abcdef"


@pytest.fixture()
def client(tmp_path, monkeypatch) -> TestClient:
    monkeypatch.setenv("QORGAN_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("QORGAN_TAXONOMY_PATH", "data/taxonomy/tactics.yaml")
    processed = tmp_path / "processed"
    incidents = [
        Incident(
            id="i1", dialogue_id="i1", transcript=TRANSCRIPT,
            label=Label(risk=0.9, tactic_tags=(TacticTag(id="safe_account"),)),
            number_hash=hashed("+7 700 101 20 30"), number_prefix=prefix("+7 700 101 20 30"),
        )
    ]
    write_organizations_jsonl([Organization(id="org_0", members=("i1",))], processed / "organizations.jsonl")
    write_incidents_jsonl(incidents, processed / "incidents.jsonl")
    return TestClient(app)


def _audit(tmp_path):
    return load_audit(tmp_path / "processed" / "audit_log.jsonl")


def _raw_audit(tmp_path) -> str:
    path = tmp_path / "processed" / "audit_log.jsonl"
    return path.read_text(encoding="utf-8") if path.exists() else ""


def _open(client, headers, body=None, incident="i1"):
    return client.post(
        f"/api/admin/incidents/{incident}/open", params={"backend": "mock"},
        json=body if body is not None else {"purpose": "pattern_review"}, headers=headers,
    )


# --- fail closed -----------------------------------------------------------------------------


def test_console_is_closed_when_no_analyst_credentials_are_configured(client, tmp_path, monkeypatch):
    monkeypatch.setenv("QORGAN_ANALYST_KEYS", "")
    res = client.get("/api/admin/overview", headers=as_investigator())
    assert res.status_code == 503
    assert "QORGAN_ANALYST_KEYS" in res.json()["detail"]
    assert _open(client, as_investigator()).status_code == 503
    assert _raw_audit(tmp_path) == ""


def test_console_is_closed_without_an_audit_chain_key(client, tmp_path, monkeypatch):
    """No trustworthy audit trail, no console: accountability is a precondition, not a log."""
    monkeypatch.setenv("QORGAN_AUDIT_CHAIN_KEY", "")
    res = client.get("/api/admin/overview", headers=as_analyst())
    assert res.status_code == 503
    assert "QORGAN_AUDIT_CHAIN_KEY" in res.json()["detail"]
    assert _open(client, as_investigator()).status_code == 503
    assert _raw_audit(tmp_path) == ""


def test_missing_key_is_401_and_not_audited(client, tmp_path):
    res = client.get("/api/admin/overview")
    assert res.status_code == 401
    assert res.headers["WWW-Authenticate"] == "ApiKey"
    assert _raw_audit(tmp_path) == ""  # anonymous hits carry no identity; they are only rate-limited


def test_the_old_caller_supplied_identity_no_longer_authenticates(client):
    assert client.get("/api/admin/overview", headers={"X-Analyst-Id": INVESTIGATOR_ID}).status_code == 401
    assert client.get("/api/admin/overview", params={"analyst": INVESTIGATOR_ID}).status_code == 401
    assert _open(client, {"X-Analyst-Id": INVESTIGATOR_ID}).status_code == 401


def test_a_wrong_key_is_401_and_audited_without_the_presented_secret(client, tmp_path):
    res = client.get("/api/admin/overview", headers={KEY_HEADER: WRONG_KEY})
    assert res.status_code == 401

    [entry] = _audit(tmp_path)
    assert (entry.actor_kind, entry.actor_id, entry.action) == ("analyst", "unauthenticated", "auth.denied")
    assert entry.subject == "route:GET /api/admin/overview"
    assert entry.outcome == "denied:invalid_key"
    assert WRONG_KEY not in _raw_audit(tmp_path)


def test_failed_attempts_are_rate_limited_per_client_and_do_not_flood_the_log(client, tmp_path):
    codes = [client.get("/api/admin/overview", headers={KEY_HEADER: WRONG_KEY}).status_code for _ in range(14)]
    assert codes[:10] == [401] * 10 and set(codes[10:]) == {429}
    assert len(_audit(tmp_path)) == 10


# --- identity comes from the credential only ------------------------------------------------


def test_session_endpoint_names_the_signed_in_analyst_and_role(client, tmp_path):
    analyst = client.get("/api/admin/session", headers=as_analyst())
    investigator = client.get("/api/admin/session", headers=as_investigator())

    assert analyst.status_code == 200
    assert analyst.json() == {"id": ANALYST_ID, "role": "analyst", "can_open_cases": False,
                              "open_purposes": ["pattern_review", "citizen_request", "partner_request"]}
    assert investigator.json()["id"] == INVESTIGATOR_ID and investigator.json()["can_open_cases"] is True
    starts = [(e.actor_id, e.action, e.outcome) for e in _audit(tmp_path)]
    assert starts == [(ANALYST_ID, "session.start", "ok:analyst"), (INVESTIGATOR_ID, "session.start", "ok:investigator")]


def test_feedback_is_attributed_to_the_credential_not_to_a_claimed_id(client, tmp_path):
    headers = {**as_analyst(), "X-Analyst-Id": "mallory"}
    res = client.post(
        "/api/admin/organizations/org_0/feedback", params={"analyst": "mallory"},
        json={"action": "confirm"}, headers=headers,
    )
    assert res.status_code == 200, res.text
    [event] = load_feedback(tmp_path / "processed" / "org_feedback.jsonl")
    assert event.analyst_id == ANALYST_ID
    [entry] = _audit(tmp_path)
    assert (entry.actor_id, entry.action) == (ANALYST_ID, "org.confirm")
    assert "mallory" not in _raw_audit(tmp_path)


def test_ingest_is_an_audited_analyst_action(client, tmp_path, monkeypatch):
    import qorgan.api_admin as api_admin

    class _NoEmbed:
        def encode(self, *args, **kwargs):  # pragma: no cover - nothing pending, never called
            raise AssertionError

    monkeypatch.setattr(api_admin, "_EMBEDDER_OVERRIDE", _NoEmbed())
    assert client.post("/api/admin/ingest", headers=as_analyst()).status_code == 200
    [entry] = _audit(tmp_path)
    assert (entry.actor_id, entry.action, entry.outcome) == (ANALYST_ID, "reports.ingest", "ok:0")


# --- the full transcript: investigator role + declared purpose ------------------------------


def test_an_analyst_cannot_open_a_case_and_the_refusal_is_audited(client, tmp_path):
    res = _open(client, as_analyst())
    assert res.status_code == 403
    assert "investigator" in res.json()["detail"]
    assert TRANSCRIPT not in res.text

    [entry] = _audit(tmp_path)
    assert (entry.actor_id, entry.action, entry.outcome) == (ANALYST_ID, "access.denied", "denied:needs_investigator")
    assert entry.subject == "route:POST /api/admin/incidents/{incident_id}/open"


def test_an_analyst_can_still_see_the_excerpt_analysis(client):
    res = client.get("/api/admin/incidents/i1/analysis", params={"backend": "mock"}, headers=as_analyst())
    assert res.status_code == 200 and "transcript" not in res.json()


def test_opening_a_case_needs_a_purpose_from_the_fixed_list(client, tmp_path):
    headers = as_investigator()
    assert client.post("/api/admin/incidents/i1/open", params={"backend": "mock"}, headers=headers).status_code == 422
    assert _open(client, headers, body={}).status_code == 422
    assert _open(client, headers, body={"purpose": "curiosity"}).status_code == 422
    assert _open(client, headers, body={"purpose": "citizen_request", "note": "caller +7 700 101 20 30"}).status_code == 422
    assert _raw_audit(tmp_path) == ""  # nothing was revealed, nothing to account for

    res = _open(client, headers, body={"purpose": "citizen_request", "note": "hotline ticket follow-up"})
    assert res.status_code == 200, res.text
    assert res.json()["transcript"] == TRANSCRIPT
    [entry] = _audit(tmp_path)
    assert (entry.actor_kind, entry.actor_id, entry.action, entry.subject) == ("analyst", INVESTIGATOR_ID, "case.open", "incident:i1")
    assert entry.purpose == "citizen_request"
    assert entry.outcome == "ok: hotline ticket follow-up"
    raw = _raw_audit(tmp_path)
    assert "продиктуйте" not in raw and "101 20 30" not in raw


def test_the_audit_line_is_written_before_the_transcript_is_released(client, tmp_path, monkeypatch):
    """If the audit write fails, the transcript must not be returned."""

    def refuse(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(auth, "append_audit", refuse)
    res = _open(client, as_investigator())
    assert res.status_code == 503
    assert TRANSCRIPT not in res.text


def test_opens_are_rate_limited_per_investigator_and_refusals_audited(client, tmp_path, monkeypatch):
    monkeypatch.setattr(auth, "_OPEN_LIMITER", SlidingWindowLimiter(max_requests=2, window_seconds=3600))
    codes = [_open(client, as_investigator()).status_code for _ in range(3)]
    assert codes == [200, 200, 429]
    outcomes = [e.outcome for e in _audit(tmp_path)]
    assert outcomes == ["ok", "ok", "denied:open_rate_limited"]


def test_every_analyst_has_a_request_budget(client, monkeypatch):
    monkeypatch.setattr(auth, "_ANALYST_LIMITER", SlidingWindowLimiter(max_requests=3, window_seconds=60))
    codes = [client.get("/api/admin/overview", headers=as_analyst()).status_code for _ in range(4)]
    assert codes == [200, 200, 200, 429]
    assert client.get("/api/admin/overview", headers=as_investigator()).status_code == 200  # per analyst


# --- hygiene --------------------------------------------------------------------------------


def test_console_responses_are_never_cached(client):
    assert client.get("/api/admin/overview", headers=as_analyst()).headers["Cache-Control"] == "no-store"
    assert _open(client, as_investigator()).headers["Cache-Control"] == "no-store"


def test_the_console_audit_trail_verifies_as_one_chain(client, tmp_path):
    client.get("/api/admin/session", headers=as_investigator())
    client.get("/api/admin/overview", headers={KEY_HEADER: WRONG_KEY})
    _open(client, as_analyst())
    _open(client, as_investigator(), body={"purpose": "partner_request"})
    client.post("/api/admin/organizations/org_0/feedback", json={"action": "dismiss"}, headers=as_analyst())

    report = verify_audit(tmp_path / "processed" / "audit_log.jsonl", key=AUDIT_CHAIN_KEY.encode())
    assert report.ok and report.chained == 5
    lines = [json.loads(line) for line in _raw_audit(tmp_path).splitlines()]
    assert [line["action"] for line in lines] == ["session.start", "auth.denied", "access.denied", "case.open", "org.dismiss"]


# --- the unaudited analysis stays within the excerpt ------------------------------------------

LATE_PHRASE = "переведите деньги на безопасный счёт"
LONG_TRANSCRIPT = (
    "Алло, добрый день. " + "Мы уточняем данные по вашему обращению, оставайтесь на линии. " * 4 + LATE_PHRASE + "."
)


def test_the_excerpt_analysis_withholds_trigger_phrases_beyond_the_excerpt(client, tmp_path):
    """Measured 2026-09-25 on the seeded incidents: excerpt + trigger phrases exposed a median
    75 % of each transcript through the unaudited analysis. Evidence past the excerpt is
    counted, not quoted -- neither as a span nor inside the templated reason -- until `open`."""
    processed = tmp_path / "processed"
    incident = Incident(id="i2", dialogue_id="i2", transcript=LONG_TRANSCRIPT,
                        label=Label(risk=0.9, tactic_tags=(TacticTag(id="safe_account"),)))
    write_incidents_jsonl([incident], processed / "incidents.jsonl")
    write_organizations_jsonl([Organization(id="org_0", members=("i2",))], processed / "organizations.jsonl")
    assert LONG_TRANSCRIPT.index(LATE_PHRASE) > 200

    analysis = client.get("/api/admin/incidents/i2/analysis", params={"backend": "mock"}, headers=as_investigator()).json()
    assert analysis["spans"] == [] and analysis["withheld_spans"] == 1
    assert LATE_PHRASE not in json.dumps(analysis, ensure_ascii=False)
    assert analysis["tags"]  # the tactic evidence itself is still shown

    opened = _open(client, as_investigator(), incident="i2").json()
    assert [s["text"] for s in opened["spans"]] == [LATE_PHRASE] and opened["withheld_spans"] == 0
    assert LATE_PHRASE in opened["reason"]
