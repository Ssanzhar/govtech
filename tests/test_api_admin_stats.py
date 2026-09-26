"""TDD tests for `qorgan.api_admin_stats` — the analyst results/statistics endpoint."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from fastapi.testclient import TestClient

from qorgan.analytics.pipeline import write_organizations_jsonl
from qorgan.api import app
from qorgan.data.incident_seed import write_incidents_jsonl
from qorgan.data.schema import Incident, Label, Organization, TacticTag
from support.analysts import as_analyst
from support.numbers import stored_report


@pytest.fixture()
def client() -> TestClient:
    return TestClient(app, headers=as_analyst())


def _incident(iid: str, *, ts: datetime | None, tags: tuple[str, ...] = ()) -> Incident:
    return Incident(
        id=iid,
        dialogue_id=iid,
        transcript="это служба безопасности банка, продиктуйте код",
        label=Label(risk=0.9, tactic_tags=tuple(TacticTag(id=t) for t in tags)),
        timestamp=ts,
    )


def _seed(tmp_path, monkeypatch, *, organizations, incidents) -> None:
    monkeypatch.setenv("QORGAN_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("QORGAN_TAXONOMY_PATH", "data/taxonomy/tactics.yaml")
    processed = tmp_path / "processed"
    write_organizations_jsonl(organizations, processed / "organizations.jsonl")
    write_incidents_jsonl(incidents, processed / "incidents.jsonl")


def test_stats_degrade_when_no_analysis(client: TestClient, tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("QORGAN_DATA_DIR", str(tmp_path))

    body = client.get("/api/admin/stats").json()

    assert body["available"] is False
    assert body["activity"] == []
    assert body["top_organizations"] == []
    assert body["reports"] == {"submitted": 0, "ingested": 0, "pending": 0, "signals_only": 0}


def test_stats_activity_is_zero_filled_and_counts_days(
    client: TestClient, tmp_path, monkeypatch
) -> None:
    now = datetime.now()
    incidents = [
        _incident("i1", ts=now - timedelta(days=1)),
        _incident("i2", ts=now - timedelta(days=1)),
        _incident("i3", ts=now),
    ]
    org = Organization(id="org_0", members=("i1", "i2", "i3"))
    _seed(tmp_path, monkeypatch, organizations=[org], incidents=incidents)

    body = client.get("/api/admin/stats").json()

    assert body["available"] is True
    activity = body["activity"]
    assert len(activity) == 30  # fixed window, zero-filled
    assert activity[-1]["date"] == now.date().isoformat()
    by_date = {p["date"]: p["count"] for p in activity}
    assert by_date[(now - timedelta(days=1)).date().isoformat()] == 2
    assert sum(p["count"] for p in activity) == 3


def test_stats_trend_and_top_organizations(client: TestClient, tmp_path, monkeypatch) -> None:
    now = datetime.now()
    incidents = [
        _incident("i1", ts=now - timedelta(days=1), tags=("impersonation_bank",)),
        _incident("i2", ts=now - timedelta(days=2)),
        _incident("i3", ts=now - timedelta(days=10)),
        _incident("i4", ts=now - timedelta(days=11)),
    ]
    orgs = [
        Organization(id="org_0", members=("i1", "i2", "i3")),
        Organization(id="org_1", members=("i4",), is_novel=True),
    ]
    _seed(tmp_path, monkeypatch, organizations=orgs, incidents=incidents)

    body = client.get("/api/admin/stats").json()

    trend = body["trend"]
    assert trend["this_week"] == 2
    assert trend["last_week"] == 2
    assert trend["delta_pct"] == pytest.approx(0.0)
    top = body["top_organizations"]
    assert [o["id"] for o in top] == ["org_0", "org_1"]  # by size, descending
    assert top[0]["incidents"] == 3
    assert top[0]["name"]  # display name resolved
    assert top[1]["is_novel"] is True
    assert body["novel_schemes"] == 1


def test_stats_report_counts(client: TestClient, tmp_path, monkeypatch) -> None:
    from qorgan.analytics.intake import report_incident_id

    now = datetime.now()
    ingested_draft = stored_report(
        number=None, transcript="переведите деньги на безопасный счёт",
        timestamp=datetime(2026, 7, 15, 10, 0, tzinfo=UTC), risk_score=84.0,
    )
    pending_draft = stored_report(
        number=None, transcript="назовите код из смс срочно", flagged_phrases=(), tactic_ids=(),
        timestamp=datetime(2026, 7, 16, 10, 0, tzinfo=UTC), risk_score=90.0,
    )
    incidents = [
        _incident("i1", ts=now),
        _incident(report_incident_id(ingested_draft), ts=now),  # already in the analysis
    ]
    org = Organization(id="org_0", members=tuple(i.id for i in incidents))
    _seed(tmp_path, monkeypatch, organizations=[org], incidents=incidents)
    reports = tmp_path / "processed" / "citizen_reports.jsonl"
    reports.write_text(
        ingested_draft.model_dump_json() + "\n" + pending_draft.model_dump_json() + "\n",
        encoding="utf-8",
    )

    body = client.get("/api/admin/stats").json()

    assert body["reports"] == {"submitted": 2, "ingested": 1, "pending": 1, "signals_only": 0}


def test_stats_signals_only_partner_reports_are_pending_and_counted_separately(client: TestClient, tmp_path, monkeypatch) -> None:
    """Regression (code review): `ingested` must be actual incident membership; signals-only
    reports are pending for number-graph placement (C9) and reported on their own."""
    from qorgan.reports.store import prepare_report
    from support.numbers import TEST_HMAC_KEY

    incidents = [_incident("i1", ts=datetime.now())]
    _seed(tmp_path, monkeypatch, organizations=[Organization(id="org_0", members=("i1",))], incidents=incidents)
    reports = tmp_path / "processed" / "citizen_reports.jsonl"
    lines = [
        prepare_report(
            transcript="", phone_number="+7 700 555 66 77", flagged_phrases=(), tactic_ids=("otp_request",),
            timestamp=datetime(2026, 9, 17, 10, i, tzinfo=UTC), risk_score=100.0, hmac_key=TEST_HMAC_KEY,
            source="partner", consent_basis="customer_consent", partner_id="bank_a", partner_reference=f"C-{i}",
        ).model_dump_json()
        for i in range(3)
    ]
    reports.write_text("\n".join(lines) + "\n", encoding="utf-8")

    body = client.get("/api/admin/stats").json()

    assert body["reports"] == {"submitted": 3, "ingested": 0, "pending": 3, "signals_only": 3}
