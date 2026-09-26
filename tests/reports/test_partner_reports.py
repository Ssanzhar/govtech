"""TDD tests for the partner-report extensions of the reports package (PLAN_2026-09 C5):
`StoredReport` provenance fields, signals-only reports, quota counting and idempotency
lookups over the reports file."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from qorgan.analytics.intake import pending_reports, report_incident_id
from qorgan.reports.model import StoredReport
from qorgan.reports.partner import find_partner_report, partner_reports_since
from qorgan.reports.store import append_report, load_reports, prepare_report
from support.numbers import TEST_HMAC_KEY, hashed

NOW = datetime(2026, 9, 17, 12, 0, tzinfo=UTC)


def _partner_report(**overrides) -> StoredReport:
    base = dict(
        transcript="",
        phone_number="+7 700 555 66 77",
        flagged_phrases=(),
        tactic_ids=("otp_request", "safe_account"),
        timestamp=NOW,
        risk_score=100.0,
        hmac_key=TEST_HMAC_KEY,
        source="partner",
        consent_basis="customer_consent",
        partner_id="bank_a",
        partner_reference="CASE-1",
    )
    return prepare_report(**{**base, **overrides})


# --- schema --------------------------------------------------------------------------------


def test_partner_report_carries_provenance_and_hashed_number():
    report = _partner_report()
    assert report.source == "partner"
    assert report.partner_id == "bank_a"
    assert report.partner_reference == "CASE-1"
    assert report.consent_basis == "customer_consent"
    assert report.number_hash == hashed("+7 700 555 66 77")
    assert report.number_prefix == "+7 700 ***"
    assert report.transcript == "" and report.tactic_ids == ("otp_request", "safe_account")


def test_signals_only_needs_at_least_one_tactic():
    with pytest.raises(ValidationError):
        _partner_report(tactic_ids=())


def test_citizen_reports_still_require_a_transcript():
    with pytest.raises(ValidationError):
        prepare_report(
            transcript="   ", phone_number=None, flagged_phrases=(), tactic_ids=("otp_request",),
            timestamp=NOW, risk_score=50.0, hmac_key=None,
        )


def test_partner_source_and_partner_id_go_together():
    with pytest.raises(ValidationError):
        _partner_report(partner_id=None)
    with pytest.raises(ValidationError):
        prepare_report(
            transcript="алло это банк", phone_number=None, flagged_phrases=(), tactic_ids=(),
            timestamp=NOW, risk_score=50.0, hmac_key=None, partner_id="bank_a",  # citizen source
        )


def test_partner_transcript_is_scrubbed_on_receipt_like_citizen_reports():
    report = _partner_report(transcript="перезвоните на +7 701 222 33 44 срочно", tactic_ids=())
    assert "222 33 44" not in report.transcript and "[PHONE]" in report.transcript


# --- incident ids / intake -----------------------------------------------------------------


def test_signals_only_reports_get_distinct_deterministic_ids():
    a = _partner_report(partner_reference="CASE-1")
    b = _partner_report(partner_reference="CASE-2", tactic_ids=("remote_access",))
    same_as_a = _partner_report(partner_reference="CASE-1")
    assert report_incident_id(a) == report_incident_id(same_as_a)
    assert report_incident_id(a) != report_incident_id(b)
    assert report_incident_id(a).startswith("report-")


def test_signals_only_reports_are_pending_for_number_graph_placement(tmp_path):
    """No transcript -> nothing to embed, but the number graph can still place them (C9)."""
    path = tmp_path / "citizen_reports.jsonl"
    append_report(_partner_report(), path)
    with_text = _partner_report(transcript="это служба безопасности банка, назовите код", partner_reference="CASE-3")
    append_report(with_text, path)

    pending = pending_reports(path, incidents=[])

    assert [r.partner_reference for r in pending] == ["CASE-1", "CASE-3"]


# --- quota + idempotency helpers -----------------------------------------------------------


def test_partner_reports_since_counts_by_receipt_time_for_that_partner_only(tmp_path):
    path = tmp_path / "citizen_reports.jsonl"
    window = NOW - timedelta(hours=24)
    append_report(_partner_report(received_at=NOW - timedelta(hours=1)), path)
    append_report(_partner_report(received_at=NOW - timedelta(hours=30), partner_reference="OLD"), path)
    # Backdated `timestamp` (partner-supplied occurred_at) must still count: receipt was now.
    append_report(_partner_report(timestamp=NOW - timedelta(days=400), received_at=NOW, partner_reference="BACKDATED"), path)
    append_report(_partner_report(partner_id="telecom_b", partner_reference="T-1", received_at=NOW), path)
    append_report(
        prepare_report(transcript="алло это банк", phone_number=None, flagged_phrases=(), tactic_ids=(),
                       timestamp=NOW, risk_score=50.0, hmac_key=None, received_at=NOW),
        path,
    )
    reports = load_reports(path)

    assert partner_reports_since(reports, partner_id="bank_a", since=window) == 2
    assert partner_reports_since(reports, partner_id="telecom_b", since=window) == 1
    assert partner_reports_since(reports, partner_id="nobody", since=window) == 0
    assert partner_reports_since([], partner_id="bank_a", since=window) == 0


def test_legacy_reports_without_received_at_fall_back_to_timestamp():
    # Rows written before `received_at` existed: prepare_report now always stamps it.
    legacy = _partner_report(timestamp=NOW - timedelta(hours=2)).model_copy(update={"received_at": None})
    assert legacy.received_at is None
    assert partner_reports_since([legacy], partner_id="bank_a", since=NOW - timedelta(hours=24)) == 1


def test_find_partner_report_matches_partner_and_reference_only(tmp_path):
    path = tmp_path / "citizen_reports.jsonl"
    mine = _partner_report(partner_reference="CASE-1")
    append_report(mine, path)
    append_report(_partner_report(partner_id="telecom_b", partner_reference="CASE-1"), path)
    reports = load_reports(path)

    assert find_partner_report(reports, partner_id="bank_a", reference="CASE-1") == mine
    assert find_partner_report(reports, partner_id="bank_a", reference="CASE-9") is None
    assert find_partner_report(reports, partner_id="bank_a", reference=None) is None
