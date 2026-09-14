"""TDD tests for `qorgan.reports` -- consented reports as they are actually stored.

Contract (PLAN_2026-09 B5/C2/C3, ADR D14): a submitted report is persisted only after
data minimisation -- transcript PII-scrubbed, caller number reduced to an HMAC digest +
display prefix -- under an unguessable receipt id, and it can be deleted or expired.
"""

from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from qorgan.privacy.numbers import MissingHmacKeyError, hash_phone_number
from qorgan.reports.model import StoredReport, report_to_incident
from qorgan.reports.store import (
    append_report,
    load_reports,
    prepare_report,
    purge_expired,
    remove_report,
)

KEY = b"unit-test-key"
NOW = datetime(2026, 9, 14, 12, 0, tzinfo=UTC)
TRANSCRIPT = "Алло, это служба безопасности банка.\nПродиктуйте код из SMS.\nМой номер +7 700 101 20 30."


def _prepare(**overrides):
    base = dict(
        transcript=TRANSCRIPT,
        phone_number="+7 700 555 66 77",
        flagged_phrases=("Продиктуйте код из SMS", "+7 700 101 20 30"),
        tactic_ids=("otp_request",),
        timestamp=NOW,
        risk_score=86.0,
        hmac_key=KEY,
    )
    return prepare_report(**{**base, **overrides})


# --- prepare_report: minimisation happens before anything is stored ---------------------------


def test_prepare_scrubs_transcript_and_hashes_number():
    report = _prepare()
    assert "+7 700 101 20 30" not in report.transcript and "[PHONE]" in report.transcript
    assert report.number_hash == hash_phone_number("+7 700 555 66 77", key=KEY)
    assert report.number_prefix == "+7 700 ***"
    assert report.source == "citizen" and report.consent_basis == "citizen_explicit_submit"
    assert len(report.receipt_id) == 24


def test_prepare_drops_flagged_phrases_that_no_longer_occur_after_scrubbing():
    report = _prepare()
    assert report.flagged_phrases == ("Продиктуйте код из SMS",)


def test_prepare_without_number_leaves_hash_and_prefix_none():
    report = _prepare(phone_number=None)
    assert report.number_hash is None and report.number_prefix is None


def test_prepare_without_key_refuses_a_number_but_not_a_numberless_report():
    with pytest.raises(MissingHmacKeyError):
        _prepare(hmac_key=None)
    assert _prepare(hmac_key=None, phone_number=None).number_hash is None


def test_prepare_rejects_an_unparseable_number():
    with pytest.raises(ValueError):
        _prepare(phone_number="call me maybe")


def test_receipt_ids_are_unique_and_unguessable_looking():
    ids = {_prepare().receipt_id for _ in range(20)}
    assert len(ids) == 20


# --- StoredReport: the schema refuses raw PII ---------------------------------------------------


def test_stored_report_refuses_unscrubbed_transcript():
    with pytest.raises(ValidationError):
        StoredReport(
            receipt_id="a" * 24, transcript="звоните +7 700 101 20 30", timestamp=NOW, risk_score=1.0
        )


def test_stored_report_refuses_raw_number_in_hash_field():
    with pytest.raises(ValidationError):
        StoredReport(receipt_id="a" * 24, transcript="hi", timestamp=NOW, risk_score=1.0, number_hash="+7 700 101 20 30")


def test_report_to_incident_carries_hash_prefix_and_regrounded_spans():
    report = _prepare()
    incident = report_to_incident(report, incident_id="report-001")
    assert incident.number_hash == report.number_hash
    assert incident.number_prefix == "+7 700 ***"
    assert [s.text for s in incident.label.trigger_spans] == ["Продиктуйте код из SMS"]
    assert incident.label.risk == pytest.approx(0.86)
    assert incident.timestamp == NOW


# --- store: append / load / remove / purge ----------------------------------------------------


def test_append_and_load_round_trip(tmp_path):
    path = tmp_path / "reports.jsonl"
    a, b = _prepare(), _prepare(phone_number=None)
    append_report(a, path)
    append_report(b, path)
    assert load_reports(path) == [a, b]


def test_load_missing_file_is_empty(tmp_path):
    assert load_reports(tmp_path / "nope.jsonl") == []


def test_stored_file_never_contains_the_raw_number(tmp_path):
    path = tmp_path / "reports.jsonl"
    append_report(_prepare(), path)
    text = path.read_text(encoding="utf-8")
    assert "5556677" not in text and "555 66 77" not in text and "101 20 30" not in text


def test_remove_report_rewrites_without_it_and_returns_it(tmp_path):
    path = tmp_path / "reports.jsonl"
    a, b = _prepare(), _prepare()
    append_report(a, path)
    append_report(b, path)
    removed = remove_report(a.receipt_id, path)
    assert removed == a
    assert load_reports(path) == [b]
    assert remove_report("f" * 24, path) is None


def test_purge_expired_removes_only_reports_older_than_retention(tmp_path):
    path = tmp_path / "reports.jsonl"
    old = _prepare(timestamp=NOW - timedelta(days=200))
    fresh = _prepare(timestamp=NOW - timedelta(days=10))
    append_report(old, path)
    append_report(fresh, path)
    purged = purge_expired(path, retention_days=180, now=NOW)
    assert purged == [old]
    assert load_reports(path) == [fresh]


def test_purge_rejects_non_positive_retention(tmp_path):
    with pytest.raises(ValueError):
        purge_expired(tmp_path / "r.jsonl", retention_days=0, now=NOW)
