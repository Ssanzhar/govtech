"""ADR D66: the scrubber gets stricter over time; stored records must not break when it does.

The content check (`scrub_text(value) == value`) guards every WRITE. A record already on disk
was checked by the scrubber of its day: re-checking it on read with a stricter scrubber would
make the audit chain "fail" verification without any tampering, and make the reports file
unreadable (console, deletion by receipt and the retention purge all read it). So stored
records are validated with `STORED_RECORD_CONTEXT`, which skips only the content check.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

import qorgan.analytics.feedback as feedback_module
import qorgan.audit as audit_module
import qorgan.reports.model as report_model
from qorgan.analytics.feedback import FeedbackEvent, OrgSnapshot, append_feedback, load_feedback
from qorgan.audit import AuditEntry, append_audit, load_audit, verify_audit
from qorgan.reports.store import append_report, load_reports, prepare_report

NOW = datetime(2026, 10, 9, 12, 0, tzinfo=UTC)
KEY = b"tests-only-audit-chain-key-0123456789abcdef"


def _stricter(text: str) -> str:
    """Stands in for a future scrubber that flags everything currently stored."""
    return text + " [MASKED]"


def test_a_stricter_scrubber_does_not_break_the_audit_chain(tmp_path, monkeypatch):
    path = tmp_path / "audit_log.jsonl"
    entry = AuditEntry(timestamp=NOW, actor_kind="analyst", actor_id="ana", action="case.open", outcome="ok")
    append_audit(entry, path, key=KEY)
    append_audit(entry, path, key=KEY)

    monkeypatch.setattr(audit_module, "scrub_text", _stricter)
    assert verify_audit(path, key=KEY).ok
    assert load_audit(path) == [entry, entry]
    with pytest.raises(ValidationError):  # a new write is still checked
        AuditEntry(timestamp=NOW, actor_kind="analyst", actor_id="ana", action="case.open", outcome="ok")


def test_a_report_stored_under_an_older_scrubber_still_loads(tmp_path, monkeypatch):
    path = tmp_path / "citizen_reports.jsonl"
    report = prepare_report(
        transcript="Назовите код из SMS.",
        phone_number=None,
        flagged_phrases=(),
        tactic_ids=("otp_request",),
        timestamp=NOW,
        risk_score=80.0,
        hmac_key=None,
    )
    append_report(report, path)

    monkeypatch.setattr(report_model, "scrub_text", _stricter)
    assert load_reports(path) == [report]
    with pytest.raises(ValidationError):
        report_model.StoredReport.model_validate(json.loads(report.model_dump_json()))


def test_feedback_stored_under_an_older_scrubber_still_loads(tmp_path, monkeypatch):
    path = tmp_path / "org_feedback.jsonl"
    event = FeedbackEvent(
        timestamp=NOW, analyst_id="ana", action="confirm", org=OrgSnapshot(members=("incident-1",)), note="ok"
    )
    append_feedback(event, path)

    monkeypatch.setattr(feedback_module, "scrub_text", _stricter)
    assert load_feedback(path) == [event]
