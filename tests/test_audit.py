"""TDD tests for `qorgan.audit` -- the append-only, content-free audit log shared by the
partner API (PLAN_2026-09 C5) and the analyst drill-down (C4)."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from qorgan.audit import AuditEntry, append_audit, load_audit

NOW = datetime(2026, 9, 17, 12, 0, tzinfo=UTC)


def _entry(**overrides) -> AuditEntry:
    base = dict(
        timestamp=NOW,
        actor_kind="partner",
        actor_id="bank_a",
        action="report.submit",
        subject="receipt:0123456789abcdef01234567",
        outcome="stored",
    )
    return AuditEntry(**{**base, **overrides})


def test_append_then_load_round_trips_in_order(tmp_path):
    path = tmp_path / "audit_log.jsonl"
    first = _entry()
    second = _entry(action="report.delete", outcome="deleted")

    append_audit(first, path)
    append_audit(second, path)

    assert load_audit(path) == [first, second]


def test_missing_log_means_no_entries(tmp_path):
    assert load_audit(tmp_path / "nope.jsonl") == []


def test_entries_are_content_free_by_construction():
    """Audit lines describe *that* something happened, never call content or numbers."""
    with pytest.raises(ValidationError):
        _entry(subject="caller +7 700 555 66 77")
    with pytest.raises(ValidationError):
        _entry(outcome="stored transcript: продиктуйте код из смс на +77001112233")
    with pytest.raises(ValidationError):
        _entry(actor_id="")


def test_entries_are_frozen():
    entry = _entry()
    with pytest.raises(ValidationError):
        entry.outcome = "changed"  # type: ignore[misc]


# --- system-generated ids are not call content (found 2026-09-24) ---------------------------

def test_a_receipt_subject_whose_hex_looks_like_a_number_is_still_accepted():
    """`subject="receipt:<32 hex>"` is a system-generated opaque id. ~0.46 % of random
    receipts contain a digit run the PII scrubber reads as a phone number, which used to make
    the audit write fail — losing the audit record for roughly 1 partner submission in 217."""
    from qorgan.audit import AuditEntry

    entry = AuditEntry(
        timestamp=datetime(2026, 9, 24, tzinfo=UTC),
        actor_kind="partner",
        actor_id="partner:acme",
        action="report.submit",
        subject="receipt:743c0ef8f88913843287eaa388fd69bc",
        outcome="stored",
    )
    assert entry.subject == "receipt:743c0ef8f88913843287eaa388fd69bc"


def test_a_subject_carrying_real_call_content_is_still_refused():
    from qorgan.audit import AuditEntry

    with pytest.raises(ValidationError):
        AuditEntry(
            timestamp=datetime(2026, 9, 24, tzinfo=UTC),
            actor_kind="partner",
            actor_id="partner:acme",
            action="report.submit",
            subject="caller +7 701 234 56 78 said to transfer",
            outcome="stored",
        )
