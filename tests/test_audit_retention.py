"""ADR D66 / legal gap M4: the audit log is kept at least 3 years and is never purged with the
180-day report retention.

The retention is explicit (`QORGAN_AUDIT_RETENTION_DAYS`, floor `MIN_AUDIT_RETENTION_DAYS`) and
structural: no purge or deletion code path touches `audit_log.jsonl`, so a purge run leaves the
log byte-for-byte intact and `python -m qorgan.audit verify` still passes.
"""

from __future__ import annotations

import ast
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from qorgan.audit import AUDIT_FILENAME, AuditEntry, append_audit, verify_audit
from qorgan.config import MIN_AUDIT_RETENTION_DAYS, ConfigError, get_config, load_config
from qorgan.reports.purge import purge_configured
from qorgan.reports.store import REPORTS_FILENAME, append_report, load_reports
from support.numbers import stored_report

KEY = b"tests-only-audit-chain-key-0123456789abcdef"
_SRC = Path(__file__).resolve().parents[1] / "src" / "qorgan"
# Every module that deletes or rewrites stored data on the retention / deletion paths.
_DELETION_MODULES = ("reports/purge.py", "reports/store.py", "analytics/intake.py", "analytics/feedback.py")


def test_audit_retention_is_at_least_three_years_by_default():
    assert MIN_AUDIT_RETENTION_DAYS >= 3 * 365 + 1  # three years, one of them a leap year
    cfg = load_config({})
    assert cfg.audit_retention_days >= MIN_AUDIT_RETENTION_DAYS
    assert cfg.audit_retention_days > cfg.report_retention_days


def test_audit_retention_can_be_raised_but_never_below_the_floor():
    raised = load_config({"QORGAN_AUDIT_RETENTION_DAYS": str(5 * 365)})
    assert raised.audit_retention_days == 5 * 365
    for too_short in ("180", str(MIN_AUDIT_RETENTION_DAYS - 1), "0"):
        with pytest.raises(ConfigError):
            load_config({"QORGAN_AUDIT_RETENTION_DAYS": too_short})


def test_a_report_purge_leaves_the_audit_log_byte_identical_and_verifiable(tmp_path, monkeypatch):
    monkeypatch.setenv("QORGAN_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("QORGAN_REPORT_RETENTION_DAYS", "180")
    processed = tmp_path / "processed"
    reports = processed / REPORTS_FILENAME
    now = datetime.now(UTC)
    expired = stored_report(number=None, timestamp=now - timedelta(days=400))
    append_report(expired, reports)
    audit = processed / AUDIT_FILENAME
    old = AuditEntry(
        timestamp=now - timedelta(days=400), actor_kind="analyst", actor_id="ana", action="case.open", outcome="ok"
    )
    append_audit(old, audit, key=KEY)
    append_audit(old.model_copy(update={"timestamp": now}), audit, key=KEY)
    before = audit.read_bytes()

    removed = purge_configured(get_config(), now=now)

    assert removed == [expired.receipt_id] and load_reports(reports) == []
    assert audit.read_bytes() == before
    assert verify_audit(audit, key=KEY).ok


@pytest.mark.parametrize("module", _DELETION_MODULES)
def test_no_deletion_path_references_the_audit_log(module):
    tree = ast.parse((_SRC / module).read_text(encoding="utf-8"))
    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    names |= {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
    strings = {node.value for node in ast.walk(tree) if isinstance(node, ast.Constant) and isinstance(node.value, str)}
    assert "AUDIT_FILENAME" not in names
    assert not any("audit_log" in s for s in strings)
