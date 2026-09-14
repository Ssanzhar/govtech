"""Tests for `python -m qorgan.reports.purge` -- retention applied everywhere a report reached."""

from datetime import UTC, datetime, timedelta

import pytest

from qorgan.reports.purge import purge
from qorgan.reports.store import append_report, load_reports
from support.numbers import stored_report

NOW = datetime(2026, 9, 14, 12, 0, tzinfo=UTC)


def _paths(tmp_path):
    return {
        "reports_path": tmp_path / "citizen_reports.jsonl",
        "incidents_path": tmp_path / "incidents.jsonl",
        "organizations_path": tmp_path / "organizations.jsonl",
        "embeddings_path": tmp_path / "incident_embeddings.npz",
    }


def test_purge_removes_only_expired_reports_and_reports_receipts(tmp_path):
    paths = _paths(tmp_path)
    old = stored_report(number=None, timestamp=NOW - timedelta(days=200))
    fresh = stored_report(number=None, timestamp=NOW - timedelta(days=5))
    append_report(old, paths["reports_path"])
    append_report(fresh, paths["reports_path"])

    removed = purge(retention_days=180, now=NOW, **paths)

    assert removed == [old.receipt_id]
    assert load_reports(paths["reports_path"]) == [fresh]


def test_purge_with_no_reports_is_a_noop(tmp_path):
    assert purge(retention_days=180, now=NOW, **_paths(tmp_path)) == []


def test_purge_rejects_non_positive_retention(tmp_path):
    with pytest.raises(ValueError):
        purge(retention_days=0, now=NOW, **_paths(tmp_path))
