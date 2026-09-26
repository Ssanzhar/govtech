"""Retention rules for stored reports, on the server's clock (privacy iteration, 2026-09-26).

A report expires `retention_days` after the server received it (`received_at`). The
client-supplied `timestamp` is only a hint about when the call happened: it is accepted only
inside [now - retention, now + a small clock skew], so it can neither make a report immortal
(a future date) nor drag it into the purge on arrival (a date older than retention).

Rows stored before `received_at` existed fall back to their `timestamp` when it is plausible;
a legacy row dated in the future has no establishable age and expires at once (data
minimisation wins over keeping something nobody can date). Pure functions; no I/O.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from qorgan.reports.model import StoredReport

# How far ahead of the server a client clock may run (a phone set a few minutes fast).
CLIENT_CLOCK_SKEW = timedelta(minutes=5)


def _aware(stamp: datetime) -> datetime:
    """Naive stamps are UTC by the store's convention."""
    return stamp if stamp.tzinfo is not None else stamp.replace(tzinfo=UTC)


def _check_retention(retention_days: int) -> None:
    if retention_days <= 0:
        raise ValueError(f"retention_days must be > 0, got {retention_days}")


def client_timestamp_in_window(stamp: datetime, *, now: datetime, retention_days: int) -> bool:
    """Is a client-claimed call time plausible: not beyond the clock skew, not past retention?"""
    _check_retention(retention_days)
    now = _aware(now)
    return now - timedelta(days=retention_days) <= _aware(stamp) <= now + CLIENT_CLOCK_SKEW


def retention_anchor(report: StoredReport, *, now: datetime) -> datetime | None:
    """When the retention clock started for `report`, or None if that cannot be established."""
    if report.received_at is not None:
        return _aware(report.received_at)
    stamp = _aware(report.timestamp)  # legacy row: only the client's word
    return stamp if stamp <= _aware(now) + CLIENT_CLOCK_SKEW else None


def is_expired(report: StoredReport, *, now: datetime, retention_days: int) -> bool:
    _check_retention(retention_days)
    anchor = retention_anchor(report, now=now)
    return anchor is None or anchor < _aware(now) - timedelta(days=retention_days)


def expires_at(received_at: datetime, *, retention_days: int) -> datetime:
    """The latest moment a report received at `received_at` is kept."""
    _check_retention(retention_days)
    return _aware(received_at) + timedelta(days=retention_days)
