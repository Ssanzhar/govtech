"""Retention runs on the server's clock (privacy iteration, 2026-09-26).

Before: `purge.expired_receipts` compared the *client-supplied* `timestamp`, which the citizen
route took unbounded -- a future date never expired, a past date was purged at once. Now the
anchor is `received_at` (server clock at storage); the client timestamp is only a bounded hint.
"""

from datetime import UTC, datetime, timedelta

import pytest

from qorgan.reports.retention import CLIENT_CLOCK_SKEW, client_timestamp_in_window, is_expired
from support.numbers import stored_report

NOW = datetime(2026, 9, 26, 12, 0, tzinfo=UTC)


@pytest.mark.parametrize(
    ("stamp", "ok"),
    [
        (NOW, True),
        (NOW + CLIENT_CLOCK_SKEW - timedelta(seconds=1), True),  # a slightly fast phone clock
        (NOW + CLIENT_CLOCK_SKEW + timedelta(seconds=1), False),  # the future: would never expire
        (NOW - timedelta(days=179), True),
        (NOW - timedelta(days=181), False),  # older than retention: nothing to keep it for
        (datetime(2026, 9, 26, 11, 0), True),  # naive = UTC
    ],
)
def test_client_timestamp_is_bounded_by_skew_and_retention(stamp, ok):
    assert client_timestamp_in_window(stamp, now=NOW, retention_days=180) is ok


def test_expiry_follows_received_at_not_the_client_timestamp():
    future_claim = stored_report(number=None, timestamp=NOW + timedelta(days=3650), received_at=NOW - timedelta(days=181))
    backdated = stored_report(number=None, timestamp=NOW - timedelta(days=400), received_at=NOW - timedelta(days=1))
    assert is_expired(future_claim, now=NOW, retention_days=180)  # before: never expired
    assert not is_expired(backdated, now=NOW, retention_days=180)  # before: purged at once


def test_legacy_rows_without_received_at_fall_back_to_a_plausible_timestamp():
    old = stored_report(number=None, timestamp=NOW - timedelta(days=200)).model_copy(update={"received_at": None})
    fresh = stored_report(number=None, timestamp=NOW - timedelta(days=2)).model_copy(update={"received_at": None})
    future = stored_report(number=None, timestamp=NOW + timedelta(days=30)).model_copy(update={"received_at": None})
    assert is_expired(old, now=NOW, retention_days=180)
    assert not is_expired(fresh, now=NOW, retention_days=180)
    # A legacy row whose age cannot be established is not kept forever: it goes now.
    assert is_expired(future, now=NOW, retention_days=180)


def test_retention_must_be_positive():
    with pytest.raises(ValueError):
        is_expired(stored_report(number=None), now=NOW, retention_days=0)
