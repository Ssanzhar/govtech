"""TDD tests for `qorgan.analytics.rank` -- scam organization priority scoring.

Convention: bigger, more recent, faster-growing organizations rank higher for the
analyst queue. Everything here is pure (`math` + `datetime`) -- no I/O.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from qorgan.analytics.rank import (
    growth_score,
    priority_score,
    rank_indices,
    recency_score,
)

# --- recency_score -----------------------------------------------------------------------


def test_recency_score_is_one_when_latest_equals_now():
    now = datetime(2026, 7, 10, 12, 0, 0)
    assert recency_score(now, now) == pytest.approx(1.0)


def test_recency_score_is_half_at_half_horizon():
    now = datetime(2026, 7, 10, 12, 0, 0)
    latest = now - timedelta(days=15)
    assert recency_score(latest, now, horizon_days=30.0) == pytest.approx(0.5)


def test_recency_score_is_zero_beyond_horizon():
    now = datetime(2026, 7, 10, 12, 0, 0)
    latest = now - timedelta(days=45)
    assert recency_score(latest, now, horizon_days=30.0) == pytest.approx(0.0)


def test_recency_score_is_zero_exactly_at_horizon():
    now = datetime(2026, 7, 10, 12, 0, 0)
    latest = now - timedelta(days=30)
    assert recency_score(latest, now, horizon_days=30.0) == pytest.approx(0.0)


def test_recency_score_future_latest_clamps_to_one():
    now = datetime(2026, 7, 10, 12, 0, 0)
    latest = now + timedelta(days=5)
    assert recency_score(latest, now) == pytest.approx(1.0)


def test_recency_score_default_horizon_is_thirty_days():
    now = datetime(2026, 7, 10, 12, 0, 0)
    latest = now - timedelta(days=15)
    assert recency_score(latest, now) == pytest.approx(0.5)


@pytest.mark.parametrize("bad_horizon", [0.0, -1.0, -30.0])
def test_recency_score_non_positive_horizon_raises(bad_horizon):
    now = datetime(2026, 7, 10, 12, 0, 0)
    with pytest.raises(ValueError):
        recency_score(now, now, horizon_days=bad_horizon)


def test_recency_score_result_always_in_unit_interval():
    now = datetime(2026, 7, 10, 12, 0, 0)
    for days_ago in [-10, 0, 1, 10, 29, 30, 31, 100]:
        score = recency_score(now - timedelta(days=days_ago), now, horizon_days=30.0)
        assert 0.0 <= score <= 1.0


# --- growth_score ------------------------------------------------------------------------


def test_growth_score_returns_fraction_recent_over_total():
    assert growth_score(3, 10) == pytest.approx(0.3)


def test_growth_score_zero_total_returns_zero():
    assert growth_score(0, 0) == pytest.approx(0.0)


def test_growth_score_all_recent_returns_one():
    assert growth_score(5, 5) == pytest.approx(1.0)


def test_growth_score_none_recent_returns_zero():
    assert growth_score(0, 5) == pytest.approx(0.0)


def test_growth_score_negative_recent_count_raises():
    with pytest.raises(ValueError):
        growth_score(-1, 5)


def test_growth_score_negative_total_count_raises():
    with pytest.raises(ValueError):
        growth_score(1, -5)


def test_growth_score_recent_greater_than_total_raises():
    with pytest.raises(ValueError):
        growth_score(6, 5)


# --- priority_score ------------------------------------------------------------------------


def test_priority_score_is_bounded_in_unit_interval():
    score = priority_score(size=1000, recency=1.0, growth=1.0)
    assert 0.0 <= score <= 1.0


def test_priority_score_zero_inputs_is_zero():
    assert priority_score(size=0, recency=0.0, growth=0.0) == pytest.approx(0.0)


def test_priority_score_max_inputs_is_one():
    assert priority_score(size=10**9, recency=1.0, growth=1.0) == pytest.approx(1.0)


def test_priority_score_monotonic_in_size():
    low = priority_score(size=1, recency=0.5, growth=0.5)
    high = priority_score(size=100, recency=0.5, growth=0.5)
    assert high > low


def test_priority_score_monotonic_in_recency():
    low = priority_score(size=10, recency=0.1, growth=0.5)
    high = priority_score(size=10, recency=0.9, growth=0.5)
    assert high > low


def test_priority_score_monotonic_in_growth():
    low = priority_score(size=10, recency=0.5, growth=0.1)
    high = priority_score(size=10, recency=0.5, growth=0.9)
    assert high > low


def test_priority_score_negative_size_raises():
    with pytest.raises(ValueError):
        priority_score(size=-1, recency=0.5, growth=0.5)


@pytest.mark.parametrize("bad_recency", [-0.1, 1.1])
def test_priority_score_recency_out_of_range_raises(bad_recency):
    with pytest.raises(ValueError):
        priority_score(size=10, recency=bad_recency, growth=0.5)


@pytest.mark.parametrize("bad_growth", [-0.1, 1.1])
def test_priority_score_growth_out_of_range_raises(bad_growth):
    with pytest.raises(ValueError):
        priority_score(size=10, recency=0.5, growth=bad_growth)


# --- rank_indices ------------------------------------------------------------------------


def test_rank_indices_orders_descending():
    assert rank_indices([0.1, 0.9, 0.5]) == [1, 2, 0]


def test_rank_indices_ties_broken_by_smaller_index():
    assert rank_indices([0.5, 0.5, 0.9]) == [2, 0, 1]


def test_rank_indices_empty_returns_empty_list():
    assert rank_indices([]) == []


def test_rank_indices_single_element():
    assert rank_indices([0.42]) == [0]


def test_rank_indices_all_equal_preserves_original_order():
    assert rank_indices([0.3, 0.3, 0.3]) == [0, 1, 2]
