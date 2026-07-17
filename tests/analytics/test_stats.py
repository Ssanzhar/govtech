"""TDD tests for `qorgan.analytics.stats` — pure dashboard statistics helpers."""

from datetime import date, datetime

import pytest

from qorgan.analytics.stats import activity_series, weekly_trend
from qorgan.data.schema import Incident, Label

TODAY = date(2026, 7, 17)
NOW = datetime(2026, 7, 17, 12, 0)


def _incident(iid: str, ts: datetime | None) -> Incident:
    return Incident(
        id=iid,
        dialogue_id=iid,
        transcript="это служба безопасности банка",
        label=Label(risk=0.9),
        timestamp=ts,
    )


# --- activity_series ----------------------------------------------------------------------------


def test_activity_series_zero_fills_every_day() -> None:
    incidents = [
        _incident("a", datetime(2026, 7, 15, 9, 0)),
        _incident("b", datetime(2026, 7, 15, 18, 0)),
        _incident("c", datetime(2026, 7, 17, 8, 0)),
    ]

    series = activity_series(incidents, days=7, today=TODAY)

    assert len(series) == 7
    assert series[0][0] == date(2026, 7, 11)
    assert series[-1][0] == date(2026, 7, 17)
    counts = {d.isoformat(): c for d, c in series}
    assert counts["2026-07-15"] == 2
    assert counts["2026-07-17"] == 1
    assert counts["2026-07-12"] == 0


def test_activity_series_ignores_undated_and_out_of_window_incidents() -> None:
    incidents = [
        _incident("a", None),
        _incident("b", datetime(2020, 1, 1)),
        _incident("c", datetime(2026, 7, 17, 8, 0)),
    ]

    series = activity_series(incidents, days=3, today=TODAY)

    assert sum(c for _, c in series) == 1


def test_activity_series_rejects_non_positive_days() -> None:
    with pytest.raises(ValueError):
        activity_series([], days=0, today=TODAY)


# --- weekly_trend -------------------------------------------------------------------------------


def test_weekly_trend_counts_this_and_previous_window() -> None:
    incidents = [
        _incident("a", datetime(2026, 7, 16)),  # this week
        _incident("b", datetime(2026, 7, 12)),  # this week
        _incident("c", datetime(2026, 7, 15)),  # this week
        _incident("d", datetime(2026, 7, 5)),  # previous week
        _incident("e", datetime(2026, 6, 1)),  # older — ignored
        _incident("f", None),  # undated — ignored
    ]

    this_week, last_week, delta_pct = weekly_trend(incidents, now=NOW)

    assert this_week == 3
    assert last_week == 1
    assert delta_pct == pytest.approx(200.0)


def test_weekly_trend_delta_is_none_without_prior_data() -> None:
    incidents = [_incident("a", datetime(2026, 7, 16))]

    this_week, last_week, delta_pct = weekly_trend(incidents, now=NOW)

    assert this_week == 1
    assert last_week == 0
    assert delta_pct is None
