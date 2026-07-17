"""Pure statistics helpers for the analyst dashboard's results view.

Everything here is a deterministic function of its inputs (`datetime` arithmetic and
counting only) — no I/O, no clock reads; callers inject `today`/`now` so tests and the
API stay reproducible.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from datetime import date, datetime, timedelta

from qorgan.data.schema import Incident

DEFAULT_TREND_WINDOW_DAYS = 7


def activity_series(
    incidents: Sequence[Incident], *, days: int, today: date
) -> list[tuple[date, int]]:
    """Per-day incident counts for the `days`-day window ending at `today`, inclusive.

    Every day in the window appears exactly once (zero-filled) so charts render a
    continuous axis. Undated incidents are skipped, never guessed.

    Raises `ValueError` if `days <= 0`.
    """
    if days <= 0:
        raise ValueError(f"days must be > 0, got {days}")

    counts = Counter(
        incident.timestamp.date() for incident in incidents if incident.timestamp is not None
    )
    start = today - timedelta(days=days - 1)
    return [
        (day, counts.get(day, 0))
        for day in (start + timedelta(days=offset) for offset in range(days))
    ]


def weekly_trend(
    incidents: Sequence[Incident],
    *,
    now: datetime,
    window_days: int = DEFAULT_TREND_WINDOW_DAYS,
) -> tuple[int, int, float | None]:
    """Incident counts for the trailing window vs the one before it.

    Returns `(this_window, previous_window, delta_pct)`; `delta_pct` is `None` when the
    previous window is empty (a percentage against zero would be meaningless).

    Raises `ValueError` if `window_days <= 0`.
    """
    if window_days <= 0:
        raise ValueError(f"window_days must be > 0, got {window_days}")

    this_start = now - timedelta(days=window_days)
    previous_start = now - timedelta(days=2 * window_days)
    dated = [incident.timestamp for incident in incidents if incident.timestamp is not None]
    this_window = sum(1 for ts in dated if ts > this_start)
    previous_window = sum(1 for ts in dated if previous_start < ts <= this_start)
    delta_pct = (
        (this_window - previous_window) / previous_window * 100.0 if previous_window else None
    )
    return this_window, previous_window, delta_pct
