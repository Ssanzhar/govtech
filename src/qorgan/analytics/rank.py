"""Scam organization priority scoring for the analyst queue.

Invariant: bigger, more recent, faster-growing organizations rank higher. Every score
here is a pure function of its inputs (`math` + `datetime` only) clamped to `[0, 1]`;
no I/O, no mutation of inputs.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from math import log1p

_UNIT_MIN = 0.0
_UNIT_MAX = 1.0
SECONDS_PER_DAY = 86400.0

_SIZE_SATURATION = 50
_W_SIZE = 0.4
_W_RECENCY = 0.35
_W_GROWTH = 0.25


def _validate_unit_interval(value: float, name: str) -> None:
    if not _UNIT_MIN <= value <= _UNIT_MAX:
        raise ValueError(f"{name} must be in [0, 1], got {value}")


def recency_score(latest: datetime, now: datetime, *, horizon_days: float = 30.0) -> float:
    """Linear recency decay: `1.0` when `latest == now`, decaying to `0.0` at
    `horizon_days` old, clamped to `[0, 1]` (older than horizon -> `0.0`; `latest` in
    the future -> `1.0`).

    Raises `ValueError` if `horizon_days <= 0`.
    """
    if horizon_days <= 0:
        raise ValueError(f"horizon_days must be > 0, got {horizon_days}")

    age_days = (now - latest).total_seconds() / SECONDS_PER_DAY
    raw_score = 1.0 - age_days / horizon_days
    return min(_UNIT_MAX, max(_UNIT_MIN, raw_score))


def growth_score(recent_count: int, total_count: int) -> float:
    """Fraction of an org's incidents that are recent: `recent_count / total_count`,
    in `[0, 1]`. Returns `0.0` if `total_count == 0`.

    Raises `ValueError` if either count is negative, or `recent_count > total_count`.
    """
    if recent_count < 0:
        raise ValueError(f"recent_count must be >= 0, got {recent_count}")
    if total_count < 0:
        raise ValueError(f"total_count must be >= 0, got {total_count}")
    if recent_count > total_count:
        raise ValueError(
            f"recent_count ({recent_count}) must be <= total_count ({total_count})"
        )

    return recent_count / total_count if total_count else 0.0


def priority_score(*, size: int, recency: float, growth: float) -> float:
    """Weighted blend of size, recency, and growth into a single `[0, 1]` priority.

    Size contributes via a log-scaled, saturating term so that ever-larger
    organizations yield diminishing returns:
    `min(1.0, log1p(size) / log1p(_SIZE_SATURATION))`.

    Raises `ValueError` if `size < 0`, or `recency`/`growth` are outside `[0, 1]`.
    """
    if size < 0:
        raise ValueError(f"size must be >= 0, got {size}")
    _validate_unit_interval(recency, "recency")
    _validate_unit_interval(growth, "growth")

    size_term = min(_UNIT_MAX, log1p(size) / log1p(_SIZE_SATURATION))
    return _W_SIZE * size_term + _W_RECENCY * recency + _W_GROWTH * growth


def rank_indices(scores: Sequence[float]) -> list[int]:
    """Return indices into `scores` sorted descending by score, ties broken by the
    smaller index. Returns `[]` for empty input.
    """
    return sorted(range(len(scores)), key=lambda index: (-scores[index], index))
