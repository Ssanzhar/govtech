"""FPR-constrained alert-threshold selection for the demo's alert cutoff.

Convention (matches `qorgan.eval.metrics`): positive class = scam = 1, negative =
legitimate = 0. FPR is the PRIMARY constraint (CLAUDE.md SS3.5) -- a false alarm on a
real bank call destroys trust, so `select_threshold` maximizes recall subject to an FPR
ceiling rather than the reverse. All metric math is delegated to `qorgan.eval.metrics`;
nothing here re-derives a confusion-matrix rate.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from qorgan.eval.metrics import binarize, f1, false_positive_rate, precision, recall

_RATE_MIN = 0.0
_RATE_MAX = 1.0
_MIN_STEPS = 2


@dataclass(frozen=True)
class ThresholdChoice:
    """The selected alert cutoff plus the metrics it achieves on the given data."""

    threshold: float
    fpr: float
    recall: float
    precision: float
    f1: float


def _validate_non_empty_equal_length(y_true: Sequence[int], y_scores: Sequence[float]) -> None:
    if len(y_true) != len(y_scores):
        raise ValueError(
            f"y_true and y_scores must have the same length, got {len(y_true)} and {len(y_scores)}"
        )
    if not y_true:
        raise ValueError("y_true and y_scores must not be empty")


def _validate_rate(value: float, name: str) -> None:
    if not _RATE_MIN <= value <= _RATE_MAX:
        raise ValueError(f"{name} must be in [0, 1], got {value}")


def sweep_thresholds(
    y_true: Sequence[int], y_scores: Sequence[float], *, steps: int = 101
) -> list[dict]:
    """Evaluate `steps` thresholds evenly spaced over `[0.0, 1.0]` inclusive.

    Returns one dict per threshold, `{"threshold", "fpr", "recall", "precision", "f1"}`,
    ascending by threshold. Raises `ValueError` if `y_true`/`y_scores` differ in length,
    are empty, or `steps < 2`.
    """
    _validate_non_empty_equal_length(y_true, y_scores)
    if steps < _MIN_STEPS:
        raise ValueError(f"steps must be >= {_MIN_STEPS}, got {steps}")

    y_true_list = list(y_true)
    y_scores_list = list(y_scores)
    last_index = steps - 1

    rows: list[dict] = []
    for i in range(steps):
        threshold = i / last_index
        y_pred = binarize(y_scores_list, threshold)
        rows.append(
            {
                "threshold": threshold,
                "fpr": false_positive_rate(y_true_list, y_pred),
                "recall": recall(y_true_list, y_pred),
                "precision": precision(y_true_list, y_pred),
                "f1": f1(y_true_list, y_pred),
            }
        )
    return rows


def select_threshold(
    y_true: Sequence[int],
    y_scores: Sequence[float],
    *,
    max_fpr: float,
    min_recall: float = 0.0,
    steps: int = 101,
) -> ThresholdChoice:
    """Pick the alert cutoff that maximizes recall subject to `fpr <= max_fpr` and
    `recall >= min_recall`.

    Tie-break among qualifying thresholds: lower fpr, then higher threshold. If no
    threshold satisfies both constraints, fall back to the threshold with the minimum
    fpr overall (tie-break: higher recall, then higher threshold).

    Raises `ValueError` if `max_fpr`/`min_recall` are outside `[0, 1]`, or if
    `y_true`/`y_scores` are invalid per `sweep_thresholds`.
    """
    _validate_non_empty_equal_length(y_true, y_scores)
    _validate_rate(max_fpr, "max_fpr")
    _validate_rate(min_recall, "min_recall")

    rows = sweep_thresholds(y_true, y_scores, steps=steps)
    qualifying = [row for row in rows if row["fpr"] <= max_fpr and row["recall"] >= min_recall]

    if qualifying:
        best = max(qualifying, key=lambda row: (row["recall"], -row["fpr"], row["threshold"]))
    else:
        best = max(rows, key=lambda row: (-row["fpr"], row["recall"], row["threshold"]))

    return ThresholdChoice(
        threshold=best["threshold"],
        fpr=best["fpr"],
        recall=best["recall"],
        precision=best["precision"],
        f1=best["f1"],
    )
