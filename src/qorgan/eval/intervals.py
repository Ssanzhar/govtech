"""Confidence intervals for the FPR-first eval harness (PLAN_2026-09 A1).

A point estimate on a small split is not a metric: `FPR = 0.000` on 24 negatives is
consistent with a true FPR above 10 %. Every headline rate therefore carries an exact
binomial (Clopper-Pearson) interval, and PR-AUC a percentile-bootstrap one. Pure functions,
deterministic (the bootstrap is seeded), no I/O.

Conventions: `confidence` is the two-sided coverage (0.95 -> a 95 % interval);
`upper_bound` is the one-sided bound to quote for "at most X with 95 % confidence".
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy.stats import beta
from sklearn.metrics import average_precision_score

DEFAULT_CONFIDENCE = 0.95
DEFAULT_BOOTSTRAP_RESAMPLES = 1000
DEFAULT_BOOTSTRAP_SEED = 42

_METHOD_CLOPPER_PEARSON = "clopper-pearson"
_METHOD_BOOTSTRAP = "bootstrap-percentile"


@dataclass(frozen=True)
class Interval:
    """A closed interval `[low, high]` with the coverage and method that produced it."""

    low: float
    high: float
    confidence: float
    method: str

    def __post_init__(self) -> None:
        if self.low > self.high:
            raise ValueError(f"interval bounds are inverted: low={self.low} > high={self.high}")

    def as_tuple(self) -> tuple[float, float]:
        return (self.low, self.high)


def _validate_counts(successes: int, trials: int) -> None:
    if trials <= 0:
        raise ValueError(f"trials must be > 0, got {trials}")
    if not 0 <= successes <= trials:
        raise ValueError(f"successes must be in [0, {trials}], got {successes}")


def _validate_confidence(confidence: float) -> None:
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")


def clopper_pearson(successes: int, trials: int, *, confidence: float = DEFAULT_CONFIDENCE) -> Interval:
    """Exact two-sided binomial interval for `successes / trials` (Clopper & Pearson, 1934).

    Uses the beta-quantile form: `low = Beta(a/2; k, n-k+1)`, `high = Beta(1-a/2; k+1, n-k)`,
    with `low = 0` when `k = 0` and `high = 1` when `k = n`. Raises `ValueError` for
    `trials <= 0`, `successes` outside `[0, trials]`, or `confidence` outside `(0, 1)`.
    """
    _validate_counts(successes, trials)
    _validate_confidence(confidence)
    alpha = 1.0 - confidence
    low = 0.0 if successes == 0 else float(beta.ppf(alpha / 2, successes, trials - successes + 1))
    high = 1.0 if successes == trials else float(beta.ppf(1 - alpha / 2, successes + 1, trials - successes))
    return Interval(low=low, high=high, confidence=confidence, method=_METHOD_CLOPPER_PEARSON)


def upper_bound(successes: int, trials: int, *, confidence: float = DEFAULT_CONFIDENCE) -> float:
    """One-sided exact upper bound: the largest rate still consistent with the data at
    `confidence`. Equals the high end of the two-sided interval at `2*confidence - 1`
    (e.g. one-sided 95 % == two-sided 90 %). With zero successes this is the "rule of three"
    number: `0/59 -> 0.05`.
    """
    _validate_counts(successes, trials)
    _validate_confidence(confidence)
    if successes == trials:
        return 1.0
    return float(beta.ppf(confidence, successes + 1, trials - successes))


def binomial_interval(
    numerator: int, denominator: int, *, confidence: float = DEFAULT_CONFIDENCE
) -> Interval | None:
    """`clopper_pearson` for a rate whose denominator may be zero (an absent class);
    returns `None` in that case so reports can render `-` instead of a fake `[0, 1]`."""
    if denominator == 0:
        return None
    return clopper_pearson(numerator, denominator, confidence=confidence)


def bootstrap_pr_auc(
    y_true: Sequence[int],
    y_scores: Sequence[float],
    *,
    n_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    confidence: float = DEFAULT_CONFIDENCE,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> Interval | None:
    """Percentile-bootstrap interval for average precision (PR-AUC).

    Resamples `(y_true, y_scores)` pairs with replacement `n_resamples` times; resamples
    that lose one class (PR-AUC undefined) are discarded rather than scored as 0. Returns
    `None` when the input itself lacks both classes. Raises `ValueError` for mismatched
    lengths, `n_resamples <= 0`, or an invalid `confidence`.
    """
    truths = np.asarray(list(y_true), dtype=int)
    scores = np.asarray(list(y_scores), dtype=float)
    if truths.shape[0] != scores.shape[0]:
        raise ValueError(f"y_true and y_scores must have the same length, got {truths.shape[0]} and {scores.shape[0]}")
    if n_resamples <= 0:
        raise ValueError(f"n_resamples must be > 0, got {n_resamples}")
    _validate_confidence(confidence)
    if truths.shape[0] == 0 or len(np.unique(truths)) < 2:
        return None

    rng = np.random.default_rng(seed)
    n = truths.shape[0]
    estimates: list[float] = []
    for _ in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        sample_truths = truths[idx]
        if len(np.unique(sample_truths)) < 2:
            continue
        estimates.append(float(average_precision_score(sample_truths, scores[idx])))
    if not estimates:
        return None

    alpha = 1.0 - confidence
    low, high = np.percentile(estimates, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return Interval(low=float(low), high=float(high), confidence=confidence, method=_METHOD_BOOTSTRAP)
