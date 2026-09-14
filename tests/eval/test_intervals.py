"""TDD tests for `qorgan.eval.intervals` -- confidence intervals for the eval harness.

Why this exists (PLAN_2026-09 A1): a point estimate of `FPR = 0.000` on 24 negatives is
statistically compatible with a true FPR above 10 %. Every headline rate must therefore
carry an exact binomial (Clopper-Pearson) interval, and PR-AUC a bootstrap one.
Reference values below were computed independently with `scipy.stats.beta.ppf`.
"""

import pytest

from qorgan.eval.intervals import (
    Interval,
    binomial_interval,
    bootstrap_pr_auc,
    clopper_pearson,
    upper_bound,
)

# --- clopper_pearson (two-sided exact binomial) ------------------------------------------------


def test_zero_successes_of_24_two_sided_95():
    ci = clopper_pearson(0, 24)
    assert ci.low == 0.0
    assert ci.high == pytest.approx(0.1423, abs=1e-3)
    assert ci.confidence == 0.95
    assert ci.method == "clopper-pearson"


def test_all_successes_of_24_two_sided_95():
    ci = clopper_pearson(24, 24)
    assert ci.low == pytest.approx(0.8577, abs=1e-3)
    assert ci.high == 1.0


def test_four_of_24_matches_reference():
    ci = clopper_pearson(4, 24)
    assert ci.low == pytest.approx(0.047, abs=1e-3)
    assert ci.high == pytest.approx(0.374, abs=1e-3)


def test_interval_contains_point_estimate():
    for k, n in ((0, 10), (3, 10), (10, 10), (1, 300)):
        ci = clopper_pearson(k, n)
        assert ci.low <= k / n <= ci.high


def test_symmetry_under_relabeling():
    a = clopper_pearson(4, 24)
    b = clopper_pearson(20, 24)
    assert a.low == pytest.approx(1.0 - b.high)
    assert a.high == pytest.approx(1.0 - b.low)


def test_wider_confidence_gives_wider_interval():
    narrow = clopper_pearson(4, 24, confidence=0.90)
    wide = clopper_pearson(4, 24, confidence=0.99)
    assert wide.low <= narrow.low and wide.high >= narrow.high


@pytest.mark.parametrize("k, n", [(0, 0), (-1, 10), (11, 10)])
def test_invalid_counts_raise(k, n):
    with pytest.raises(ValueError):
        clopper_pearson(k, n)


@pytest.mark.parametrize("confidence", [0.0, 1.0, 1.5, -0.1])
def test_invalid_confidence_raises(confidence):
    with pytest.raises(ValueError):
        clopper_pearson(1, 10, confidence=confidence)


# --- upper_bound (one-sided) -----------------------------------------------------------------


def test_one_sided_upper_bound_zero_of_24():
    # The number to quote for "FPR is at most X with 95 % confidence".
    assert upper_bound(0, 24) == pytest.approx(0.1173, abs=1e-3)


def test_one_sided_upper_bound_equals_two_sided_90_high():
    assert upper_bound(4, 24, confidence=0.95) == pytest.approx(
        clopper_pearson(4, 24, confidence=0.90).high
    )


def test_rule_of_three_needs_59_negatives_for_five_percent():
    assert upper_bound(0, 58) > 0.05
    assert upper_bound(0, 59) <= 0.05


# --- binomial_interval (rate helper with an undefined denominator) ----------------------------


def test_binomial_interval_returns_none_for_zero_denominator():
    assert binomial_interval(0, 0) is None


def test_binomial_interval_delegates_to_clopper_pearson():
    assert binomial_interval(4, 24) == clopper_pearson(4, 24)


# --- bootstrap_pr_auc ------------------------------------------------------------------------


def test_bootstrap_pr_auc_perfect_separation_is_degenerate_at_one():
    ci = bootstrap_pr_auc([1, 1, 0, 0], [0.9, 0.8, 0.2, 0.1], n_resamples=200)
    assert ci is not None
    assert ci.low == pytest.approx(1.0)
    assert ci.high == pytest.approx(1.0)
    assert ci.method == "bootstrap-percentile"


def test_bootstrap_pr_auc_is_deterministic_for_a_seed():
    y_true = [1, 0, 1, 0, 1, 0, 1, 0]
    y_scores = [0.9, 0.6, 0.55, 0.5, 0.45, 0.3, 0.7, 0.2]
    a = bootstrap_pr_auc(y_true, y_scores, n_resamples=300, seed=7)
    b = bootstrap_pr_auc(y_true, y_scores, n_resamples=300, seed=7)
    assert a == b


def test_bootstrap_pr_auc_interval_brackets_the_point_estimate():
    from qorgan.eval.metrics import pr_auc

    y_true = [1, 0, 1, 0, 1, 0, 1, 0, 0, 1]
    y_scores = [0.9, 0.6, 0.55, 0.5, 0.45, 0.3, 0.7, 0.2, 0.65, 0.35]
    ci = bootstrap_pr_auc(y_true, y_scores, n_resamples=500)
    assert ci is not None
    assert ci.low <= pr_auc(y_true, y_scores) <= ci.high


def test_bootstrap_pr_auc_returns_none_without_both_classes():
    assert bootstrap_pr_auc([1, 1, 1], [0.9, 0.8, 0.7]) is None
    assert bootstrap_pr_auc([], []) is None


def test_bootstrap_pr_auc_rejects_bad_arguments():
    with pytest.raises(ValueError):
        bootstrap_pr_auc([1, 0], [0.9, 0.1], n_resamples=0)
    with pytest.raises(ValueError):
        bootstrap_pr_auc([1, 0], [0.9])


# --- Interval value object -------------------------------------------------------------------


def test_interval_is_immutable():
    ci = Interval(low=0.1, high=0.2, confidence=0.95, method="clopper-pearson")
    with pytest.raises(Exception):
        ci.low = 0.3  # type: ignore[misc]


def test_interval_rejects_inverted_bounds():
    with pytest.raises(ValueError):
        Interval(low=0.5, high=0.4, confidence=0.95, method="clopper-pearson")


def test_interval_as_tuple():
    assert Interval(low=0.1, high=0.2, confidence=0.95, method="x").as_tuple() == (0.1, 0.2)
