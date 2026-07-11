"""TDD tests for `qorgan.eval.threshold` -- FPR-constrained alert-threshold selection.

Convention (matches `qorgan.eval.metrics`): positive class = scam = 1. FPR is the
PRIMARY constraint here (CLAUDE.md SS3.5) -- the demo must not fire on a real bank call.
"""

import pytest

from qorgan.eval.threshold import ThresholdChoice, select_threshold, sweep_thresholds

# --- sweep_thresholds ------------------------------------------------------------------


def test_sweep_thresholds_returns_exactly_steps_rows_ordered_ascending():
    y_true = [0, 0, 1, 1]
    y_scores = [0.1, 0.4, 0.6, 0.9]

    rows = sweep_thresholds(y_true, y_scores, steps=5)

    assert len(rows) == 5
    thresholds = [row["threshold"] for row in rows]
    assert thresholds == sorted(thresholds)
    assert thresholds == pytest.approx([0.0, 0.25, 0.5, 0.75, 1.0])


def test_sweep_thresholds_row_has_expected_keys():
    rows = sweep_thresholds([0, 1], [0.2, 0.8], steps=3)
    assert set(rows[0].keys()) == {"threshold", "fpr", "recall", "precision", "f1"}


def test_sweep_thresholds_default_steps_is_101():
    rows = sweep_thresholds([0, 1], [0.2, 0.8])
    assert len(rows) == 101


def test_sweep_thresholds_length_mismatch_raises():
    with pytest.raises(ValueError):
        sweep_thresholds([0, 1], [0.5])


def test_sweep_thresholds_empty_raises():
    with pytest.raises(ValueError):
        sweep_thresholds([], [])


@pytest.mark.parametrize("bad_steps", [0, 1, -5])
def test_sweep_thresholds_steps_less_than_2_raises(bad_steps):
    with pytest.raises(ValueError):
        sweep_thresholds([0, 1], [0.2, 0.8], steps=bad_steps)


# --- select_threshold --------------------------------------------------------------------


def test_select_threshold_perfectly_separable_scores_picks_gap_with_zero_fpr_full_recall():
    y_true = [0, 0, 0, 1, 1, 1]
    y_scores = [0.1, 0.2, 0.3, 0.7, 0.8, 0.9]

    choice = select_threshold(y_true, y_scores, max_fpr=0.0)

    assert isinstance(choice, ThresholdChoice)
    assert choice.fpr == pytest.approx(0.0)
    assert choice.recall == pytest.approx(1.0)
    assert choice.precision == pytest.approx(1.0)
    assert choice.f1 == pytest.approx(1.0)
    # tie-break "higher threshold" among the gap -> the highest threshold that still
    # keeps every positive (min positive score is 0.7).
    assert choice.threshold == pytest.approx(0.7)


def test_select_threshold_strict_max_fpr_forces_higher_threshold_and_lowers_recall():
    y_true = [0, 0, 1, 1, 1]
    y_scores = [0.5, 0.2, 0.3, 0.6, 0.9]

    unconstrained = select_threshold(y_true, y_scores, max_fpr=1.0, steps=11)
    constrained = select_threshold(y_true, y_scores, max_fpr=0.4, steps=11)

    assert unconstrained.threshold == pytest.approx(0.3)
    assert unconstrained.fpr == pytest.approx(0.5)
    assert unconstrained.recall == pytest.approx(1.0)

    assert constrained.threshold == pytest.approx(0.6)
    assert constrained.fpr == pytest.approx(0.0)
    assert constrained.recall == pytest.approx(2 / 3)
    assert constrained.recall < unconstrained.recall
    assert constrained.threshold > unconstrained.threshold


def test_select_threshold_no_threshold_meets_constraints_falls_back_to_min_fpr():
    # Overlapping scores (negative scores higher than positive) -- fpr and recall can
    # never both be good at once, so no threshold satisfies max_fpr AND min_recall.
    y_true = [1, 0]
    y_scores = [0.4, 0.6]

    choice = select_threshold(y_true, y_scores, max_fpr=0.0, min_recall=0.5, steps=3)

    # min fpr across the whole sweep is 0.0, uniquely at threshold=1.0.
    assert choice.threshold == pytest.approx(1.0)
    assert choice.fpr == pytest.approx(0.0)
    assert choice.recall == pytest.approx(0.0)


def test_select_threshold_length_mismatch_raises():
    with pytest.raises(ValueError):
        select_threshold([1, 0], [0.5], max_fpr=0.1)


def test_select_threshold_empty_raises():
    with pytest.raises(ValueError):
        select_threshold([], [], max_fpr=0.1)


@pytest.mark.parametrize("bad_max_fpr", [-0.01, 1.01, -1, 2])
def test_select_threshold_max_fpr_out_of_range_raises(bad_max_fpr):
    with pytest.raises(ValueError):
        select_threshold([1, 0], [0.9, 0.1], max_fpr=bad_max_fpr)


@pytest.mark.parametrize("bad_min_recall", [-0.01, 1.01, -1, 2])
def test_select_threshold_min_recall_out_of_range_raises(bad_min_recall):
    with pytest.raises(ValueError):
        select_threshold([1, 0], [0.9, 0.1], max_fpr=0.1, min_recall=bad_min_recall)


def test_select_threshold_steps_less_than_2_raises():
    with pytest.raises(ValueError):
        select_threshold([1, 0], [0.9, 0.1], max_fpr=0.1, steps=1)
