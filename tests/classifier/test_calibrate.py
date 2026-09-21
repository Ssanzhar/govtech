"""TDD tests for `qorgan.classifier.calibrate` -- pure temperature-scaling application,
NLL, temperature fitting, and persistence.
"""

import math

import pytest

from qorgan.classifier.calibrate import (
    apply_temperature,
    fit_temperature,
    load_temperature,
    negative_log_likelihood,
    save_temperature,
)

# --- apply_temperature ---------------------------------------------------------------------


def test_apply_temperature_one_is_plain_sigmoid():
    probs = apply_temperature([0.0], temperature=1.0)
    assert probs == [pytest.approx(0.5)]


def test_apply_temperature_one_matches_manual_sigmoid_values():
    logits = [-2.0, 0.0, 2.0]
    probs = apply_temperature(logits, temperature=1.0)
    expected = [1 / (1 + math.exp(-logit)) for logit in logits]
    assert probs == pytest.approx(expected)


def test_apply_temperature_higher_temperature_pulls_toward_half():
    logit = [4.0]
    low_temp = apply_temperature(logit, temperature=1.0)[0]
    high_temp = apply_temperature(logit, temperature=4.0)[0]
    # both above 0.5 (positive logit) but high_temp is closer to 0.5
    assert 0.5 < high_temp < low_temp


def test_apply_temperature_negative_logit_higher_temperature_closer_to_half():
    logit = [-4.0]
    low_temp = apply_temperature(logit, temperature=1.0)[0]
    high_temp = apply_temperature(logit, temperature=4.0)[0]
    assert low_temp < high_temp < 0.5


@pytest.mark.parametrize("bad_temperature", [0.0, -1.0, -0.001])
def test_apply_temperature_non_positive_temperature_raises(bad_temperature):
    with pytest.raises(ValueError):
        apply_temperature([0.0], temperature=bad_temperature)


def test_apply_temperature_large_positive_logit_near_one_no_overflow():
    probs = apply_temperature([1000.0], temperature=1.0)
    assert probs[0] == pytest.approx(1.0, abs=1e-9)


def test_apply_temperature_large_negative_logit_near_zero_no_overflow():
    probs = apply_temperature([-1000.0], temperature=1.0)
    assert probs[0] == pytest.approx(0.0, abs=1e-9)


def test_apply_temperature_empty_input_returns_empty_list():
    assert apply_temperature([], temperature=1.0) == []


def test_apply_temperature_returns_list_of_floats():
    probs = apply_temperature([0.5, -0.5], temperature=2.0)
    assert isinstance(probs, list)
    assert all(isinstance(value, float) for value in probs)
    assert all(0.0 <= value <= 1.0 for value in probs)


# --- negative_log_likelihood ---------------------------------------------------------------


def test_nll_lower_when_probs_match_targets():
    logits = [5.0, -5.0, 5.0, -5.0]
    targets = [1, 0, 1, 0]
    good = negative_log_likelihood(logits, targets, temperature=1.0)
    bad = negative_log_likelihood(logits, [0, 1, 0, 1], temperature=1.0)
    assert good < bad


def test_nll_length_mismatch_raises():
    with pytest.raises(ValueError):
        negative_log_likelihood([1.0, 2.0], [1], temperature=1.0)


# --- fit_temperature -----------------------------------------------------------------------


def test_fit_temperature_returns_positive_and_not_worse_than_unit():
    # Overconfident logits with some errors -> a temperature > 1 should not increase NLL.
    logits = [8.0, 7.0, -6.0, 9.0, -8.0, 6.0]
    targets = [1, 0, 0, 1, 0, 1]
    t = fit_temperature(logits, targets)
    assert t > 0
    assert negative_log_likelihood(logits, targets, t) <= negative_log_likelihood(logits, targets, 1.0) + 1e-9


def test_fit_temperature_softens_overconfident_model():
    # A confidently-wrong example should push temperature above 1 (soften probabilities).
    logits = [10.0, -10.0, 10.0, 10.0]
    targets = [1, 0, 1, 0]  # last one confidently wrong
    assert fit_temperature(logits, targets) > 1.0


def test_fit_temperature_empty_raises():
    with pytest.raises(ValueError):
        fit_temperature([], [])


def test_fit_temperature_non_binary_targets_raises():
    with pytest.raises(ValueError):
        fit_temperature([1.0, 2.0], [1, 2])


# --- save / load ---------------------------------------------------------------------------


def test_save_and_load_temperature_round_trips(tmp_path):
    path = tmp_path / "calib" / "temperature.json"
    save_temperature(1.734, path)
    assert path.exists()
    assert load_temperature(path) == pytest.approx(1.734)


def test_load_temperature_missing_defaults_to_one(tmp_path):
    assert load_temperature(tmp_path / "absent.json") == 1.0


# --- per-tactic thresholds (ADR D30) ------------------------------------------------------------


def test_tune_tactic_thresholds_picks_the_f1_maximising_threshold_per_tactic():
    from qorgan.classifier.calibrate import tune_tactic_thresholds

    label_space = ["otp_request", "urgency", "rare"]
    # otp_request: eight true rows score >= 0.75, four false rows sit at 0.5-0.65 -> 0.7 is the lowest perfect cut
    # urgency: nothing beats 0.5 (perfect at 0.5) -> stays 0.5 · rare: 2 positives (< 8) -> default
    probs = [[0.95, 0.9, 0.9], [0.92, 0.1, 0.9], [0.9, 0.9, 0.1], [0.88, 0.1, 0.1], [0.85, 0.9, 0.1], [0.8, 0.1, 0.1], [0.78, 0.9, 0.1], [0.75, 0.1, 0.1],
             [0.65, 0.9, 0.1], [0.6, 0.1, 0.1], [0.55, 0.9, 0.1], [0.5, 0.1, 0.1]]
    truth = [[1, 1, 1], [1, 0, 1], [1, 1, 0], [1, 0, 0], [1, 1, 0], [1, 0, 0], [1, 1, 0], [1, 0, 0],
             [0, 1, 0], [0, 0, 0], [0, 1, 0], [0, 0, 0]]
    chosen = tune_tactic_thresholds(probs, truth, label_space)
    assert chosen == {"otp_request": 0.7, "urgency": 0.5, "rare": 0.5}
    with pytest.raises(ValueError):
        tune_tactic_thresholds(probs, truth[:-1], label_space)


def test_tune_tactic_thresholds_never_exceeds_the_hard_signal_floor():
    from qorgan.classifier.calibrate import TACTIC_THRESHOLD_GRID
    from qorgan.live.meter import HARD_SIGNAL_CONFIDENCE_FLOOR

    assert max(TACTIC_THRESHOLD_GRID) < HARD_SIGNAL_CONFIDENCE_FLOOR  # a confident hard signal always survives


def test_tune_tactic_thresholds_prefers_the_lowest_threshold_on_ties():
    from qorgan.classifier.calibrate import tune_tactic_thresholds

    # every threshold from 0.5 to 0.7 gives the same perfect F1 -> keep 0.5 (recall)
    probs = [[0.95]] * 8 + [[0.1]] * 3
    truth = [[1]] * 8 + [[0]] * 3
    assert tune_tactic_thresholds(probs, truth, ["x"]) == {"x": 0.5}
