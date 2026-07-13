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
