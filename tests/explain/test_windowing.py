"""TDD tests for `qorgan.explain.windowing` — cumulative windows + hysteresis (gap G8)."""

import pytest

from qorgan.explain.windowing import Window, apply_hysteresis, windows

# --- windows() -------------------------------------------------------------------------


def test_windows_basic_cumulative_join():
    result = windows(["a", "b", "c"])

    assert [w.text for w in result] == ["a", "a\nb", "a\nb\nc"]


def test_windows_end_index_matches_cumulative_length():
    result = windows(["a", "b", "c"])

    assert [w.end_index for w in result] == [1, 2, 3]


def test_windows_single_utterance():
    result = windows(["only one"])

    assert len(result) == 1
    assert result[0].text == "only one"
    assert result[0].end_index == 1


def test_windows_empty_utterances_returns_empty_list():
    assert windows([]) == []


def test_windows_step_parameter_skips_intermediate_utterances():
    result = windows(["a", "b", "c", "d"], step=2)

    assert [w.end_index for w in result] == [2, 4]
    assert [w.text for w in result] == ["a\nb", "a\nb\nc\nd"]


def test_windows_step_parameter_includes_final_partial_window():
    result = windows(["a", "b", "c"], step=2)

    assert [w.end_index for w in result] == [2, 3]


def test_windows_invalid_step_raises():
    with pytest.raises(ValueError):
        windows(["a", "b"], step=0)


def test_windows_returns_window_dataclass_instances():
    result = windows(["a"])

    assert isinstance(result[0], Window)


# --- apply_hysteresis() -----------------------------------------------------------------


def test_apply_hysteresis_enters_when_score_crosses_enter_threshold():
    scores = [0.1, 0.2, 0.9, 0.9]
    state = apply_hysteresis(scores, enter=0.7, exit=0.4)

    assert state == [False, False, True, True]


def test_apply_hysteresis_stays_on_between_exit_and_enter_after_entering():
    scores = [0.9, 0.5, 0.9]
    state = apply_hysteresis(scores, enter=0.7, exit=0.4)

    # 0.5 is below enter but above exit -- must stay ON once entered.
    assert state == [True, True, True]


def test_apply_hysteresis_exits_when_score_drops_below_exit():
    scores = [0.9, 0.3, 0.9]
    state = apply_hysteresis(scores, enter=0.7, exit=0.4)

    assert state == [True, False, True]


def test_apply_hysteresis_never_enters_if_always_below_enter():
    scores = [0.1, 0.5, 0.6]
    state = apply_hysteresis(scores, enter=0.7, exit=0.4)

    assert state == [False, False, False]


def test_apply_hysteresis_empty_scores_returns_empty_list():
    assert apply_hysteresis([], enter=0.7, exit=0.4) == []


def test_apply_hysteresis_returns_list_of_bool_same_length_as_scores():
    scores = [0.1, 0.9, 0.2]
    state = apply_hysteresis(scores, enter=0.7, exit=0.4)

    assert len(state) == len(scores)
    assert all(isinstance(value, bool) for value in state)


def test_apply_hysteresis_enter_below_exit_raises_value_error():
    with pytest.raises(ValueError):
        apply_hysteresis([0.5], enter=0.4, exit=0.7)


def test_apply_hysteresis_enter_equal_exit_is_allowed():
    scores = [0.6, 0.4, 0.6]
    state = apply_hysteresis(scores, enter=0.5, exit=0.5)

    assert state == [True, False, True]
