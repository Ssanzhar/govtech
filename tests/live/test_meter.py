"""TDD tests for `qorgan.live.meter` — the 0-100 suspicion meter (design spec §08).

Methodology under test: asymmetric EMA (rises fast, decays slowly), ASR-confidence
weighting, hard-signal floors (one distinct hard signal → 61 / High, two → 81 / Critical,
never lowering), 55/45 hysteresis latch, 30/60/80 display bands, and an evidence ledger
so the meter can always answer "why did you just go up?".
"""

import pytest

from qorgan.live.meter import (
    BAND_HIGH_MAX,
    BAND_LOW_MAX,
    BAND_MEDIUM_MAX,
    DOUBLE_HARD_SIGNAL_FLOOR,
    HARD_SIGNAL_CONFIDENCE_FLOOR,
    SINGLE_HARD_SIGNAL_FLOOR,
    MeterState,
    band,
    initial_state,
    update,
)

# With the default alphas (up 0.5, down 0.12) and latch (enter 55, exit 45):
# risk 1.0 from 0 → 50 → 75 → ...; decay from 75 → 66 → 58.08 → 51.11 → 44.98.


# --- initial state -----------------------------------------------------------------------


def test_initial_state_is_calm():
    state = initial_state()

    assert state.score == 0.0
    assert state.latched is False
    assert state.turn_index == 0
    assert state.hard_signal_ids == frozenset()
    assert state.ledger == ()
    assert band(state.score) == "low"


# --- asymmetric EMA ----------------------------------------------------------------------


def test_single_high_risk_turn_rises_gradually_not_instantly():
    state = update(initial_state(), risk=1.0)

    assert state.score == pytest.approx(50.0)
    assert band(state.score) == "medium"
    assert state.latched is False  # 50 < enter threshold 55


def test_two_consistent_high_risk_turns_reach_high_band_and_latch():
    state = update(update(initial_state(), risk=1.0), risk=1.0)

    assert state.score == pytest.approx(75.0)
    assert band(state.score) == "high"
    assert state.latched is True


def test_decay_is_slower_than_rise():
    risen = update(update(initial_state(), risk=1.0), risk=1.0)  # 75
    decayed = update(risen, risk=0.0)

    assert decayed.score == pytest.approx(66.0)  # 75 - 0.12·75, not 75 - 0.5·75
    assert decayed.latched is True  # still above exit


def test_unlatching_requires_sustained_calm():
    state = update(update(initial_state(), risk=1.0), risk=1.0)  # 75, latched

    for _ in range(3):  # 66 → 58.08 → 51.11: all above exit 45
        state = update(state, risk=0.0)
        assert state.latched is True

    state = update(state, risk=0.0)  # 44.98 ≤ 45 → releases
    assert state.latched is False


def test_score_never_exceeds_100_or_drops_below_0():
    state = initial_state()
    for _ in range(50):
        state = update(state, risk=1.0)
    assert state.score <= 100.0

    for _ in range(200):
        state = update(state, risk=0.0)
    assert state.score >= 0.0


# --- ASR-confidence weighting -------------------------------------------------------------


def test_zero_asr_confidence_does_not_move_the_needle():
    state = update(initial_state(), risk=1.0, asr_confidence=0.0)

    assert state.score == 0.0


def test_half_asr_confidence_halves_the_step():
    state = update(initial_state(), risk=1.0, asr_confidence=0.5)

    assert state.score == pytest.approx(25.0)


# --- hard-signal floors --------------------------------------------------------------------


def test_one_confident_hard_signal_floors_score_at_high_band():
    state = update(initial_state(), risk=0.2, hard_signals={"otp_request": 0.9})

    assert state.score == pytest.approx(SINGLE_HARD_SIGNAL_FLOOR)
    assert band(state.score) == "high"
    assert state.latched is True  # 61 ≥ enter 55


def test_hard_signal_below_confidence_floor_is_ignored():
    state = update(
        initial_state(),
        risk=0.2,
        hard_signals={"otp_request": HARD_SIGNAL_CONFIDENCE_FLOOR - 0.1},
    )

    assert state.score < SINGLE_HARD_SIGNAL_FLOOR
    assert state.hard_signal_ids == frozenset()


def test_two_distinct_hard_signals_floor_score_at_critical():
    state = update(
        initial_state(), risk=0.2, hard_signals={"otp_request": 0.9, "safe_account": 0.95}
    )

    assert state.score == pytest.approx(DOUBLE_HARD_SIGNAL_FLOOR)
    assert band(state.score) == "critical"


def test_same_hard_signal_repeated_does_not_count_twice():
    first = update(initial_state(), risk=0.2, hard_signals={"otp_request": 0.9})
    second = update(first, risk=0.2, hard_signals={"otp_request": 0.9})

    assert second.score < DOUBLE_HARD_SIGNAL_FLOOR
    assert second.hard_signal_ids == frozenset({"otp_request"})


def test_hard_signals_accumulate_across_turns():
    first = update(initial_state(), risk=0.2, hard_signals={"otp_request": 0.9})
    second = update(first, risk=0.2, hard_signals={"secrecy": 0.85})

    assert second.score == pytest.approx(DOUBLE_HARD_SIGNAL_FLOOR)
    assert second.hard_signal_ids == frozenset({"otp_request", "secrecy"})


def test_floor_never_lowers_an_already_higher_score():
    state = initial_state()
    for _ in range(6):
        state = update(state, risk=1.0)  # well above 81
    high_score = state.score

    floored = update(state, risk=1.0, hard_signals={"otp_request": 0.9})
    assert floored.score >= high_score


# --- bands ---------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("score", "expected"),
    [
        (0.0, "low"),
        (BAND_LOW_MAX, "low"),
        (BAND_LOW_MAX + 0.5, "medium"),
        (BAND_MEDIUM_MAX, "medium"),
        (BAND_MEDIUM_MAX + 0.5, "high"),
        (BAND_HIGH_MAX, "high"),
        (BAND_HIGH_MAX + 0.5, "critical"),
        (100.0, "critical"),
    ],
)
def test_band_edges(score, expected):
    assert band(score) == expected


def test_band_rejects_out_of_range_scores():
    with pytest.raises(ValueError):
        band(-1.0)
    with pytest.raises(ValueError):
        band(100.5)


# --- ledger & immutability -------------------------------------------------------------------


def test_ledger_records_every_update_with_before_and_after():
    state = update(update(initial_state(), risk=1.0), risk=0.0, hard_signals={"secrecy": 0.9})

    assert len(state.ledger) == 2
    first, second = state.ledger
    assert first.turn_index == 1
    assert first.score_before == 0.0
    assert first.score_after == pytest.approx(50.0)
    assert second.turn_index == 2
    assert second.new_hard_signal_ids == ("secrecy",)


def test_update_does_not_mutate_the_input_state():
    state = initial_state()
    update(state, risk=1.0)

    assert state.score == 0.0
    assert state.ledger == ()


# --- input validation ------------------------------------------------------------------------


def test_update_rejects_out_of_range_risk():
    with pytest.raises(ValueError):
        update(initial_state(), risk=1.5)


def test_update_rejects_out_of_range_asr_confidence():
    with pytest.raises(ValueError):
        update(initial_state(), risk=0.5, asr_confidence=-0.1)


def test_meter_state_is_frozen():
    state = initial_state()
    with pytest.raises(Exception):
        state.score = 99.0  # type: ignore[misc]


def test_latch_uses_configured_hysteresis_thresholds(monkeypatch):
    monkeypatch.setenv("QORGAN_RISK_THRESHOLD_ENTER", "0.90")
    monkeypatch.setenv("QORGAN_RISK_THRESHOLD_EXIT", "0.10")

    state = update(update(initial_state(), risk=1.0), risk=1.0)  # 75 < 90

    assert state.latched is False


def test_smoothing_uses_configured_alphas(monkeypatch):
    monkeypatch.setenv("QORGAN_METER_ALPHA_UP", "1.0")

    state = update(initial_state(), risk=0.7)

    assert state.score == pytest.approx(70.0)


def test_hard_negative_style_call_stays_low():
    """A calm, legit call (low risk every turn, no hard signals) never leaves Low."""
    state = initial_state()
    for _ in range(10):
        state = update(state, risk=0.05)

    assert band(state.score) == "low"
    assert state.latched is False
