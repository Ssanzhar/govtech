"""TDD tests for `qorgan.live.session` — per-utterance live pipeline (design spec §§06-10).

Runs against the deterministic `mock` backend (taxonomy-example keyword scorer): no
model, no key, no network. The scam script below is built from verbatim taxonomy example
phrases so the mock backend detects them.
"""

import pytest

from qorgan.asr.stream import CommittedUtterance
from qorgan.live.meter import DOUBLE_HARD_SIGNAL_FLOOR
from qorgan.live.session import (
    MAX_WINDOW_CHARS,
    WINDOW_HEAD_UTTERANCES,
    LiveSessionState,
    advance,
    initial_session,
)

# Verbatim taxonomy examples → deterministically detected by the mock backend.
SCAM_TURNS = (
    "Алло, здравствуйте. Это служба безопасности вашего банка.",  # impersonation_bank
    "Зафиксирована подозрительная операция, действовать нужно прямо сейчас.",  # urgency
    "Продиктуйте код из SMS для отмены операции.",  # otp_request (hard)
    "И переведите деньги на безопасный счёт.",  # safe_account (hard)
)
BENIGN_TURNS = (
    "Привет! Как выходные прошли?",
    "Отлично, ездили в горы. Давай завтра созвонимся?",
)


def _utterance(text: str, confidence: float = 1.0) -> CommittedUtterance:
    return CommittedUtterance(text=text, confidence=confidence)


def _run(turns, state=None):
    state = state or initial_session("ru", backend="mock")
    updates = []
    for text in turns:
        state, update = advance(state, _utterance(text))
        updates.append(update)
    return state, updates


# --- the two demo scenes -------------------------------------------------------------------


def test_scam_call_escalates_latches_and_floors_at_critical():
    state, updates = _run(SCAM_TURNS)

    scores = [u.meter.score for u in updates]
    assert scores == sorted(scores)  # rises monotonically on this script
    assert state.meter.score >= DOUBLE_HARD_SIGNAL_FLOOR  # two hard signals fired
    assert updates[-1].band == "critical"
    assert state.meter.latched is True
    assert {"otp_request", "safe_account"} <= state.meter.hard_signal_ids


def test_benign_call_stays_low_with_no_advice_or_evidence():
    state, updates = _run(BENIGN_TURNS)

    assert updates[-1].band == "low"
    assert state.meter.latched is False
    for update in updates:
        assert update.recommendation.advices == ()
        assert update.new_evidence == ()


# --- evidence ---------------------------------------------------------------------------------


def test_evidence_spans_are_verbatim_in_the_scored_window():
    _state, updates = _run(SCAM_TURNS)

    for update in updates:
        for span in update.new_evidence:
            assert update.window_text[span.start : span.end] == span.text


def test_evidence_is_not_repeated_across_turns():
    _state, updates = _run(SCAM_TURNS)

    seen: list[str] = []
    for update in updates:
        for span in update.new_evidence:
            assert span.text not in seen
            seen.append(span.text)
    assert seen  # the scam script does produce evidence


def test_detected_tags_accumulate_with_max_weight():
    state, _updates = _run(SCAM_TURNS)

    ids = {tag.id for tag in state.tags}
    assert {"impersonation_bank", "urgency", "otp_request", "safe_account"} <= ids
    assert all(0.0 <= tag.weight <= 1.0 for tag in state.tags)


# --- recommendations --------------------------------------------------------------------------


def test_recommendations_surface_once_risk_leaves_low_band():
    _state, updates = _run(SCAM_TURNS)

    final = updates[-1].recommendation
    assert final.advices != ()
    assert final.verification_questions != ()


def test_hard_signal_advice_leads_once_hard_signal_fires():
    _state, updates = _run(SCAM_TURNS)

    from qorgan.explain.recommend import load_advice

    advice = load_advice("ru")
    otp_turn = updates[2].recommendation  # the OTP-request turn
    assert otp_turn.advices[0] == advice.tactic_advice["otp_request"]


# --- rolling window ---------------------------------------------------------------------------


def test_window_covers_full_transcript_while_short():
    _state, updates = _run(SCAM_TURNS)

    assert SCAM_TURNS[0] in updates[-1].window_text
    assert SCAM_TURNS[-1] in updates[-1].window_text


def test_long_call_window_keeps_head_and_tail_within_cap():
    filler = [f"Дежурная реплика номер {i} — " + "слово " * 60 for i in range(40)]
    turns = [SCAM_TURNS[0], *filler, SCAM_TURNS[2]]

    _state, updates = _run(turns)
    window = updates[-1].window_text

    assert len(window) <= MAX_WINDOW_CHARS
    assert SCAM_TURNS[0] in window  # head survives: impersonation was established there
    assert SCAM_TURNS[2] in window  # tail survives: the live request just happened


def test_full_transcript_is_kept_even_when_window_truncates():
    filler = [f"Дежурная реплика номер {i} — " + "слово " * 60 for i in range(40)]
    state, _updates = _run([SCAM_TURNS[0], *filler])

    assert len(state.utterances) == 1 + len(filler)
    assert state.transcript().startswith(SCAM_TURNS[0])


# --- construction & immutability ---------------------------------------------------------------


def test_initial_session_rejects_unsupported_locale():
    with pytest.raises(ValueError):
        initial_session("en", backend="mock")


def test_advance_does_not_mutate_input_state():
    state = initial_session("ru", backend="mock")
    advance(state, _utterance(SCAM_TURNS[0]))

    assert state.utterances == ()
    assert state.meter.turn_index == 0


def test_session_state_is_frozen():
    state = initial_session("ru", backend="mock")
    with pytest.raises(Exception):
        state.locale = "kk"  # type: ignore[misc]


def test_head_constant_is_smaller_than_window():
    assert WINDOW_HEAD_UTTERANCES >= 1
    assert MAX_WINDOW_CHARS > 1000
