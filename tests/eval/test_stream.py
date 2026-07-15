"""TDD tests for `qorgan.eval.stream` — the streaming (per-turn) eval harness.

Deterministic via the `mock` backend (taxonomy-example keyword scorer, same fixture as
`tests/live/test_session.py`): no model, no key, no network. The scam script is built
from verbatim taxonomy example phrases so the mock backend detects them turn by turn and
the meter escalates to a latch; the benign script never latches.
"""

import pytest

from qorgan.asr.stream import CommittedUtterance
from qorgan.data.generate import write_dialogues_jsonl
from qorgan.data.schema import Dialogue, Label, Utterance
from qorgan.eval.stream import (
    StreamReport,
    StreamResult,
    _percentile,
    evaluate_stream,
    format_stream_report,
    main,
    replay_dialogue,
)
from qorgan.live.session import advance, initial_session

# Verbatim taxonomy examples (mirrors tests/live/test_session.py) — deterministically
# detected by the mock backend: impersonation_bank, urgency, otp_request (hard),
# safe_account (hard). Two hard signals -> the meter floors at Critical and latches.
SCAM_TURNS = (
    "Алло, здравствуйте. Это служба безопасности вашего банка.",
    "Зафиксирована подозрительная операция, действовать нужно прямо сейчас.",
    "Продиктуйте код из SMS для отмены операции.",
    "И переведите деньги на безопасный счёт.",
)
BENIGN_TURNS = (
    "Привет! Как выходные прошли?",
    "Отлично, ездили в горы. Давай завтра созвонимся?",
)


def _dialogue(dialogue_id: str, turns: tuple[str, ...], *, risk: float) -> Dialogue:
    utterances = tuple(Utterance(speaker="caller", text=text) for text in turns)
    return Dialogue(id=dialogue_id, language="ru", utterances=utterances, label=Label(risk=risk))


def _scam_dialogue(dialogue_id: str = "scam1") -> Dialogue:
    return _dialogue(dialogue_id, SCAM_TURNS, risk=0.9)


def _benign_dialogue(dialogue_id: str = "benign1") -> Dialogue:
    return _dialogue(dialogue_id, BENIGN_TURNS, risk=0.02)


def _first_latch_turn(turns: tuple[str, ...]) -> int | None:
    """Reference computation against the raw session, to cross-check `replay_dialogue`."""
    state = initial_session("ru", backend="mock")
    first: int | None = None
    for index, text in enumerate(turns, start=1):
        state, update = advance(state, CommittedUtterance(text=text, confidence=1.0))
        if update.meter.latched and first is None:
            first = index
    return first


# --- replay_dialogue --------------------------------------------------------------------


def test_scam_dialogue_latches_with_correct_turns_to_alert():
    result = replay_dialogue(_scam_dialogue(), locale="ru", backend="mock")

    assert isinstance(result, StreamResult)
    assert result.is_scam is True
    assert result.latched is True
    assert result.turns_to_alert == _first_latch_turn(SCAM_TURNS)
    assert result.final_band == "critical"
    assert result.max_score > 0.0


def test_benign_dialogue_never_latches():
    result = replay_dialogue(_benign_dialogue(), locale="ru", backend="mock")

    assert result.is_scam is False
    assert result.latched is False
    assert result.turns_to_alert is None
    assert result.final_band == "low"


def test_dialogue_id_is_preserved():
    result = replay_dialogue(_scam_dialogue("my-id-42"), locale="ru", backend="mock")
    assert result.dialogue_id == "my-id-42"


def test_is_scam_mirrors_eval_run_truth_threshold():
    from qorgan.eval.run import _TRUTH_THRESHOLD

    just_below = _dialogue("d1", SCAM_TURNS, risk=_TRUTH_THRESHOLD - 0.01)
    just_at = _dialogue("d2", SCAM_TURNS, risk=_TRUTH_THRESHOLD)

    assert replay_dialogue(just_below, locale="ru", backend="mock").is_scam is False
    assert replay_dialogue(just_at, locale="ru", backend="mock").is_scam is True


def test_replay_dialogue_raises_for_unsupported_locale():
    with pytest.raises(ValueError):
        replay_dialogue(_scam_dialogue(), locale="en", backend="mock")


def test_replay_dialogue_raises_clearly_for_unsupported_backend():
    with pytest.raises(Exception, match="backend"):
        replay_dialogue(_scam_dialogue(), locale="ru", backend="not-a-real-backend")


# --- evaluate_stream ---------------------------------------------------------------------


def test_evaluate_stream_mixed_set_aggregates_correctly():
    dialogues = [
        _scam_dialogue("scam1"),
        _scam_dialogue("scam2"),
        _benign_dialogue("benign1"),
        _benign_dialogue("benign2"),
    ]
    report = evaluate_stream(dialogues, locale="ru", backend="mock")

    assert isinstance(report, StreamReport)
    assert report.false_latch_rate == 0.0
    assert report.alert_hit_rate == 1.0
    assert report.n_positive == 2
    assert report.n_negative == 2
    expected_turn = float(_first_latch_turn(SCAM_TURNS))
    assert report.median_turns_to_alert == pytest.approx(expected_turn)
    assert report.p90_turns_to_alert == pytest.approx(expected_turn)


def test_evaluate_stream_counts_a_false_latch():
    """A dialogue labeled negative whose transcript still contains scam phrases (e.g. a
    mislabeled or borderline hard-negative) should count toward false_latch_rate."""
    false_latch = _dialogue("fl1", SCAM_TURNS, risk=0.02)
    dialogues = [false_latch, _benign_dialogue("benign1")]

    report = evaluate_stream(dialogues, locale="ru", backend="mock")

    assert report.n_negative == 2
    assert report.false_latch_rate == pytest.approx(0.5)


def test_evaluate_stream_all_negative_has_zero_false_latch_rate_and_none_percentiles():
    dialogues = [_benign_dialogue("b1"), _benign_dialogue("b2")]
    report = evaluate_stream(dialogues, locale="ru", backend="mock")

    assert report.false_latch_rate == 0.0
    assert report.alert_hit_rate == 0.0
    assert report.n_positive == 0
    assert report.n_negative == 2
    assert report.median_turns_to_alert is None
    assert report.p90_turns_to_alert is None


def test_evaluate_stream_single_positive_percentiles_equal_its_turn():
    report = evaluate_stream([_scam_dialogue("solo")], locale="ru", backend="mock")

    expected = float(_first_latch_turn(SCAM_TURNS))
    assert report.n_positive == 1
    assert report.median_turns_to_alert == pytest.approx(expected)
    assert report.p90_turns_to_alert == pytest.approx(expected)


def test_evaluate_stream_rejects_empty_dialogue_list():
    with pytest.raises(ValueError):
        evaluate_stream([], locale="ru", backend="mock")


# --- _percentile (pure helper) -------------------------------------------------------------


def test_percentile_single_value_returns_that_value_at_any_quantile():
    assert _percentile([7], 0.5) == 7.0
    assert _percentile([7], 0.9) == 7.0
    assert _percentile([7], 0.0) == 7.0


def test_percentile_median_is_linear_interpolation_of_middle_two():
    assert _percentile([1, 2, 3, 4], 0.5) == pytest.approx(2.5)


def test_percentile_p90_known_value():
    assert _percentile(list(range(1, 11)), 0.9) == pytest.approx(9.1)


def test_percentile_raises_for_empty_sequence():
    with pytest.raises(ValueError):
        _percentile([], 0.5)


# --- format_stream_report / CLI --------------------------------------------------------------


def test_format_stream_report_places_false_latch_rate_first():
    report = evaluate_stream([_scam_dialogue(), _benign_dialogue()], locale="ru", backend="mock")
    text = format_stream_report({"test": report})

    header_line = text.splitlines()[0]
    assert header_line.lower().index("false") < header_line.lower().index("alert")
    assert "test" in text
    assert "|" in text


def test_format_stream_report_renders_dash_for_none_percentiles():
    report = evaluate_stream([_benign_dialogue("b1"), _benign_dialogue("b2")], locale="ru", backend="mock")
    text = format_stream_report({"real_heldout": report})
    assert "-" in text


def test_main_cli_prints_report_with_false_latch_rate_first(tmp_path, capsys):
    dialogues = [_scam_dialogue("s1"), _benign_dialogue("n1")]
    write_dialogues_jsonl(dialogues, tmp_path / "test.jsonl")

    main(["--split", "test", "--backend", "mock", "--processed-dir", str(tmp_path), "--locale", "ru"])

    out = capsys.readouterr().out
    assert "test" in out
    header_line = out.splitlines()[0]
    assert header_line.lower().index("false") < header_line.lower().index("alert")


def test_main_cli_raises_clearly_for_missing_split(tmp_path):
    with pytest.raises(FileNotFoundError):
        main(["--split", "nope", "--backend", "mock", "--processed-dir", str(tmp_path)])


# --- frozen models -----------------------------------------------------------------------------


def test_stream_result_is_frozen():
    result = replay_dialogue(_scam_dialogue(), locale="ru", backend="mock")
    with pytest.raises(Exception):
        result.latched = False  # type: ignore[misc]


def test_stream_report_is_frozen():
    report = evaluate_stream([_scam_dialogue()], locale="ru", backend="mock")
    with pytest.raises(Exception):
        report.false_latch_rate = 1.0  # type: ignore[misc]
