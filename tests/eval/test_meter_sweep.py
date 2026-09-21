"""TDD tests for `qorgan.eval.meter_sweep` (PLAN A6 methodology): per-turn traces cached
once, meter variants simulated through the production `meter.update` with an injected
config, and a false-latch-first table."""

from __future__ import annotations

import json

from qorgan.config import load_config
from qorgan.data.schema import ScoreResult, TacticTag
from qorgan.eval.meter_sweep import (
    MeterVariant,
    TurnTrace,
    load_traces,
    render_sweep,
    save_traces,
    simulate,
    sweep,
    trace_dialogue,
)

CFG = load_config({"QORGAN_METER_MIN_TURNS_TO_ARM": "3", "QORGAN_METER_SHORT_WINDOW_TURNS": "1"})


def _trace(dialogue_id, positive, risks, hard=None):
    return TurnTrace(dialogue_id=dialogue_id, split="test", positive=positive, turns=tuple(
        {"risk": r, "hard": (hard or {}).get(i, {})} for i, r in enumerate(risks, 1)
    ))


def test_trace_dialogue_records_risk_and_hard_signals_per_turn():
    calls = []

    def score_fn(text):
        calls.append(text)
        return ScoreResult(risk=0.9, tags=(TacticTag(id="otp_request", weight=0.95), TacticTag(id="urgency", weight=0.7)), backend="fake")

    trace = trace_dialogue("d1", "test", True, ["алло", "назовите код"], score_fn=score_fn, hard_signal_ids={"otp_request"})
    assert trace.dialogue_id == "d1" and trace.positive and len(trace.turns) == 2
    assert trace.turns[1]["risk"] == 0.9 and trace.turns[1]["hard"] == {"otp_request": 0.95}
    assert len(calls) == 2 and "назовите код" in calls[1]  # the rolling window grows per turn


def test_simulate_uses_the_production_meter_with_the_variant_applied():
    legit_opener = _trace("legit", False, [0.91, 0.75, 0.35])
    scam = _trace("scam", True, [0.85, 0.9, 0.9])
    shipped = MeterVariant(min_turns_to_arm=1)
    a6 = MeterVariant(min_turns_to_arm=3)
    assert simulate(legit_opener, shipped, CFG) == 2  # the turn-2 transient latch
    assert simulate(legit_opener, a6, CFG) is None
    assert simulate(scam, shipped, CFG) == 2 and simulate(scam, a6, CFG) == 3
    hard_on_turn_one = _trace("hard", True, [0.3, 0.3], hard={1: {"otp_request": 0.95}})
    assert simulate(hard_on_turn_one, a6, CFG) == 1  # a confident hard signal arms at once
    damped = MeterVariant(min_turns_to_arm=1, short_window_turns=2)
    assert simulate(_trace("short", True, [0.84, 0.68, 0.64]), damped, CFG) is None  # the 3-turn scam that damping loses


def test_sweep_reports_false_latch_first_and_turns_to_alert():
    traces = [
        _trace("legit", False, [0.91, 0.75, 0.35]),
        _trace("legit2", False, [0.1, 0.2, 0.1]),
        _trace("scam", True, [0.85, 0.9, 0.9]),
        _trace("scam2", True, [0.95, 0.95]),
    ]
    rows = sweep(traces, [MeterVariant(min_turns_to_arm=1), MeterVariant(min_turns_to_arm=3)], CFG)
    assert [r.variant.min_turns_to_arm for r in rows] == [1, 3]
    assert rows[0].by_split["test"].false_latches == 1 and rows[0].by_split["test"].alert_hits == 2
    assert rows[1].by_split["test"].false_latches == 0 and rows[1].by_split["test"].alert_hits == 1  # the 2-turn scam cannot arm
    assert rows[1].by_split["test"].median_turns_to_alert == 3
    table = render_sweep(rows)
    assert "| variant |" in table and "test false-latch" in table and "1/2" in table


def test_traces_round_trip_through_the_cache(tmp_path):
    traces = [_trace("a", True, [0.5, 0.6]), _trace("b", False, [0.1])]
    path = tmp_path / "traces.json"
    save_traces(traces, path)
    assert load_traces(path) == traces
    assert json.loads(path.read_text())["format_version"] == 1
