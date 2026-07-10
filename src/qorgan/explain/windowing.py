"""Cumulative-window scoring + alert hysteresis (gap G8 -- "the risk meter climbs").

`windows()` turns a sequence of utterance texts into cumulative transcript prefixes so a
caller can score a call incrementally, turn-by-turn, and animate a climbing risk meter.
`apply_hysteresis()` turns a sequence of risk scores into a stable on/off alert state,
using separate enter/exit thresholds (`config.risk_threshold_enter/_exit`) so the meter
doesn't flicker around a single threshold.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from qorgan.data.schema import UTTERANCE_JOIN


@dataclass(frozen=True)
class Window:
    """One cumulative scoring window: utterances[0:end_index] joined into `text`."""

    end_index: int
    text: str


def windows(utterances: Sequence[str], *, step: int = 1) -> list[Window]:
    """Split `utterances` into cumulative windows for incremental scoring.

    `window[i].text` is `utterances[0:end_index]` joined with the same delimiter used by
    `Dialogue.transcript()`, so windowed scores stay comparable to a full-transcript
    score. `step` utterances are added per window (default 1 = score after every turn);
    the final window always covers the full transcript, even if it falls short of a full
    `step`.
    """
    if step < 1:
        raise ValueError(f"step must be >= 1, got {step}")
    if not utterances:
        return []

    total = len(utterances)
    result: list[Window] = []
    end_index = step
    while end_index < total:
        result.append(Window(end_index=end_index, text=UTTERANCE_JOIN.join(utterances[:end_index])))
        end_index += step

    if not result or result[-1].end_index != total:
        result.append(Window(end_index=total, text=UTTERANCE_JOIN.join(utterances)))
    return result


def apply_hysteresis(scores: Sequence[float], *, enter: float, exit: float) -> list[bool]:
    """Turn a sequence of risk scores into a stable on/off alert state.

    Once the alert enters (score >= `enter`), it stays on until the score drops to or
    below `exit`. Requires `exit <= enter` (a defense-in-depth check mirroring the one in
    `config.Config`, since this function may be called with ad-hoc thresholds outside the
    global config too).
    """
    if exit > enter:
        raise ValueError(f"exit ({exit}) must be <= enter ({enter})")

    state: list[bool] = []
    is_on = False
    for score in scores:
        if is_on:
            if score <= exit:
                is_on = False
        elif score >= enter:
            is_on = True
        state.append(is_on)
    return state
