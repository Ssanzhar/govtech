"""The 0-100 suspicion meter — exact scoring methodology from the design spec (§08).

The displayed score follows the calibrated per-turn risk through an asymmetric EMA:
it rises fast (two consistent turns reach the target band) and decays slowly (a scammer
changing topic does not reset accumulated evidence). Each update is weighted by the ASR
confidence of the utterance that produced it, hard-signal tactics floor the score at the
High/Critical band edges, the warning latch uses the shipped FPR-tuned hysteresis pair
(`config.risk_threshold_enter/_exit`), and every change is recorded in an evidence
ledger so the meter can always answer "why did you just go up?".

Pure and immutable: `update()` returns a new `MeterState`, never mutates its input.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from qorgan.config import Config, get_config

# Display bands (design spec §04/§08): Low 0-30 · Medium 31-60 · High 61-80 · Critical 81-100.
BAND_LOW_MAX = 30.0
BAND_MEDIUM_MAX = 60.0
BAND_HIGH_MAX = 80.0
SCORE_MAX = 100.0

# Hard-signal shortcut (§08 step 4): a hard-signal tactic detected at/above this
# confidence floors the score at High; a second *distinct* one floors it at Critical.
# Floors only ever raise the score. Aligned with the band edges above (61 = first score
# inside High, 81 = first inside Critical).
HARD_SIGNAL_CONFIDENCE_FLOOR = 0.8
SINGLE_HARD_SIGNAL_FLOOR = BAND_MEDIUM_MAX + 1.0
DOUBLE_HARD_SIGNAL_FLOOR = BAND_HIGH_MAX + 1.0

Band = Literal["low", "medium", "high", "critical"]


class MeterEvent(BaseModel):
    """One ledger entry: what a single update did to the score, and why."""

    model_config = ConfigDict(frozen=True)

    turn_index: int = Field(ge=1)
    risk: float = Field(ge=0.0, le=1.0)
    asr_confidence: float = Field(ge=0.0, le=1.0)
    score_before: float = Field(ge=0.0, le=SCORE_MAX)
    score_after: float = Field(ge=0.0, le=SCORE_MAX)
    new_hard_signal_ids: tuple[str, ...] = ()


class MeterState(BaseModel):
    """Immutable meter state after `turn_index` updates."""

    model_config = ConfigDict(frozen=True)

    score: float = Field(ge=0.0, le=SCORE_MAX)
    latched: bool
    turn_index: int = Field(ge=0)
    hard_signal_ids: frozenset[str]
    ledger: tuple[MeterEvent, ...]


def initial_state() -> MeterState:
    """The calm pre-call state: score 0, unlatched, no evidence."""
    return MeterState(score=0.0, latched=False, turn_index=0, hard_signal_ids=frozenset(), ledger=())


def band(score: float) -> Band:
    """Map a 0-100 score to its display band. Raises `ValueError` out of range."""
    if not 0.0 <= score <= SCORE_MAX:
        raise ValueError(f"score must be in [0, {SCORE_MAX}], got {score}")
    if score <= BAND_LOW_MAX:
        return "low"
    if score <= BAND_MEDIUM_MAX:
        return "medium"
    if score <= BAND_HIGH_MAX:
        return "high"
    return "critical"


def update(
    state: MeterState,
    *,
    risk: float,
    asr_confidence: float = 1.0,
    hard_signals: Mapping[str, float] | None = None,
    config: Config | None = None,
) -> MeterState:
    """Advance the meter by one committed utterance; returns a new state.

    Args:
        state: The meter state before this turn.
        risk: Calibrated scam probability in [0, 1] from `predict.score()` over the
            rolling window.
        asr_confidence: Transcription confidence in [0, 1] for the utterance that
            produced this update; garbled audio moves the needle less (§08 step 2).
        hard_signals: Detected hard-signal tactic ids → detection confidence. Signals
            below `HARD_SIGNAL_CONFIDENCE_FLOOR` are ignored; the rest accumulate for
            the whole call (a hard signal never scrolls out of evidence).
        config: Injectable configuration (tests); defaults to `get_config()`.
    """
    if not 0.0 <= risk <= 1.0:
        raise ValueError(f"risk must be in [0, 1], got {risk}")
    if not 0.0 <= asr_confidence <= 1.0:
        raise ValueError(f"asr_confidence must be in [0, 1], got {asr_confidence}")

    cfg = config or get_config()

    # §08 step 3: asymmetric, confidence-weighted EMA toward the calibrated target.
    target = SCORE_MAX * risk
    alpha = cfg.meter_alpha_up if target > state.score else cfg.meter_alpha_down
    score = state.score + alpha * asr_confidence * (target - state.score)

    # §08 step 4: hard-signal floors (never lower the score).
    confident = frozenset(
        signal_id
        for signal_id, confidence in (hard_signals or {}).items()
        if confidence >= HARD_SIGNAL_CONFIDENCE_FLOOR
    )
    accumulated = state.hard_signal_ids | confident
    if len(accumulated) >= 2:
        score = max(score, DOUBLE_HARD_SIGNAL_FLOOR)
    elif len(accumulated) == 1:
        score = max(score, SINGLE_HARD_SIGNAL_FLOOR)
    score = min(max(score, 0.0), SCORE_MAX)

    # §08 step 5: the warning latch reuses the shipped FPR-tuned hysteresis pair,
    # deliberately independent of the cosmetic band edges. Same enter/exit semantics as
    # `explain.windowing.apply_hysteresis`, on the 0-100 scale.
    enter = cfg.risk_threshold_enter * SCORE_MAX
    exit_ = cfg.risk_threshold_exit * SCORE_MAX
    latched = (score > exit_) if state.latched else (score >= enter)

    event = MeterEvent(
        turn_index=state.turn_index + 1,
        risk=risk,
        asr_confidence=asr_confidence,
        score_before=state.score,
        score_after=score,
        new_hard_signal_ids=tuple(sorted(confident - state.hard_signal_ids)),
    )
    return MeterState(
        score=score,
        latched=latched,
        turn_index=state.turn_index + 1,
        hard_signal_ids=accumulated,
        ledger=(*state.ledger, event),
    )
