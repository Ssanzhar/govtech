"""Per-utterance live-call pipeline (design spec §§06-10): committed utterance →
rolling window → `predict.score()` → suspicion meter → grounded evidence →
recommendations.

Pure and immutable: `advance()` returns a new `LiveSessionState` plus the `LiveUpdate`
the UI renders for that turn. The rolling window keeps *head + tail* once a call outgrows
the cap — the opening (where impersonation is established) and the recent turns (where
requests happen) — while the full utterance list is always retained for the post-call
summary and report.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from qorgan.asr.stream import CommittedUtterance
from qorgan.classifier.predict import score
from qorgan.config import get_config
from qorgan.data.schema import UTTERANCE_JOIN, ScoreResult, Span, TacticTag
from qorgan.explain.recommend import Recommendation, recommend
from qorgan.live import meter as meter_mod
from qorgan.live.meter import Band, MeterState
from qorgan.taxonomy import get_taxonomy

# Rolling-window cap (§07): ~1,500 tokens of context. Beyond it, the window keeps the
# first `WINDOW_HEAD_UTTERANCES` turns plus as many trailing turns as fit.
MAX_WINDOW_CHARS = 6000
WINDOW_HEAD_UTTERANCES = 4


class LiveUpdate(BaseModel):
    """Everything the UI needs to render one turn: meter, band, grounded evidence
    (spans over `window_text`, first time each phrase is seen), and recommendations."""

    model_config = ConfigDict(frozen=True)

    meter: MeterState
    band: Band
    window_text: str
    result: ScoreResult
    new_evidence: tuple[Span, ...] = ()
    recommendation: Recommendation


class LiveSessionState(BaseModel):
    """Immutable state of one live call: full transcript, meter, accumulated tactics,
    and the evidence phrases already surfaced (for cross-turn dedup)."""

    model_config = ConfigDict(frozen=True)

    locale: str
    backend: str | None = None
    utterances: tuple[str, ...] = ()
    meter: MeterState
    tags: tuple[TacticTag, ...] = ()
    seen_evidence: tuple[str, ...] = ()

    def transcript(self) -> str:
        """The full call so far, joined the same way corpus transcripts are."""
        return UTTERANCE_JOIN.join(self.utterances)


def initial_session(locale: str, *, backend: str | None = None) -> LiveSessionState:
    """A fresh pre-call session. Raises `ValueError` for an unsupported locale."""
    cfg = get_config()
    if locale not in cfg.supported_locales:
        raise ValueError(f"Unsupported locale {locale!r}; expected one of {cfg.supported_locales}")
    return LiveSessionState(locale=locale, backend=backend, meter=meter_mod.initial_state())


def advance(
    state: LiveSessionState, utterance: CommittedUtterance
) -> tuple[LiveSessionState, LiveUpdate]:
    """Consume one committed utterance; return the new state and the UI update."""
    utterances = (*state.utterances, utterance.text)
    window_text = _rolling_window(utterances)
    result = score(window_text, backend=state.backend)

    taxonomy = get_taxonomy()
    hard_ids = set(taxonomy.hard_signal_ids())
    hard_signals = {tag.id: tag.weight for tag in result.tags if tag.id in hard_ids}

    meter_state = meter_mod.update(
        state.meter,
        risk=result.risk,
        asr_confidence=utterance.confidence,
        hard_signals=hard_signals,
    )
    band = meter_mod.band(meter_state.score)

    tags = _accumulate_tags(state.tags, result.tags)
    new_evidence = tuple(
        span for span in result.attributions if span.text not in set(state.seen_evidence)
    )

    # Guidance surfaces from Medium band up (§10) — a calm call gets no advice at all.
    recommendation = (
        recommend(tags, state.locale, confidence=result.raw_confidence)
        if band != "low"
        else Recommendation()
    )

    next_state = LiveSessionState(
        locale=state.locale,
        backend=state.backend,
        utterances=utterances,
        meter=meter_state,
        tags=tags,
        seen_evidence=(*state.seen_evidence, *(span.text for span in new_evidence)),
    )
    update = LiveUpdate(
        meter=meter_state,
        band=band,
        window_text=window_text,
        result=result,
        new_evidence=new_evidence,
        recommendation=recommendation,
    )
    return next_state, update


def _rolling_window(utterances: tuple[str, ...]) -> str:
    """Join utterances for scoring, truncating to head + tail once over the cap.

    The newest utterance is always included, even if it alone exceeds the budget —
    the current turn is what the meter is reacting to.
    """
    full = UTTERANCE_JOIN.join(utterances)
    if len(full) <= MAX_WINDOW_CHARS:
        return full

    head = list(utterances[:WINDOW_HEAD_UTTERANCES])
    budget = MAX_WINDOW_CHARS - len(UTTERANCE_JOIN.join(head)) - len(UTTERANCE_JOIN)
    tail: list[str] = []
    for text in reversed(utterances[WINDOW_HEAD_UTTERANCES:]):
        cost = len(text) + len(UTTERANCE_JOIN)
        if budget - cost < 0 and tail:
            break
        tail.insert(0, text)
        budget -= cost
    return UTTERANCE_JOIN.join((*head, *tail))


def _accumulate_tags(
    existing: tuple[TacticTag, ...], new: tuple[TacticTag, ...]
) -> tuple[TacticTag, ...]:
    """Union of detected tactics across the call, keeping the max weight per id."""
    weights: dict[str, float] = {tag.id: tag.weight for tag in existing}
    for tag in new:
        weights[tag.id] = max(weights.get(tag.id, 0.0), tag.weight)
    return tuple(TacticTag(id=tactic_id, weight=weight) for tactic_id, weight in weights.items())
