"""Frozen data contracts shared across the whole pipeline (generation, labeling,
classification, explanation, clustering).

Every model here is immutable (`frozen=True`) and validated at construction. The
central invariant enforced across the corpus and the live classifier alike: a trigger
`Span` must be a **verbatim substring** of the transcript it claims to come from --
explanations are grounded, never hallucinated (CLAUDE.md SS6).
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# Delimiter used to join `Utterance.text` into a single transcript string. Shared by
# `Dialogue.transcript()` and `qorgan.explain.windowing` so cumulative windows and
# corpus transcripts stay byte-for-byte consistent.
UTTERANCE_JOIN = "\n"

SupportedLanguage = Literal["ru", "kk", "mixed"]


class SpanValidationError(ValueError):
    """Raised when a trigger/attribution span is not a verbatim substring of its transcript."""


class Span(BaseModel):
    """A character-offset span into some transcript, plus the verbatim text it covers.

    Self-consistency (offsets vs. text length) is validated here. Whether `text` is
    actually a substring of a *particular* transcript is validated by
    `validate_verbatim_spans()` wherever a span and its owning transcript are both
    available (e.g. `Dialogue`, `Incident`, `explain.explainer.explain`).
    """

    model_config = ConfigDict(frozen=True)

    text: str
    start: int = Field(ge=0)
    end: int = Field(ge=0)

    @field_validator("text")
    @classmethod
    def _text_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Span.text must not be blank")
        return value

    @model_validator(mode="after")
    def _end_after_start_and_length_matches(self) -> "Span":
        if self.end <= self.start:
            raise ValueError(f"Span.end ({self.end}) must be greater than Span.start ({self.start})")
        if self.end - self.start != len(self.text):
            raise ValueError(
                f"Span length ({self.end - self.start}) does not match "
                f"len(text)={len(self.text)} for text={self.text!r}"
            )
        return self


def validate_verbatim_spans(spans: Sequence[Span], transcript: str) -> None:
    """Raise `SpanValidationError` unless every span's claimed offsets slice `transcript`
    into exactly `span.text`."""
    for span in spans:
        actual = transcript[span.start : span.end]
        if actual != span.text:
            raise SpanValidationError(
                f"Span text {span.text!r} at [{span.start}:{span.end}] does not match "
                f"transcript slice {actual!r}"
            )


def spans_from_phrases(phrases: Sequence[str], transcript: str) -> tuple[Span, ...]:
    """Resolve raw phrase strings (e.g. LLM-returned "trigger phrases") into grounded
    `Span`s via a verbatim substring search in `transcript`.

    LLMs are unreliable at reporting character offsets, so callers should only ever ask
    a model for phrase *text*, then locate it themselves with this helper. Any phrase not
    actually present verbatim in `transcript` is silently dropped -- we never fabricate a
    highlighted span. Shared by `classifier/llm_classifier.py`, `classifier/predict.py`'s
    mock heuristic, and `data/generate.py`.
    """
    spans: list[Span] = []
    for phrase in phrases:
        if not isinstance(phrase, str) or not phrase:
            continue
        start = transcript.find(phrase)
        if start == -1:
            continue
        spans.append(Span(text=phrase, start=start, end=start + len(phrase)))
    return tuple(spans)


class TacticTag(BaseModel):
    """A tactic label applied to a dialogue/incident/score, with a contribution weight."""

    model_config = ConfigDict(frozen=True)

    id: str
    weight: float = Field(default=1.0, ge=0.0, le=1.0)

    @field_validator("id")
    @classmethod
    def _id_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("TacticTag.id must not be blank")
        return value


class Utterance(BaseModel):
    """One turn in a dialogue."""

    model_config = ConfigDict(frozen=True)

    speaker: str
    text: str

    @field_validator("speaker", "text")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Utterance.speaker/text must not be blank")
        return value


class Label(BaseModel):
    """Ground-truth (or LLM-assigned) label for a `Dialogue`/`Incident`."""

    model_config = ConfigDict(frozen=True)

    risk: float = Field(ge=0.0, le=1.0)
    tactic_tags: tuple[TacticTag, ...] = ()
    trigger_spans: tuple[Span, ...] = ()
    is_hard_negative: bool = False


class Dialogue(BaseModel):
    """A synthetic or real training/eval dialogue: utterances + label, self-grounded."""

    model_config = ConfigDict(frozen=True)

    id: str
    language: SupportedLanguage
    utterances: tuple[Utterance, ...]
    label: Label

    @field_validator("utterances")
    @classmethod
    def _at_least_one_utterance(cls, value: tuple[Utterance, ...]) -> tuple[Utterance, ...]:
        if not value:
            raise ValueError("Dialogue must contain at least one utterance")
        return value

    def transcript(self) -> str:
        """The dialogue rendered as a single transcript string (utterances joined)."""
        return UTTERANCE_JOIN.join(u.text for u in self.utterances)

    @model_validator(mode="after")
    def _trigger_spans_are_verbatim(self) -> "Dialogue":
        validate_verbatim_spans(self.label.trigger_spans, self.transcript())
        return self


class Incident(BaseModel):
    """A Level-2 incident record: transcript + label + linking metadata (phone/time/script)."""

    model_config = ConfigDict(frozen=True)

    id: str
    dialogue_id: str
    transcript: str
    label: Label
    phone_number: str | None = None
    timestamp: datetime | None = None
    script_family: str | None = None

    @model_validator(mode="after")
    def _trigger_spans_are_verbatim(self) -> "Incident":
        validate_verbatim_spans(self.label.trigger_spans, self.transcript)
        return self


class ScoreResult(BaseModel):
    """Unified classifier output -- the `classifier/predict.py` contract.

    `backend` records which classifier produced the result (`llm` | `xlmr` | `mock`).
    `raw_confidence` is the backend's self-reported confidence, uncalibrated unless the
    backend is `xlmr` (gap G7 -- see `explain/explainer.py` for how this is surfaced).
    """

    model_config = ConfigDict(frozen=True)

    risk: float = Field(ge=0.0, le=1.0)
    tags: tuple[TacticTag, ...] = ()
    attributions: tuple[Span, ...] = ()
    backend: str
    raw_confidence: float | None = Field(default=None, ge=0.0, le=1.0)

    @field_validator("backend")
    @classmethod
    def _backend_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("ScoreResult.backend must not be blank")
        return value


class Explanation(BaseModel):
    """The `explain/explainer.py` contract: a UI-ready, grounded, localized explanation."""

    model_config = ConfigDict(frozen=True)

    reason: str
    highlights: tuple[Span, ...] = ()
    tags: tuple[TacticTag, ...] = ()
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    # Localized qualitative confidence band ("high/medium/low", RU/KK), derived from
    # `confidence` + backend calibration (D4-2). Optional so older callers stay valid.
    confidence_label: str | None = None
    caveat: str
    human_note: str


class Organization(BaseModel):
    """The `analytics/cluster.py` contract: a clustered scam "organization" for L2."""

    model_config = ConfigDict(frozen=True)

    id: str
    members: tuple[str, ...]
    numbers: tuple[str, ...] = ()
    representative_script: str | None = None
    priority: float = 0.0
    is_novel: bool = False

    @field_validator("members")
    @classmethod
    def _at_least_one_member(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value:
            raise ValueError("Organization.members must not be empty")
        return value
