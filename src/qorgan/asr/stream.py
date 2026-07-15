"""Pseudo-streaming utterance source for the live pipeline (design spec §06).

Two ways to obtain a stream of committed utterances for `qorgan.live.session`:

- `replay_transcript()` — deterministic replay of an existing transcript, one committed
  utterance per line. Powers the live demo and tests; also the honest stand-in for a
  true streaming decoder in the web prototype (the design spec's Vosk tier is a mobile
  port, out of this repo's scope).
- `stream_transcribe()` — VAD-segmented pseudo-streaming over an audio file via
  faster-whisper: each recognized segment is committed as one utterance, with the
  segment's average log-probability mapped to a [0, 1] ASR confidence that the meter
  uses for confidence weighting (§08 step 2).
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

_FULL_CONFIDENCE = 1.0


class CommittedUtterance(BaseModel):
    """One committed (endpoint-final) utterance with its transcription confidence."""

    model_config = ConfigDict(frozen=True)

    text: str
    confidence: float = Field(ge=0.0, le=1.0)
    start_s: float | None = Field(default=None, ge=0.0)
    end_s: float | None = Field(default=None, ge=0.0)
    # Which recognizer won the per-utterance language vote ("kk"/"ru"); None for sources
    # that don't vote (replay, whisper segments).
    language: str | None = None

    @field_validator("text")
    @classmethod
    def _text_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("CommittedUtterance.text must not be blank")
        return value


def replay_transcript(
    transcript: str, *, default_confidence: float = _FULL_CONFIDENCE
) -> Iterator[CommittedUtterance]:
    """Yield one committed utterance per non-blank transcript line, in order.

    Raises `ValueError` for a blank transcript or an out-of-range confidence — fail fast
    before the first utterance is consumed.
    """
    if not 0.0 <= default_confidence <= 1.0:
        raise ValueError(f"default_confidence must be in [0, 1], got {default_confidence}")
    lines = [line.strip() for line in transcript.splitlines() if line.strip()]
    if not lines:
        raise ValueError("transcript must contain at least one non-blank line")
    for line in lines:
        yield CommittedUtterance(text=line, confidence=default_confidence)


def stream_transcribe(
    audio_path: str | Path, *, model: Any | None = None
) -> Iterator[CommittedUtterance]:
    """Yield committed utterances from `audio_path`, one per faster-whisper segment.

    Confidence is `exp(avg_logprob)` clamped to [0, 1] — the segment-level probability
    proxy whisper exposes. `model` is injectable for tests (same contract as
    `asr.transcribe`: `.transcribe(path) -> (segments, info)`); production lazily loads
    the configured int8 model.
    """
    resolved = Path(audio_path)
    if not resolved.exists():
        raise FileNotFoundError(f"Audio file not found: {resolved}")

    from qorgan.asr.transcribe import _load_model

    whisper_model = model if model is not None else _load_model()
    segments, _info = whisper_model.transcribe(str(resolved))
    for segment in segments:
        text = segment.text.strip()
        if not text:
            continue
        confidence = min(_FULL_CONFIDENCE, max(0.0, math.exp(segment.avg_logprob)))
        yield CommittedUtterance(
            text=text,
            confidence=confidence,
            start_s=max(0.0, float(segment.start)),
            end_s=max(0.0, float(segment.end)),
        )
