"""Dual-language Vosk streaming recognition with per-utterance voting (design spec §06).

KK/RU code-switching mid-call is the norm in Kazakhstan, and no single small streaming
model covers both. Strategy: feed the same 16 kHz mono PCM stream to a Kazakh and a
Russian `KaldiRecognizer` in parallel; when either detects an utterance endpoint, flush
the other at the same boundary and keep the hypothesis with the higher average word
confidence. The classifier downstream is code-switch-native, so per-utterance language
granularity is enough — ASR only needs to get the words right.

Between endpoints the current best partial is surfaced for the live UI. Finals are
emitted as `CommittedUtterance` (confidence = mean word confidence), plugging straight
into `qorgan.live.session.advance()`.

All vosk imports are lazy; tests inject scripted fake recognizers.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator, Mapping
from typing import Any

from pydantic import BaseModel, ConfigDict

from qorgan.config import get_config
from qorgan.asr.stream import CommittedUtterance

_FULL_CONFIDENCE = 1.0
_RECOGNIZER_CACHE: dict[tuple[str, ...], dict[str, Any]] = {}


class Partial(BaseModel):
    """An in-progress (not yet endpoint-committed) hypothesis for the live UI."""

    model_config = ConfigDict(frozen=True)

    text: str
    language: str


def recognize_stream(
    chunks: Iterable[bytes], *, recognizers: Mapping[str, Any] | None = None
) -> Iterator[Partial | CommittedUtterance]:
    """Recognize a PCM16 mono stream, yielding partials and voted committed utterances.

    Args:
        chunks: 16 kHz (``config.asr_sample_rate``) PCM16 mono byte chunks.
        recognizers: language → ``KaldiRecognizer``-like object (``AcceptWaveform`` /
            ``Result`` / ``PartialResult`` / ``FinalResult`` returning vosk JSON).
            Injectable for tests; defaults to lazily-loaded KK + RU models.
    """
    active = dict(recognizers) if recognizers is not None else _load_recognizers()
    preferred = next(iter(active))  # partial-display language; follows the last vote
    last_partial = ""

    for chunk in chunks:
        # Feed every recognizer first so both stay aligned on the same audio.
        fired = {language: rec.AcceptWaveform(chunk) for language, rec in active.items()}

        if any(fired.values()):
            language, text, confidence = _vote(
                {
                    language: (rec.Result() if fired[language] else rec.FinalResult())
                    for language, rec in active.items()
                }
            )
            last_partial = ""
            if text:
                preferred = language
                yield CommittedUtterance(text=text, confidence=confidence, language=language)
            continue

        partial_language, partial_text = _best_partial(active, preferred)
        if partial_text and partial_text != last_partial:
            last_partial = partial_text
            yield Partial(text=partial_text, language=partial_language)

    # End of stream: flush whatever both decoders still hold and vote once more.
    language, text, confidence = _vote(
        {language: rec.FinalResult() for language, rec in active.items()}
    )
    if text:
        yield CommittedUtterance(text=text, confidence=confidence, language=language)


def _vote(raw_results: Mapping[str, str]) -> tuple[str, str, float]:
    """Pick the (language, text, mean-word-confidence) hypothesis with the highest
    confidence; empty-text hypotheses always lose."""
    best_language, best_text, best_confidence = "", "", -1.0
    for language, raw in raw_results.items():
        text, confidence = _parse_result(raw)
        if text and confidence > best_confidence:
            best_language, best_text, best_confidence = language, text, confidence
    return best_language, best_text, max(best_confidence, 0.0)


def _parse_result(raw: str) -> tuple[str, float]:
    """Extract (text, mean word confidence) from a vosk Result/FinalResult JSON string.

    Vosk omits per-word ``conf`` when it is certain, so missing confidences default to
    full; the mean is clamped to [0, 1].
    """
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return "", 0.0
    text = str(payload.get("text", "")).strip()
    if not text:
        return "", 0.0
    words = payload.get("result") or []
    confidences = [float(word.get("conf", _FULL_CONFIDENCE)) for word in words]
    mean = sum(confidences) / len(confidences) if confidences else _FULL_CONFIDENCE
    return text, min(_FULL_CONFIDENCE, max(0.0, mean))


def _best_partial(recognizers: Mapping[str, Any], preferred: str) -> tuple[str, str]:
    """The preferred language's partial, falling back to any non-silent other."""
    ordered = [preferred, *(language for language in recognizers if language != preferred)]
    for language in ordered:
        payload = json.loads(recognizers[language].PartialResult())
        text = str(payload.get("partial", "")).strip()
        if text:
            return language, text
    return preferred, ""


def _load_recognizers() -> dict[str, Any]:  # pragma: no cover - loads real vosk models
    """Load (and cache) the configured KK + RU streaming recognizers.

    ``vosk.Model(model_name=...)`` downloads to ``~/.cache/vosk`` on first use.
    ``SetWords(True)`` enables the per-word confidences the voting relies on.
    """
    cfg = get_config()
    key = (cfg.vosk_model_kk, cfg.vosk_model_ru, str(cfg.asr_sample_rate))
    if key not in _RECOGNIZER_CACHE:
        from vosk import KaldiRecognizer, Model

        recognizers: dict[str, Any] = {}
        for language, model_name in (("kk", cfg.vosk_model_kk), ("ru", cfg.vosk_model_ru)):
            recognizer = KaldiRecognizer(Model(model_name=model_name), cfg.asr_sample_rate)
            recognizer.SetWords(True)
            recognizers[language] = recognizer
        _RECOGNIZER_CACHE[key] = recognizers
    return _RECOGNIZER_CACHE[key]
