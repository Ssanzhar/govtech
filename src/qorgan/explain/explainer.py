"""Turns a `ScoreResult` into a UI-ready, grounded, localized `Explanation`.

Every highlighted phrase is a real attributed span from the transcript (re-validated
here, never trusted blindly) and every tag name comes from `taxonomy.py`. The reason
string is templated (RU/KK), never free-form LLM prose (CLAUDE.md SS6 / D5 decision).

Day-1 scope: minimal inline templates. Day 4 (D4-1) extracts these into
`explain/templates_ru.yaml` / `templates_kk.yaml` for easier localization review.
"""

from __future__ import annotations

from qorgan.config import get_config
from qorgan.data.schema import Explanation, ScoreResult, validate_verbatim_spans
from qorgan.taxonomy import get_taxonomy

_CAVEAT = {
    "ru": (
        "Модель может ошибаться, особенно на нетипичных или прерванных звонках. "
        "Оценивайте риск вместе с другими признаками."
    ),
    "kk": (
        "Модель әсіресе үзілген немесе әдеттен тыс қоңырауларда қателесуі мүмкін. "
        "Тәуекелді басқа белгілермен бірге бағалаңыз."
    ),
}
_UNCALIBRATED_SUFFIX = {
    "ru": " Оценка доверия ({backend}) не откалибрована — это базовая LLM-модель.",
    "kk": " Сенімділік бағасы ({backend}) калибрленбеген — бұл негізгі LLM моделі.",
}
_HUMAN_NOTE = {
    "ru": "Это инструмент поддержки решений. Окончательное решение принимает человек.",
    "kk": "Бұл шешім қабылдауды қолдау құралы. Түпкілікті шешімді адам қабылдайды.",
}
_NO_SIGNAL_REASON = {
    "ru": "Явных признаков мошенничества не обнаружено.",
    "kk": "Алаяқтық белгілері анық байқалмады.",
}
_REASON_TEMPLATE = {
    "ru": "Обнаружены признаки: {tags}. Триггерные фразы: {spans}.",
    "kk": "Анықталған белгілер: {tags}. Триггер фразалар: {spans}.",
}

# Backends whose `raw_confidence` is calibrated (temperature/isotonic, Day 3). All other
# backends' confidence is labeled uncalibrated in the caveat (gap G7).
_CALIBRATED_BACKENDS = frozenset({"xlmr"})


class ExplainerError(ValueError):
    """Raised for invalid `explain()` inputs (unsupported locale, empty transcript)."""


def explain(result: ScoreResult, transcript: str, locale: str) -> Explanation:
    """Build a grounded, localized `Explanation` for `result` over `transcript`."""
    cfg = get_config()
    if locale not in cfg.supported_locales:
        raise ExplainerError(
            f"Unsupported locale {locale!r}; expected one of {cfg.supported_locales}"
        )
    if not transcript or not transcript.strip():
        raise ExplainerError("transcript must not be empty")

    # Re-validate at this boundary too: never show a highlight that isn't actually a
    # verbatim substring of the transcript being explained, even if `result` was scored
    # against different text upstream.
    validate_verbatim_spans(result.attributions, transcript)

    highlights = tuple(sorted(result.attributions, key=lambda span: span.start))
    reason = _build_reason(result, highlights, locale)
    caveat = _build_caveat(result, locale)

    return Explanation(
        reason=reason,
        highlights=highlights,
        tags=result.tags,
        confidence=result.raw_confidence,
        caveat=caveat,
        human_note=_HUMAN_NOTE[locale],
    )


def _build_reason(result: ScoreResult, highlights: tuple, locale: str) -> str:
    tag_names = [_display_name(tag.id, locale) for tag in result.tags]
    tag_names = [name for name in tag_names if name]
    if not tag_names:
        return _NO_SIGNAL_REASON[locale]

    span_texts = ", ".join(f"«{span.text}»" for span in highlights) if highlights else "—"
    return _REASON_TEMPLATE[locale].format(tags=", ".join(tag_names), spans=span_texts)


def _build_caveat(result: ScoreResult, locale: str) -> str:
    caveat = _CAVEAT[locale]
    if result.backend not in _CALIBRATED_BACKENDS:
        caveat += _UNCALIBRATED_SUFFIX[locale].format(backend=result.backend)
    return caveat


def _display_name(tactic_id: str, locale: str) -> str | None:
    taxonomy = get_taxonomy()
    try:
        return taxonomy.display_name(tactic_id, locale)
    except KeyError:
        # An unknown tag id (e.g. from a future taxonomy version, or a bug upstream)
        # must never crash the explanation -- just omit it from the tag-name summary.
        # `result.tags` itself is still passed through unchanged.
        return None
