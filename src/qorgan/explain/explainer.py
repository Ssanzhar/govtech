"""Turns a `ScoreResult` into a UI-ready, grounded, localized `Explanation`.

Every highlighted phrase is a real attributed span from the transcript (re-validated
here, never trusted blindly) and every tag name comes from `taxonomy.py`. All user-facing
strings are templated (RU/KK) and loaded from `explain/templates_{locale}.yaml` -- never
free-form LLM prose (CLAUDE.md §6 / D5 decision). Confidence is surfaced as a localized
band and, for uncalibrated backends, explicitly labeled so in the caveat (D4-2 / gap G7).
"""

from __future__ import annotations

from pathlib import Path

from qorgan.config import get_config
from qorgan.data.schema import Explanation, ScoreResult, validate_verbatim_spans
from qorgan.explain.templates import (
    ExplanationTemplates,
    confidence_band,
    load_templates,
    render_caveat,
    render_reason,
)
from qorgan.taxonomy import get_taxonomy

# Backends whose `raw_confidence` is calibrated (temperature scaling for xlmr; sigmoid
# calibration for the linear head). Every other backend's confidence is labeled
# uncalibrated in the caveat (gap G7).
_CALIBRATED_BACKENDS = frozenset({"xlmr", "linear"})

_TEMPLATES_DIR = Path(__file__).resolve().parent
_TEMPLATE_CACHE: dict[str, ExplanationTemplates] = {}


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

    templates = _templates_for(locale)
    highlights = tuple(sorted(result.attributions, key=lambda span: span.start))

    tactic_names = [name for name in (_display_name(tag.id, locale) for tag in result.tags) if name]
    reason = render_reason(templates, tactic_names, [span.text for span in highlights])
    caveat = render_caveat(
        templates, backend=result.backend, is_calibrated=result.backend in _CALIBRATED_BACKENDS
    )

    return Explanation(
        reason=reason,
        highlights=highlights,
        tags=result.tags,
        confidence=result.raw_confidence,
        confidence_label=confidence_band(templates, result.raw_confidence),
        caveat=caveat,
        human_note=templates.human_note,
    )


def _templates_for(locale: str) -> ExplanationTemplates:
    if locale not in _TEMPLATE_CACHE:
        _TEMPLATE_CACHE[locale] = load_templates(locale, _TEMPLATES_DIR)
    return _TEMPLATE_CACHE[locale]


def _display_name(tactic_id: str, locale: str) -> str | None:
    taxonomy = get_taxonomy()
    try:
        return taxonomy.display_name(tactic_id, locale)
    except KeyError:
        # An unknown tag id (e.g. from a future taxonomy version, or a bug upstream) must
        # never crash the explanation -- just omit it from the tag-name summary.
        # `result.tags` itself is still passed through unchanged.
        return None
