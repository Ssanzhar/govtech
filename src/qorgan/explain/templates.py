"""Loads per-locale (RU/KK) explanation templates and renders them into UI strings.

Invariant: a locale's YAML template file must define every key `ExplanationTemplates`
needs (including all four `confidence` bands) -- `load_templates` fails loudly rather
than falling back to a partially-populated template, so a broken localization file
never silently ships a blank explanation. Pure rendering only: no taxonomy/config
imports here, callers resolve tactic names and pass them in already localized.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import yaml

_HIGH_CONFIDENCE = 0.8
_MEDIUM_CONFIDENCE = 0.6

_REQUIRED_KEYS = (
    "reason_template",
    "no_signal_reason",
    "spans_placeholder",
    "caveat",
    "uncalibrated_suffix",
    "human_note",
    "confidence",
)
_REQUIRED_CONFIDENCE_BANDS = ("high", "medium", "low", "unknown")


class TemplateError(ValueError):
    """Raised when a locale's template YAML is missing, malformed, or incomplete."""


@dataclass(frozen=True)
class ExplanationTemplates:
    """A fully-resolved set of localized explanation strings for one locale."""

    reason_template: str
    no_signal_reason: str
    spans_placeholder: str
    caveat: str
    uncalibrated_suffix: str
    human_note: str
    confidence: dict[str, str]


def load_templates(locale: str, templates_dir: Path) -> ExplanationTemplates:
    """Read and validate `templates_dir / f"templates_{locale}.yaml"`.

    Raises `TemplateError` if the file is missing, the YAML is invalid or not a
    mapping, any required top-level key is missing, or `confidence` is missing any of
    `high`/`medium`/`low`/`unknown`.
    """
    path = templates_dir / f"templates_{locale}.yaml"
    if not path.exists():
        raise TemplateError(f"Template file not found: {path}")

    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise TemplateError(f"Invalid YAML in {path}: {exc}") from exc

    if not isinstance(raw, dict):
        raise TemplateError(
            f"Template file {path} must contain a top-level mapping, got {type(raw).__name__}"
        )

    for key in _REQUIRED_KEYS:
        if key not in raw:
            raise TemplateError(f"Template file {path} is missing required key {key!r}")

    confidence = raw["confidence"]
    if not isinstance(confidence, dict):
        raise TemplateError(f"Template file {path} key 'confidence' must be a mapping")
    for band in _REQUIRED_CONFIDENCE_BANDS:
        if band not in confidence:
            raise TemplateError(f"Template file {path} 'confidence' is missing key {band!r}")

    return ExplanationTemplates(
        reason_template=raw["reason_template"],
        no_signal_reason=raw["no_signal_reason"],
        spans_placeholder=raw["spans_placeholder"],
        caveat=raw["caveat"],
        uncalibrated_suffix=raw["uncalibrated_suffix"],
        human_note=raw["human_note"],
        confidence=dict(confidence),
    )


def render_reason(
    templates: ExplanationTemplates, tactic_names: Sequence[str], phrases: Sequence[str]
) -> str:
    """Render the "why this looks like a scam" line from tactic names + trigger phrases.

    Empty `tactic_names` means no signal was detected -- returns
    `templates.no_signal_reason` verbatim. Otherwise phrases are wrapped in guillemets
    («»), or `templates.spans_placeholder` is used when there are no phrases to show.
    """
    if not tactic_names:
        return templates.no_signal_reason

    spans = ", ".join(f"«{phrase}»" for phrase in phrases) if phrases else templates.spans_placeholder
    return templates.reason_template.format(tags=", ".join(tactic_names), spans=spans)


def render_caveat(templates: ExplanationTemplates, *, backend: str, is_calibrated: bool) -> str:
    """Render the model-can-be-wrong caveat, appending an uncalibrated-confidence
    disclaimer naming `backend` when `is_calibrated` is `False`."""
    caveat = templates.caveat
    if not is_calibrated:
        caveat += templates.uncalibrated_suffix.format(backend=backend)
    return caveat


def confidence_band(templates: ExplanationTemplates, confidence: float | None) -> str:
    """Map a raw confidence score to its localized band label.

    `None` -> `unknown`; `>= 0.8` -> `high`; `>= 0.6` -> `medium`; else `low`.
    """
    if confidence is None:
        return templates.confidence["unknown"]
    if confidence >= _HIGH_CONFIDENCE:
        return templates.confidence["high"]
    if confidence >= _MEDIUM_CONFIDENCE:
        return templates.confidence["medium"]
    return templates.confidence["low"]
