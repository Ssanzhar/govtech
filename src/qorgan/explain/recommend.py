"""Tactic → advice recommendation engine (design spec §10).

Recommendations are generated deterministically from detected tactics, gated by
confidence, and localized RU/KK — never free-form. Advice strings live in
`explain/advice_{locale}.yaml` beside the explanation templates; hard-signal advice
always outranks contextual advice; below a confidence floor the engine softens to
verification questions instead of asserting a scam.

Coverage invariant (validated on first use per locale): every taxonomy tactic id has an
advice string, so a detected tactic can never lack localized guidance.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict

from qorgan.config import get_config
from qorgan.data.schema import TacticTag
from qorgan.taxonomy import get_taxonomy

# Below this decision confidence the engine stops asserting "scam" and softens to
# verification questions. Matches the medium-confidence band edge in `templates.py`.
LOW_CONFIDENCE_FLOOR = 0.6

_ADVICE_DIR = Path(__file__).resolve().parent
_REQUIRED_KEYS = ("tactic_advice", "verification_questions", "low_confidence_note")
_ADVICE_CACHE: dict[str, "AdviceTemplates"] = {}


class AdviceError(ValueError):
    """Raised for a missing/malformed advice file, incomplete tactic coverage, or an
    unsupported locale."""


@dataclass(frozen=True)
class AdviceTemplates:
    """Fully-resolved localized advice strings for one locale."""

    tactic_advice: dict[str, str]
    verification_questions: tuple[str, ...]
    low_confidence_note: str


class Recommendation(BaseModel):
    """UI-ready recommendation set for one scoring update.

    `advices` are ranked tactic-specific do-this-now strings (hard signals first, then
    by tactic weight). When `softened` is true the engine declined to assert a scam
    (confidence below `LOW_CONFIDENCE_FLOOR`): `advices` is empty and `note` carries the
    localized verify-independently message instead.
    """

    model_config = ConfigDict(frozen=True)

    advices: tuple[str, ...] = ()
    verification_questions: tuple[str, ...] = ()
    softened: bool = False
    note: str | None = None


def load_advice(locale: str, *, advice_dir: Path = _ADVICE_DIR) -> AdviceTemplates:
    """Read and structurally validate `advice_dir / f"advice_{locale}.yaml"`.

    Raises `AdviceError` if the file is missing, the YAML is invalid or not a mapping,
    any required key is missing, or `tactic_advice` is not a mapping of non-blank
    strings. Tactic *coverage* against the live taxonomy is validated separately in
    `recommend()` (this loader stays taxonomy-free, mirroring `templates.py`).
    """
    path = advice_dir / f"advice_{locale}.yaml"
    if not path.exists():
        raise AdviceError(f"Advice file not found: {path}")

    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise AdviceError(f"Invalid YAML in {path}: {exc}") from exc

    if not isinstance(raw, dict):
        raise AdviceError(
            f"Advice file {path} must contain a top-level mapping, got {type(raw).__name__}"
        )
    for key in _REQUIRED_KEYS:
        if key not in raw:
            raise AdviceError(f"Advice file {path} is missing required key {key!r}")

    tactic_advice = raw["tactic_advice"]
    if not isinstance(tactic_advice, dict) or not all(
        isinstance(k, str) and isinstance(v, str) and v.strip() for k, v in tactic_advice.items()
    ):
        raise AdviceError(
            f"Advice file {path} key 'tactic_advice' must map tactic ids to non-blank strings"
        )
    questions = raw["verification_questions"]
    if not isinstance(questions, list) or not all(
        isinstance(q, str) and q.strip() for q in questions
    ):
        raise AdviceError(
            f"Advice file {path} key 'verification_questions' must be a list of non-blank strings"
        )

    return AdviceTemplates(
        tactic_advice=dict(tactic_advice),
        verification_questions=tuple(questions),
        low_confidence_note=str(raw["low_confidence_note"]),
    )


def recommend(
    tags: Sequence[TacticTag], locale: str, *, confidence: float | None = None
) -> Recommendation:
    """Build the ranked, localized recommendation set for the detected `tags`.

    Unknown tactic ids are skipped (mirroring the explainer: a future-taxonomy tag must
    never crash the UI). Empty `tags` yields an empty recommendation — guidance only
    surfaces when something was detected.
    """
    cfg = get_config()
    if locale not in cfg.supported_locales:
        raise AdviceError(f"Unsupported locale {locale!r}; expected one of {cfg.supported_locales}")
    if not tags:
        return Recommendation()

    templates = _advice_for(locale)
    taxonomy = get_taxonomy()
    known = [tag for tag in tags if tag.id in set(taxonomy.tactic_ids())]
    hard_ids = set(taxonomy.hard_signal_ids())

    # Hard-signal advice first, then by descending tactic weight; stable within ties.
    ranked = sorted(known, key=lambda tag: (tag.id not in hard_ids, -tag.weight))

    advices: list[str] = []
    for tag in ranked:
        advice = templates.tactic_advice[tag.id]  # coverage validated in _advice_for()
        if advice not in advices:
            advices.append(advice)

    if confidence is not None and confidence < LOW_CONFIDENCE_FLOOR:
        return Recommendation(
            advices=(),
            verification_questions=templates.verification_questions,
            softened=True,
            note=templates.low_confidence_note,
        )
    return Recommendation(
        advices=tuple(advices),
        verification_questions=templates.verification_questions,
        softened=False,
        note=None,
    )


def _advice_for(locale: str) -> AdviceTemplates:
    """Cached advice loader that also enforces the taxonomy-coverage invariant."""
    if locale not in _ADVICE_CACHE:
        templates = load_advice(locale)
        missing = set(get_taxonomy().tactic_ids()) - set(templates.tactic_advice)
        if missing:
            raise AdviceError(
                f"advice_{locale}.yaml is missing advice for tactic(s): {sorted(missing)}"
            )
        _ADVICE_CACHE[locale] = templates
    return _ADVICE_CACHE[locale]
