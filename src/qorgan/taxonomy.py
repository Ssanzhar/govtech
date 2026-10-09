"""Loads and validates the scam-tactic taxonomy from `data/taxonomy/tactics.yaml`.

Exposes the tactic label space (multi-label; a call typically exhibits several tactics),
`hard_signal` flags (strong standalone fraud indicators), localized RU/KK display names,
and the negative/hard-negative categories used to balance the corpus and drive down FPR.

Version 2 (ADR D63) holds two label spaces in one file: `v1` (the 15 ids the shipped heads
were trained on) and `v2` (12 kept + 6 `imp_*` + 5 new). `tactic_ids()` / `hard_signal_ids()`
/ `active_tactics()` describe the ACTIVE space -- what the classifier trains, decodes and is
evaluated on -- so nothing downstream of the model moves until the retrain flips
`active_label_space`. Display names, advice and report validation work over every id
(`all_tactic_ids()`), so v2 tags render and validate today.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from qorgan.config import get_config

_SUPPORTED_TAXONOMY_VERSIONS = (1, 2)
_SUPPORTED_LOCALES = ("ru", "kk", "en")
# A version-1 file has a single implicit space; every tactic belongs to it.
_V1_SPACE = "v1"


class TaxonomyError(ValueError):
    """Raised when `tactics.yaml` is missing, malformed, or fails validation."""


class TacticDefinition(BaseModel):
    """One scam-tactic label: id, localized names, description, and hard-signal flag."""

    model_config = ConfigDict(frozen=True)

    id: str
    ru: str
    kk: str
    en: str
    description: str
    hard_signal: bool
    examples_ru: tuple[str, ...] = ()
    examples_kk: tuple[str, ...] = ()
    label_spaces: tuple[str, ...] = (_V1_SPACE,)
    superseded_by: tuple[str, ...] = ()

    @field_validator("label_spaces")
    @classmethod
    def _spaces_not_empty(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value or not all(space.strip() for space in value):
            raise ValueError("label_spaces must name at least one non-blank space")
        return value

    @field_validator("id", "ru", "kk", "en", "description")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("tactic fields (id/ru/kk/description) must not be blank")
        return value


class NegativeCategory(BaseModel):
    """A NOT-scam category (risk ~= 0) used to balance the corpus; may be a hard negative."""

    model_config = ConfigDict(frozen=True)

    id: str
    note: str
    hard_negative: bool
    # Which taxonomy generation introduced it: the v1 Gemini generator (`data/generate.py`)
    # only writes v1 categories, so regenerating the existing corpus stays reproducible.
    label_spaces: tuple[str, ...] = (_V1_SPACE,)

    @field_validator("id", "note")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("negative category fields (id/note) must not be blank")
        return value


class Taxonomy(BaseModel):
    """The full validated tactic label space plus negative categories."""

    model_config = ConfigDict(frozen=True)

    version: int
    active_label_space: str = _V1_SPACE
    tactics: tuple[TacticDefinition, ...]
    negatives: tuple[NegativeCategory, ...] = ()

    @model_validator(mode="after")
    def _at_least_one_tactic(self) -> "Taxonomy":
        if not self.tactics:
            raise ValueError("taxonomy must define at least one tactic")
        return self

    @model_validator(mode="after")
    def _unique_ids(self) -> "Taxonomy":
        _raise_if_duplicates([t.id for t in self.tactics], "tactic")
        _raise_if_duplicates([n.id for n in self.negatives], "negative category")
        return self

    @model_validator(mode="after")
    def _active_space_populated_and_successors_known(self) -> "Taxonomy":
        if not any(self.active_label_space in t.label_spaces for t in self.tactics):
            raise ValueError(f"active_label_space {self.active_label_space!r} has no tactics")
        known = {t.id for t in self.tactics}
        for tactic in self.tactics:
            missing = [s for s in tactic.superseded_by if s not in known]
            if missing:
                raise ValueError(f"tactic {tactic.id!r} is superseded by unknown id(s) {missing}")
        return self

    def label_space_names(self) -> tuple[str, ...]:
        """Every space named by some tactic, in first-seen order."""
        return tuple(dict.fromkeys(space for t in self.tactics for space in t.label_spaces))

    def label_space(self, name: str) -> tuple[str, ...]:
        """The ids of space `name`, in file order (the order a head trained on it uses)."""
        if name not in self.label_space_names():
            raise TaxonomyError(f"Unknown label space {name!r}; expected one of {self.label_space_names()}")
        return tuple(t.id for t in self.tactics if name in t.label_spaces)

    def active_tactics(self) -> tuple[TacticDefinition, ...]:
        """Definitions of the active label space (what the classifier is trained on)."""
        return tuple(t for t in self.tactics if self.active_label_space in t.label_spaces)

    def tactic_ids(self) -> tuple[str, ...]:
        """The classifier's label space: the ids of the active space, in file order."""
        return tuple(t.id for t in self.active_tactics())

    def all_tactic_ids(self) -> tuple[str, ...]:
        """Every id any space defines -- for display, advice and validating stored tags."""
        return tuple(t.id for t in self.tactics)

    def hard_signal_ids(self, space: str | None = None) -> tuple[str, ...]:
        """Hard-signal ids (the cue-feature columns) of `space`, default the active one."""
        ids = set(self.label_space(space or self.active_label_space))
        return tuple(t.id for t in self.tactics if t.hard_signal and t.id in ids)

    def negative_ids(self, space: str | None = None) -> tuple[str, ...]:
        """Every negative category id, or only those of `space` when given."""
        return tuple(n.id for n in self.negatives if space is None or space in n.label_spaces)

    def hard_negative_ids(self) -> tuple[str, ...]:
        return tuple(n.id for n in self.negatives if n.hard_negative)

    def get(self, tactic_id: str) -> TacticDefinition:
        """Return the `TacticDefinition` for `tactic_id`, or raise `KeyError`."""
        for tactic in self.tactics:
            if tactic.id == tactic_id:
                return tactic
        raise KeyError(f"Unknown tactic id: {tactic_id!r}")

    def display_name(self, tactic_id: str, locale: str) -> str:
        """Localized display name for `tactic_id` (one field per supported locale)."""
        if locale not in _SUPPORTED_LOCALES:
            raise TaxonomyError(
                f"Unsupported locale {locale!r}; expected one of {_SUPPORTED_LOCALES}"
            )
        tactic = self.get(tactic_id)
        return getattr(tactic, locale)  # field names are the locale codes


def _raise_if_duplicates(ids: list[str], label: str) -> None:
    seen: set[str] = set()
    for item_id in ids:
        if item_id in seen:
            raise ValueError(f"duplicate {label} id: {item_id!r}")
        seen.add(item_id)


def load_taxonomy(path: Path | None = None) -> Taxonomy:
    """Load and validate the taxonomy YAML at `path`.

    Defaults to the configured `taxonomy_path` (see `qorgan.config`). Raises
    `TaxonomyError` on any structural or validation failure -- fail fast at this
    boundary rather than let a malformed taxonomy silently corrupt downstream labels.
    """
    resolved_path = path or get_config().taxonomy_path
    if not resolved_path.exists():
        raise TaxonomyError(f"Taxonomy file not found: {resolved_path}")

    try:
        raw = yaml.safe_load(resolved_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise TaxonomyError(f"Invalid YAML in {resolved_path}: {exc}") from exc

    if not isinstance(raw, dict):
        raise TaxonomyError(
            f"Taxonomy file {resolved_path} must contain a YAML mapping at the top level"
        )

    try:
        taxonomy = Taxonomy.model_validate(raw)
    except ValueError as exc:
        raise TaxonomyError(f"Taxonomy validation failed for {resolved_path}: {exc}") from exc

    if taxonomy.version not in _SUPPORTED_TAXONOMY_VERSIONS:
        raise TaxonomyError(
            f"Unsupported taxonomy version {taxonomy.version} in {resolved_path}; "
            f"expected one of {_SUPPORTED_TAXONOMY_VERSIONS}"
        )
    return taxonomy


# Parsed taxonomies keyed by (resolved path, mtime_ns): `get_taxonomy()` is called from every
# feature computation (dozens of times per live turn), and re-parsing the YAML each time cost
# ~25 % of a streaming replay. The mtime key keeps edits and per-test paths correct.
_TAXONOMY_CACHE: dict[tuple[str, int, str | None], Taxonomy] = {}


def get_taxonomy() -> Taxonomy:
    """Convenience accessor: the taxonomy at the configured path, cached until the file changes.
    `QORGAN_LABEL_SPACE` (ADR D63 phase 2) swaps the active label space for this process."""
    cfg = get_config()
    path = cfg.taxonomy_path
    try:
        key = (str(path.resolve()), path.stat().st_mtime_ns, cfg.label_space_override)
    except OSError as exc:  # missing file: let load_taxonomy raise its own error
        raise TaxonomyError(f"Taxonomy file not found: {path}") from exc
    cached = _TAXONOMY_CACHE.get(key)
    if cached is None:
        cached = load_taxonomy(path)
        if cfg.label_space_override and cfg.label_space_override != cached.active_label_space:
            if cfg.label_space_override not in cached.label_space_names():
                raise TaxonomyError(
                    f"QORGAN_LABEL_SPACE={cfg.label_space_override!r} is not a label space of {path} "
                    f"({cached.label_space_names()})"
                )
            cached = Taxonomy.model_validate({**cached.model_dump(), "active_label_space": cfg.label_space_override})
        _TAXONOMY_CACHE[key] = cached
    return cached
