"""Loads and validates the scam-tactic taxonomy from `data/taxonomy/tactics.yaml`.

Exposes the tactic label space (multi-label; a call typically exhibits several tactics),
`hard_signal` flags (strong standalone fraud indicators), localized RU/KK display names,
and the negative/hard-negative categories used to balance the corpus and drive down FPR.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from qorgan.config import get_config

_SUPPORTED_TAXONOMY_VERSION = 1
_SUPPORTED_LOCALES = ("ru", "kk")


class TaxonomyError(ValueError):
    """Raised when `tactics.yaml` is missing, malformed, or fails validation."""


class TacticDefinition(BaseModel):
    """One scam-tactic label: id, localized names, description, and hard-signal flag."""

    model_config = ConfigDict(frozen=True)

    id: str
    ru: str
    kk: str
    description: str
    hard_signal: bool
    examples_ru: tuple[str, ...] = ()
    examples_kk: tuple[str, ...] = ()

    @field_validator("id", "ru", "kk", "description")
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

    def tactic_ids(self) -> tuple[str, ...]:
        return tuple(t.id for t in self.tactics)

    def hard_signal_ids(self) -> tuple[str, ...]:
        return tuple(t.id for t in self.tactics if t.hard_signal)

    def negative_ids(self) -> tuple[str, ...]:
        return tuple(n.id for n in self.negatives)

    def hard_negative_ids(self) -> tuple[str, ...]:
        return tuple(n.id for n in self.negatives if n.hard_negative)

    def get(self, tactic_id: str) -> TacticDefinition:
        """Return the `TacticDefinition` for `tactic_id`, or raise `KeyError`."""
        for tactic in self.tactics:
            if tactic.id == tactic_id:
                return tactic
        raise KeyError(f"Unknown tactic id: {tactic_id!r}")

    def display_name(self, tactic_id: str, locale: str) -> str:
        """Localized (RU/KK) display name for `tactic_id`."""
        if locale not in _SUPPORTED_LOCALES:
            raise TaxonomyError(
                f"Unsupported locale {locale!r}; expected one of {_SUPPORTED_LOCALES}"
            )
        tactic = self.get(tactic_id)
        return tactic.ru if locale == "ru" else tactic.kk


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

    if taxonomy.version != _SUPPORTED_TAXONOMY_VERSION:
        raise TaxonomyError(
            f"Unsupported taxonomy version {taxonomy.version} in {resolved_path}; "
            f"expected {_SUPPORTED_TAXONOMY_VERSION}"
        )
    return taxonomy


# Parsed taxonomies keyed by (resolved path, mtime_ns): `get_taxonomy()` is called from every
# feature computation (dozens of times per live turn), and re-parsing the YAML each time cost
# ~25 % of a streaming replay. The mtime key keeps edits and per-test paths correct.
_TAXONOMY_CACHE: dict[tuple[str, int], Taxonomy] = {}


def get_taxonomy() -> Taxonomy:
    """Convenience accessor: the taxonomy at the configured path, cached until the file changes."""
    path = get_config().taxonomy_path
    try:
        key = (str(path.resolve()), path.stat().st_mtime_ns)
    except OSError as exc:  # missing file: let load_taxonomy raise its own error
        raise TaxonomyError(f"Taxonomy file not found: {path}") from exc
    cached = _TAXONOMY_CACHE.get(key)
    if cached is None:
        cached = _TAXONOMY_CACHE[key] = load_taxonomy(path)
    return cached
