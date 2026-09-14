"""Inspection ledger for the authored held-out set (PLAN_2026-09 A2).

`authored_heldout` is hand-written by the team, and some of its records were *read* while
engineering features (the reassurance and KK-boundary fixes were motivated by inspecting
specific false positives). Those records can no longer serve as an unbiased generalization
signal. The ledger is the explicit, reviewable list of which ids were inspected, when, and
why; the eval harness reports the clean and inspected subsets separately.

Rule: any authored anchor a developer reads while debugging the model gets an entry here
in the same change. Deleting entries is never the fix.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator


class LedgerError(ValueError):
    """The ledger file is missing or malformed."""


class LedgerEntry(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str = Field(min_length=1)
    inspected_on: date
    reason: str = Field(min_length=1)

    @field_validator("reason")
    @classmethod
    def _reason_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("reason must not be blank")
        return value


class InspectionLedger(BaseModel):
    model_config = ConfigDict(frozen=True)

    version: int = Field(ge=1)
    entries: tuple[LedgerEntry, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _ids_unique(self) -> "InspectionLedger":
        ids = [entry.id for entry in self.entries]
        duplicates = sorted({i for i in ids if ids.count(i) > 1})
        if duplicates:
            raise ValueError(f"duplicate ledger ids: {duplicates}")
        return self

    @property
    def ids(self) -> frozenset[str]:
        return frozenset(entry.id for entry in self.entries)


def load_inspection_ledger(path: Path | None = None) -> InspectionLedger:
    """Load and validate the ledger YAML (default: `config.inspection_ledger_path`).

    Raises `LedgerError` if the file is missing, unparseable, or fails validation --
    an absent or empty ledger is a mistake to surface, not a state to silently accept.
    """
    from qorgan.config import get_config

    active = path or get_config().inspection_ledger_path
    if not active.exists():
        raise LedgerError(f"inspection ledger not found: {active}")
    try:
        raw = yaml.safe_load(active.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise LedgerError(f"inspection ledger is not valid YAML: {active}: {exc}") from exc
    if not isinstance(raw, dict):
        raise LedgerError(f"inspection ledger must be a mapping with 'version' and 'entries': {active}")
    try:
        return InspectionLedger.model_validate(raw)
    except ValidationError as exc:
        raise LedgerError(f"inspection ledger failed validation: {active}: {exc}") from exc
