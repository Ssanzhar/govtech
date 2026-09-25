"""Append-only, content-free audit log (PLAN_2026-09 C4/C5, ADR D19).

One JSON line per action by a partner or an analyst: *who* did *what* to *which subject*
with *which outcome* -- never call content, never a number. The schema enforces that
(`scrub_text` must be a fixed point on every text field), so a careless caller cannot turn
the audit log into a second copy of the data it is supposed to account for.
"""

from __future__ import annotations

from datetime import datetime
import re
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from qorgan.data.scrub import scrub_text

AUDIT_FILENAME = "audit_log.jsonl"
_MAX_FIELD_CHARS = 200

ActorKind = Literal["partner", "analyst", "system"]


class AuditEntry(BaseModel):
    model_config = ConfigDict(frozen=True)

    timestamp: datetime
    actor_kind: ActorKind
    actor_id: str = Field(min_length=1, max_length=_MAX_FIELD_CHARS)
    action: str = Field(min_length=1, max_length=_MAX_FIELD_CHARS)
    subject: str | None = Field(default=None, max_length=_MAX_FIELD_CHARS)
    outcome: str = Field(min_length=1, max_length=_MAX_FIELD_CHARS)

    @field_validator("actor_id", "action", "subject", "outcome")
    @classmethod
    def _content_free(cls, value: str | None) -> str | None:
        if value is not None and not _is_system_id(value) and scrub_text(value) != value:
            raise ValueError("audit fields must not carry numbers, cards, IINs or other call content")
        return value


# A system-generated opaque id is not call content, but its random hex can contain a digit run
# the PII scrubber reads as a phone number -- measured at ~0.46 % of receipts, i.e. one partner
# submission in ~217 losing its audit record (found 2026-09-24). These exact shapes are exempt;
# everything else still goes through the scrubber unchanged.
_SYSTEM_ID = re.compile(r"^(receipt|report|incident|organization):[0-9a-f]{8,64}$")


def _is_system_id(value: str) -> bool:
    return bool(_SYSTEM_ID.match(value))


def append_audit(entry: AuditEntry, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(entry.model_dump_json() + "\n")
    return path


def load_audit(path: Path) -> list[AuditEntry]:
    """All entries in order; a missing file means none yet. Raises `ValueError` on a corrupt line."""
    if not path.exists():
        return []
    return [
        AuditEntry.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
