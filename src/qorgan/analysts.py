"""Analyst credential registry for the Level-2 console (`/api/admin`, PLAN_2026-09 C4).

Mirrors the partner registry (`partners.py`): credentials come from one environment
variable, `QORGAN_ANALYST_KEYS`, parsed at startup as `analyst_id:secret:role` entries
separated by commas. Parsing fails fast on anything weak or ambiguous -- short secret,
duplicate id or secret, unknown role, an id the audit log would refuse -- so a misconfigured
server refuses to start rather than opening a half-configured console. Secrets are compared
in constant time and never appear in reprs, logs or error messages.

Roles, least privilege first (a role includes every role before it):
- `analyst`: the aggregates-first console -- overview, statistics, organization detail, the
  excerpt-only model analysis, confirm / dismiss / merge feedback, ingesting pending reports.
- `investigator`: additionally the audited "open case" that returns a full (scrubbed)
  transcript -- the one act that reveals a citizen's whole call.
"""

from __future__ import annotations

import hmac
import re
from collections.abc import Sequence
from typing import Literal, get_args

from pydantic import BaseModel, ConfigDict, Field, SecretStr

from qorgan.data.scrub import scrub_text

Role = Literal["analyst", "investigator"]
ROLES: tuple[str, ...] = get_args(Role)
ANALYST_ID_PATTERN = r"^[a-z][a-z0-9_.-]{1,31}$"
MIN_SECRET_CHARS = 16
# Ids the console itself writes into the audit log; a real analyst must never share them.
RESERVED_IDS = frozenset({"anonymous-analyst", "unauthenticated", "system"})
_ENTRY_SEPARATOR = ","
_FIELD_SEPARATOR = ":"
_ANALYST_ID_RE = re.compile(ANALYST_ID_PATTERN)


class AnalystCredential(BaseModel):
    """One configured analyst: machine id, shared secret, role."""

    model_config = ConfigDict(frozen=True)

    id: str = Field(pattern=ANALYST_ID_PATTERN)
    secret: SecretStr = Field(min_length=MIN_SECRET_CHARS)
    role: Role


class Analyst(BaseModel):
    """An authenticated analyst as the API sees it (no secret)."""

    model_config = ConfigDict(frozen=True)

    id: str
    role: Role

    def has_role(self, role: Role) -> bool:
        return ROLES.index(self.role) >= ROLES.index(role)


def parse_analyst_credentials(raw: str) -> tuple[AnalystCredential, ...]:
    """Parse `id:secret:role,...`; blank means no analysts (the console is closed). Raises
    `ValueError` naming the offending analyst id, never the secret."""
    entries = [part.strip() for part in raw.split(_ENTRY_SEPARATOR) if part.strip()]
    credentials = tuple(_parse_entry(entry) for entry in entries)
    _reject_duplicates(credentials)
    return credentials


def _parse_entry(entry: str) -> AnalystCredential:
    fields = [field.strip() for field in entry.split(_FIELD_SEPARATOR)]
    if len(fields) != 3:
        raise ValueError("analyst credential must be 'analyst_id:secret:role' (the role is mandatory)")
    analyst_id, secret, role = fields
    if not _ANALYST_ID_RE.fullmatch(analyst_id):
        raise ValueError(f"analyst id {analyst_id!r} must match {ANALYST_ID_PATTERN}")
    if analyst_id in RESERVED_IDS:
        raise ValueError(f"analyst id {analyst_id!r} is reserved")
    if scrub_text(analyst_id) != analyst_id:
        raise ValueError(f"analyst id {analyst_id!r} looks like personal data (a phone number?) and would be refused by the audit log")
    if len(secret) < MIN_SECRET_CHARS:
        raise ValueError(f"analyst {analyst_id!r}: secret must be at least {MIN_SECRET_CHARS} characters")
    if role not in ROLES:
        raise ValueError(f"analyst {analyst_id!r}: role must be one of {ROLES}")
    return AnalystCredential(id=analyst_id, secret=SecretStr(secret), role=role)


def _reject_duplicates(credentials: Sequence[AnalystCredential]) -> None:
    ids = [c.id for c in credentials]
    if len(set(ids)) != len(ids):
        raise ValueError("analyst ids must be unique")
    secrets = [c.secret.get_secret_value() for c in credentials]
    if len(set(secrets)) != len(secrets):
        raise ValueError("analyst secrets must be unique (a shared secret cannot identify an analyst)")


class AnalystRegistry:
    """Constant-time secret lookup over the configured credentials."""

    def __init__(self, credentials: Sequence[AnalystCredential]) -> None:
        self._credentials = tuple(credentials)

    @property
    def is_empty(self) -> bool:
        return not self._credentials

    def authenticate(self, api_key: str | None) -> Analyst | None:
        """The analyst whose secret equals `api_key`, or None. Every secret is compared so
        timing does not reveal which (if any) matched."""
        if not api_key:
            return None
        presented = api_key.encode("utf-8")
        matched: Analyst | None = None
        for credential in self._credentials:
            if hmac.compare_digest(presented, credential.secret.get_secret_value().encode("utf-8")):
                matched = Analyst(id=credential.id, role=credential.role)
        return matched
