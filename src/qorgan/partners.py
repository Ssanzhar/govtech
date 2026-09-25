"""Partner credential registry for the partner intake API (PLAN_2026-09 C5, ADR D19).

Credentials come from one environment variable, `QORGAN_PARTNER_API_KEYS`, parsed once at
startup: `partner_id:secret[:daily_quota]` entries separated by commas. Parsing fails fast
on anything weak or ambiguous (short secret, duplicate id, duplicate secret) so a
misconfigured server refuses to start rather than opening a half-configured ingress.
Secrets are compared in constant time and never appear in reprs or logs.
"""

from __future__ import annotations

import hmac
import re
from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict, Field, SecretStr

PARTNER_ID_PATTERN = r"^[a-z][a-z0-9_-]{1,31}$"
MIN_SECRET_CHARS = 16
DEFAULT_PARTNER_DAILY_QUOTA = 200
_ENTRY_SEPARATOR = ","
_FIELD_SEPARATOR = ":"
_PARTNER_ID_RE = re.compile(PARTNER_ID_PATTERN)


class PartnerCredential(BaseModel):
    """One configured partner: its machine id, shared secret and daily report budget."""

    model_config = ConfigDict(frozen=True)

    id: str = Field(pattern=PARTNER_ID_PATTERN)
    secret: SecretStr = Field(min_length=MIN_SECRET_CHARS)
    daily_quota: int = Field(gt=0)


class Partner(BaseModel):
    """An authenticated partner as the API sees it (no secret)."""

    model_config = ConfigDict(frozen=True)

    id: str
    daily_quota: int = Field(gt=0)


def parse_partner_credentials(
    raw: str, *, default_quota: int = DEFAULT_PARTNER_DAILY_QUOTA
) -> tuple[PartnerCredential, ...]:
    """Parse `id:secret[:daily_quota],...`; blank means no partners. Raises `ValueError`."""
    entries = [part.strip() for part in raw.split(_ENTRY_SEPARATOR) if part.strip()]
    credentials = tuple(_parse_entry(entry, default_quota) for entry in entries)
    _reject_duplicates(credentials)
    return credentials


def _parse_entry(entry: str, default_quota: int) -> PartnerCredential:
    fields = entry.split(_FIELD_SEPARATOR)
    if len(fields) not in (2, 3):
        raise ValueError("partner credential must be 'partner_id:secret' or 'partner_id:secret:daily_quota'")
    partner_id, secret = fields[0].strip(), fields[1].strip()
    if not _PARTNER_ID_RE.fullmatch(partner_id):
        raise ValueError(f"partner id {partner_id!r} must match {PARTNER_ID_PATTERN}")
    if len(secret) < MIN_SECRET_CHARS:
        raise ValueError(f"partner {partner_id!r}: secret must be at least {MIN_SECRET_CHARS} characters")
    quota = default_quota
    if len(fields) == 3:
        try:
            quota = int(fields[2].strip())
        except ValueError as exc:
            raise ValueError(f"partner {partner_id!r}: daily quota must be an integer") from exc
        if quota <= 0:
            raise ValueError(f"partner {partner_id!r}: daily quota must be > 0")
    return PartnerCredential(id=partner_id, secret=SecretStr(secret), daily_quota=quota)


def _reject_duplicates(credentials: Sequence[PartnerCredential]) -> None:
    ids = [c.id for c in credentials]
    if len(set(ids)) != len(ids):
        raise ValueError("partner ids must be unique")
    secrets = [c.secret.get_secret_value() for c in credentials]
    if len(set(secrets)) != len(secrets):
        raise ValueError("partner secrets must be unique (a shared secret cannot identify a partner)")


class PartnerRegistry:
    """Constant-time secret lookup over the configured credentials."""

    def __init__(self, credentials: Sequence[PartnerCredential]) -> None:
        self._credentials = tuple(credentials)

    @property
    def is_empty(self) -> bool:
        return not self._credentials

    def authenticate(self, api_key: str | None) -> Partner | None:
        """The partner whose secret equals `api_key`, or None. Every secret is compared so
        timing does not reveal which (if any) matched."""
        if not api_key:
            return None
        presented = api_key.encode("utf-8")
        matched: Partner | None = None
        for credential in self._credentials:
            if hmac.compare_digest(presented, credential.secret.get_secret_value().encode("utf-8")):
                matched = Partner(id=credential.id, daily_quota=credential.daily_quota)
        return matched
