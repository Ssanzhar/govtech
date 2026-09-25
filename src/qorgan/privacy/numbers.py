"""Phone-number minimisation (PLAN_2026-09 C2, ADR D14).

The analyst layer links reports that share a caller number; it never needs the number.
So a number is reduced, at the moment it is received, to:
- a salted **HMAC-SHA256 digest** (`hash_phone_number`) -- stable per deployment key, so
  linking works, but not reversible without the key; and
- a **display prefix** (`display_prefix`, e.g. `+7 700 ***`) -- what an analyst may see.

The raw number is never persisted; `Incident` refuses anything that is not a digest.
Pure functions, no I/O. The key comes from `QORGAN_NUMBER_HMAC_KEY` (see `config.py`).
"""

from __future__ import annotations

import hashlib
import hmac
import re

# 96 bits of the digest: unguessable, compact enough for ids/tables.
_HASH_HEX_CHARS = 24
NUMBER_HASH_PATTERN = r"^[0-9a-f]{24}$"
_HASH_RE = re.compile(NUMBER_HASH_PATTERN)

_KZ_COUNTRY_CODE = "7"
_KZ_TRUNK_PREFIX = "8"
_KZ_NATIONAL_DIGITS = 11  # country code + 10-digit subscriber number
_KZ_SUBSCRIBER_DIGITS = 10
_MIN_DIGITS = 7
_KZ_OPERATOR_CODE_DIGITS = 3
_NON_KZ_PREFIX_DIGITS = 3


class MissingHmacKeyError(RuntimeError):
    """No HMAC key is configured -- numbers cannot be minimised, so they must not be stored."""


def normalize_phone_number(raw: str) -> str | None:
    """Canonical digit string for `raw`, or `None` if it is not a plausible number.

    Kazakh forms collapse to `7XXXXXXXXXX`: a leading trunk `8` becomes `7`, and a bare
    10-digit subscriber number gets the country code. Other countries pass through as
    their digits. Fewer than 7 digits is not a phone number.
    """
    digits = re.sub(r"\D", "", raw or "")
    if len(digits) < _MIN_DIGITS:
        return None
    if len(digits) == _KZ_SUBSCRIBER_DIGITS:
        return _KZ_COUNTRY_CODE + digits
    if len(digits) == _KZ_NATIONAL_DIGITS and digits[0] == _KZ_TRUNK_PREFIX:
        return _KZ_COUNTRY_CODE + digits[1:]
    return digits


def hash_phone_number(raw: str, *, key: bytes | None) -> str:
    """Salted HMAC-SHA256 digest (24 hex chars) of the normalised number.

    Raises `MissingHmacKeyError` for an absent/empty key and `ValueError` when `raw` does
    not normalise to a number -- callers must never fall back to storing the raw value.
    """
    if not key:
        raise MissingHmacKeyError(
            "QORGAN_NUMBER_HMAC_KEY is not set; refusing to handle a phone number without hashing it"
        )
    normalized = normalize_phone_number(raw)
    if normalized is None:
        raise ValueError(f"not a phone number: {raw!r}")
    return hmac.new(key, normalized.encode("ascii"), hashlib.sha256).hexdigest()[:_HASH_HEX_CHARS]


def is_number_hash(value: str) -> bool:
    return bool(_HASH_RE.fullmatch(value or ""))


def display_prefix(raw: str) -> str | None:
    """`+7 700 ***` for a Kazakh number, `+380 ***` for others, `None` if not a number."""
    normalized = normalize_phone_number(raw)
    if normalized is None:
        return None
    if len(normalized) == _KZ_NATIONAL_DIGITS and normalized[0] == _KZ_COUNTRY_CODE:
        operator_code = normalized[1 : 1 + _KZ_OPERATOR_CODE_DIGITS]
        return f"+{_KZ_COUNTRY_CODE} {operator_code} ***"
    return f"+{normalized[:_NON_KZ_PREFIX_DIGITS]} ***"
