"""Test credentials for the analyst console (`/api/admin`) and the audit-chain key.

`tests/conftest.py` installs these for every test, so the suite never depends on a
developer's `.env` (which may hold real local keys) and CI without a `.env` behaves the same.
"""

from __future__ import annotations

ANALYST_ID = "aigerim"
ANALYST_KEY = "tests-analyst-key-0123456789abcdef"
INVESTIGATOR_ID = "bek"
INVESTIGATOR_KEY = "tests-investigator-key-0123456789abcdef"
ANALYST_KEYS_ENV = f"{ANALYST_ID}:{ANALYST_KEY}:analyst,{INVESTIGATOR_ID}:{INVESTIGATOR_KEY}:investigator"
AUDIT_CHAIN_KEY = "tests-only-audit-chain-key-0123456789abcdef"
KEY_HEADER = "X-Analyst-Key"


def as_analyst() -> dict[str, str]:
    return {KEY_HEADER: ANALYST_KEY}


def as_investigator() -> dict[str, str]:
    return {KEY_HEADER: INVESTIGATOR_KEY}
