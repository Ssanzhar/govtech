"""Test helpers for hashed caller numbers (ADR D14): a fixed key + shortcuts."""

from __future__ import annotations

from datetime import UTC, datetime

from qorgan.privacy.numbers import display_prefix, hash_phone_number
from qorgan.reports.model import StoredReport
from qorgan.reports.store import prepare_report

TEST_HMAC_KEY = b"tests-only-hmac-key"


def hashed(number: str | None) -> str | None:
    return hash_phone_number(number, key=TEST_HMAC_KEY) if number else None


def prefix(number: str | None) -> str | None:
    return display_prefix(number) if number else None


def stored_report(
    *,
    number: str | None,
    transcript: str = "алло переведите деньги на безопасный счёт",
    flagged_phrases: tuple[str, ...] = ("переведите деньги на безопасный счёт",),
    tactic_ids: tuple[str, ...] = ("safe_account",),
    timestamp: datetime = datetime(2026, 7, 15, 10, 0, tzinfo=UTC),
    risk_score: float = 84.0,
    receipt_id: str | None = None,
) -> StoredReport:
    return prepare_report(
        transcript=transcript,
        phone_number=number,
        flagged_phrases=flagged_phrases,
        tactic_ids=tactic_ids,
        timestamp=timestamp,
        risk_score=risk_score,
        hmac_key=TEST_HMAC_KEY,
        receipt_id=receipt_id,
    )
