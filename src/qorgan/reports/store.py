"""Reports file operations: prepare (minimise) -> append -> load -> remove / purge.

One JSON line per `StoredReport`. Every write path goes through `prepare_report`, which is
where the raw transcript and caller number are reduced to what may be stored (PLAN_2026-09
C2), and every report carries a receipt so the citizen can delete it (C3). Pure over the
file it is given; the default path is resolved by callers from config.
"""

from __future__ import annotations

import secrets
from collections.abc import Sequence
from datetime import datetime, timedelta
from pathlib import Path

from qorgan.data.scrub import scrub_text
from qorgan.privacy.numbers import display_prefix, hash_phone_number
from qorgan.reports.model import CITIZEN_CONSENT_BASIS, ReportSource, StoredReport

# Citizen and partner reports share one file under `<data_dir>/processed/`; the `source`
# field tells them apart.
REPORTS_FILENAME = "citizen_reports.jsonl"
_RECEIPT_BYTES = 12  # 24 hex chars, matches RECEIPT_ID_PATTERN


def new_receipt_id() -> str:
    return secrets.token_hex(_RECEIPT_BYTES)


def prepare_report(
    *,
    transcript: str,
    phone_number: str | None,
    flagged_phrases: Sequence[str],
    tactic_ids: Sequence[str],
    timestamp: datetime,
    risk_score: float,
    hmac_key: bytes | None,
    source: ReportSource = "citizen",
    consent_basis: str = CITIZEN_CONSENT_BASIS,
    receipt_id: str | None = None,
    partner_id: str | None = None,
    partner_reference: str | None = None,
    received_at: datetime | None = None,
) -> StoredReport:
    """Reduce a reviewed draft to its storable form.

    - transcript -> `scrub_text` (emails, cards, IINs, phone numbers redacted);
    - flagged phrases -> only those still verbatim in the scrubbed transcript;
    - phone number -> HMAC digest + display prefix (raises `MissingHmacKeyError` without a
      key, `ValueError` for an unparseable number). A report without a number needs no key.
    """
    scrubbed = scrub_text(transcript)
    kept_phrases = tuple(p for p in flagged_phrases if p and p in scrubbed)
    number = (phone_number or "").strip() or None
    return StoredReport(
        receipt_id=receipt_id or new_receipt_id(),
        transcript=scrubbed,
        number_hash=hash_phone_number(number, key=hmac_key) if number else None,
        number_prefix=display_prefix(number) if number else None,
        flagged_phrases=kept_phrases,
        tactic_ids=tuple(tactic_ids),
        timestamp=timestamp,
        risk_score=risk_score,
        source=source,
        consent_basis=consent_basis,
        partner_id=partner_id,
        partner_reference=partner_reference,
        received_at=received_at,
    )


def append_report(report: StoredReport, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(report.model_dump_json() + "\n")
    return path


def load_reports(path: Path) -> list[StoredReport]:
    """All stored reports; a missing file means none yet. Raises `ValueError` on a corrupt line."""
    if not path.exists():
        return []
    return [
        StoredReport.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _rewrite(reports: Sequence[StoredReport], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(r.model_dump_json() + "\n" for r in reports), encoding="utf-8")


def remove_report(receipt_id: str, path: Path) -> StoredReport | None:
    """Delete the report with `receipt_id` (rewriting the file); returns it, or `None`."""
    reports = load_reports(path)
    match = next((r for r in reports if r.receipt_id == receipt_id), None)
    if match is None:
        return None
    _rewrite([r for r in reports if r.receipt_id != receipt_id], path)
    return match


def purge_expired(path: Path, *, retention_days: int, now: datetime) -> list[StoredReport]:
    """Remove reports whose timestamp is older than `retention_days` before `now`; returns them."""
    if retention_days <= 0:
        raise ValueError(f"retention_days must be > 0, got {retention_days}")
    reports = load_reports(path)
    cutoff = now - timedelta(days=retention_days)
    expired = [r for r in reports if as_aware(r.timestamp, now) < cutoff]
    if expired:
        _rewrite([r for r in reports if r not in expired], path)
    return expired


def as_aware(stamp: datetime, reference: datetime) -> datetime:
    """Compare naive and aware stamps safely by adopting the reference's tzinfo for naive ones."""
    if stamp.tzinfo is None and reference.tzinfo is not None:
        return stamp.replace(tzinfo=reference.tzinfo)
    if stamp.tzinfo is not None and reference.tzinfo is None:
        return stamp.replace(tzinfo=None)
    return stamp
