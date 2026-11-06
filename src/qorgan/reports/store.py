"""Reports file operations: prepare (minimise) -> append -> load -> remove / purge.

One JSON line per `StoredReport`. Every write path goes through `prepare_report`, which is
where the raw transcript and caller number are reduced to what may be stored (PLAN_2026-09
C2), and every report carries a receipt so the citizen can delete it (C3). Pure over the
file it is given; the default path is resolved by callers from config.
"""

from __future__ import annotations

import logging
import secrets
import threading
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import IO

from qorgan.config import get_config
from qorgan.data.scrub import STORED_RECORD_CONTEXT, scrub_until_stable
from qorgan.privacy.numbers import display_prefix, hash_phone_number
from qorgan.data.span_grounding import ground_flagged_phrases
from qorgan.reports.model import CITIZEN_CONSENT_BASIS, ReportSource, StoredReport
from qorgan.reports.retention import is_expired
from qorgan.atomic import write_text_atomic
from qorgan.jsonl import read_records

try:  # POSIX: also exclude other processes (see `_ReportsLock`)
    import fcntl
except ImportError:  # pragma: no cover - Windows dev boxes: the in-process lock still holds
    fcntl = None  # type: ignore[assignment]

# Citizen and partner reports share one file under `<data_dir>/processed/`; the `source`
# field tells them apart.
REPORTS_FILENAME = "citizen_reports.jsonl"
_RECEIPT_BYTES = 12  # 24 hex chars, matches RECEIPT_ID_PATTERN
_LOCK_FILENAME = ".reports.lock"
_LOGGER = logging.getLogger(__name__)


class _ReportsLock:
    """Serialises every write to the report stores. Re-entrant, so a caller can hold it around a
    read-modify-write that itself calls these helpers. A thread lock covers the server's request
    threads and scheduled purge; a file lock covers other processes (a cron purge, an ops script),
    whose appends were otherwise lost when the whole file was rewritten."""

    def __init__(self) -> None:
        self._thread = threading.RLock()
        self._depth = 0
        self._handle: IO[str] | None = None

    def __enter__(self) -> "_ReportsLock":
        self._thread.acquire()
        try:
            if self._depth == 0 and fcntl is not None:
                self._handle = self._lock_file()
        except BaseException:  # `__exit__` does not run when `__enter__` raises
            self._thread.release()
            raise
        self._depth += 1
        return self

    def __exit__(self, *_exc: object) -> None:
        self._depth -= 1
        if self._depth == 0 and self._handle is not None:
            self._handle.close()  # closing releases the lock
            self._handle = None
        self._thread.release()

    @staticmethod
    def _lock_file() -> IO[str] | None:
        processed = get_config().data_dir / "processed"
        handle = None
        try:
            processed.mkdir(parents=True, exist_ok=True)
            handle = (processed / _LOCK_FILENAME).open("a", encoding="utf-8")
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            return handle
        except OSError as exc:  # a read-only data dir, a volume without locks: the thread lock remains
            _LOGGER.warning("no file lock for the report stores (%s)", type(exc).__name__)
            if handle is not None:
                handle.close()
            return None


REPORTS_LOCK = _ReportsLock()


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
    received_at: datetime | None = None,
    source: ReportSource = "citizen",
    consent_basis: str = CITIZEN_CONSENT_BASIS,
    consent_version: str | None = None,
    receipt_id: str | None = None,
    partner_id: str | None = None,
    partner_reference: str | None = None,
) -> StoredReport:
    """Reduce a reviewed draft to its storable form.

    - transcript -> `scrub_text` (emails, cards, IINs, phone numbers redacted);
    - flagged phrases -> `ground_flagged_phrases`: verbatim in the scrubbed transcript, as-is
      or in their own scrubbed form (ADR D66), never a bare placeholder;
    - phone number -> HMAC digest + display prefix (raises `MissingHmacKeyError` without a
      key, `ValueError` for an unparseable number). A report without a number needs no key.

    `received_at` is the server's clock at storage (defaults to now, UTC): retention and quotas
    anchor on it, never on the client's `timestamp`.
    """
    scrubbed = scrub_until_stable(transcript)
    kept_phrases = ground_flagged_phrases(flagged_phrases, scrubbed, transcript)
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
        consent_version=consent_version,
        partner_id=partner_id,
        partner_reference=partner_reference,
        received_at=received_at or datetime.now(UTC),
    )


def append_report(report: StoredReport, path: Path) -> Path:
    with REPORTS_LOCK:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(report.model_dump_json() + "\n")
    return path


def load_reports(path: Path, *, skip_corrupt: bool = False) -> list[StoredReport]:
    """All stored reports; a missing file means none yet. Raises `ValueError` on a corrupt line,
    unless a deletion asks to go on without it (`jsonl.read_records`)."""
    if not path.exists():
        return []
    return read_records(
        path, lambda line: StoredReport.model_validate_json(line, context=STORED_RECORD_CONTEXT), skip_corrupt=skip_corrupt
    )


def _rewrite(reports: Sequence[StoredReport], path: Path) -> None:
    write_text_atomic(path, "".join(r.model_dump_json() + "\n" for r in reports))


def remove_report(receipt_id: str, path: Path) -> StoredReport | None:
    """Delete the report with `receipt_id` (rewriting the file); returns it, or `None`."""
    with REPORTS_LOCK:
        reports = load_reports(path, skip_corrupt=True)  # a broken line must not stop a deletion
        match = next((r for r in reports if r.receipt_id == receipt_id), None)
        if match is None:
            return None
        _rewrite([r for r in reports if r.receipt_id != receipt_id], path)
    return match


def purge_expired(path: Path, *, retention_days: int, now: datetime) -> list[StoredReport]:
    """Remove reports past retention (server-clock anchored, `reports.retention`); returns them."""
    if retention_days <= 0:
        raise ValueError(f"retention_days must be > 0, got {retention_days}")
    with REPORTS_LOCK:
        reports = load_reports(path)
        expired = [r for r in reports if is_expired(r, now=now, retention_days=retention_days)]
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
