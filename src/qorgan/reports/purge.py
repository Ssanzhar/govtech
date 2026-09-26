"""Retention: forget consented reports older than `QORGAN_REPORT_RETENTION_DAYS` (C3).

Age runs on the server's clock (`reports.retention`: `received_at`, never the client's
`timestamp`). Each expired report goes through `analytics.intake.forget_report`, so an
incident already folded into the analysis is removed as well and the organizations are
recomputed. The server runs this itself at startup and every
`QORGAN_REPORT_PURGE_INTERVAL_HOURS` (`PurgeSchedule`, started by `qorgan.api`); with the
interval at 0, schedule `python -m qorgan.reports.purge [--retention-days N] [--dry-run]` with
cron instead. Idempotent.
"""

from __future__ import annotations

import argparse
import logging
import threading
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

from qorgan.analytics.intake import forget_report
from qorgan.config import Config, get_config
from qorgan.reports.retention import is_expired
from qorgan.reports.store import REPORTS_FILENAME, REPORTS_LOCK, load_reports

_LOGGER = logging.getLogger(__name__)


def expired_receipts(reports_path: Path, *, retention_days: int, now: datetime) -> list[str]:
    if retention_days <= 0:
        raise ValueError(f"retention_days must be > 0, got {retention_days}")
    return [r.receipt_id for r in load_reports(reports_path) if is_expired(r, now=now, retention_days=retention_days)]


def purge(
    *,
    retention_days: int,
    now: datetime,
    reports_path: Path,
    incidents_path: Path,
    organizations_path: Path,
    embeddings_path: Path,
) -> list[str]:
    """Forget every expired report everywhere; returns the receipts removed."""
    removed: list[str] = []
    for receipt_id in expired_receipts(reports_path, retention_days=retention_days, now=now):
        if forget_report(
            receipt_id,
            reports_path=reports_path,
            incidents_path=incidents_path,
            organizations_path=organizations_path,
            embeddings_path=embeddings_path,
            now=now.replace(tzinfo=None),
        ):
            removed.append(receipt_id)
    return removed


def purge_configured(cfg: Config, *, now: datetime) -> list[str]:
    """One purge over the configured data dir, serialised with every other reports write."""
    processed = cfg.data_dir / "processed"
    with REPORTS_LOCK:
        return purge(
            retention_days=cfg.report_retention_days,
            now=now,
            reports_path=processed / REPORTS_FILENAME,
            incidents_path=processed / "incidents.jsonl",
            organizations_path=processed / "organizations.jsonl",
            embeddings_path=processed / "incident_embeddings.npz",
        )


class PurgeSchedule:
    """The server's own retention job: purge once now, then every `interval_hours` on a daemon
    thread until `stop()`. A failed run is logged (counts only, never content) and retried at
    the next tick -- it never takes the server down."""

    def __init__(self, cfg: Config) -> None:
        self._cfg = cfg
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> list[str]:
        removed = self._run_once()
        interval = self._cfg.report_purge_interval_hours * 3600
        if interval > 0:  # 0 = a single run; the operator schedules the CLI instead
            self._thread = threading.Thread(target=self._loop, args=(interval,), name="report-purge", daemon=True)
            self._thread.start()
        return removed

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)

    def _loop(self, interval_seconds: float) -> None:
        while not self._stop.wait(interval_seconds):
            self._run_once()

    def _run_once(self) -> list[str]:
        try:
            removed = purge_configured(self._cfg, now=datetime.now(UTC))
        except Exception as exc:  # noqa: BLE001 -- a retention hiccup must not crash the server
            _LOGGER.error("report purge failed (%s); retrying at the next tick", type(exc).__name__)
            return []
        if removed:
            _LOGGER.info("report purge: %d report(s) past retention forgotten", len(removed))
        return removed


def main(argv: Sequence[str] | None = None) -> None:  # pragma: no cover - CLI
    cfg = get_config()
    parser = argparse.ArgumentParser(description="Forget consented reports past their retention window.")
    parser.add_argument("--retention-days", type=int, default=cfg.report_retention_days)
    parser.add_argument("--dry-run", action="store_true", help="List what would be removed")
    args = parser.parse_args(argv)
    processed = cfg.data_dir / "processed"
    now = datetime.now(UTC)
    if args.dry_run:
        for receipt_id in expired_receipts(processed / REPORTS_FILENAME, retention_days=args.retention_days, now=now):
            print(receipt_id)
        return
    removed = purge(
        retention_days=args.retention_days,
        now=now,
        reports_path=processed / REPORTS_FILENAME,
        incidents_path=processed / "incidents.jsonl",
        organizations_path=processed / "organizations.jsonl",
        embeddings_path=processed / "incident_embeddings.npz",
    )
    print(f"purged {len(removed)} report(s) older than {args.retention_days} days")


if __name__ == "__main__":  # pragma: no cover
    main()
