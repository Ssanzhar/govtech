"""Retention: forget consented reports older than `QORGAN_REPORT_RETENTION_DAYS` (C3).

Run: `python -m qorgan.reports.purge [--retention-days N] [--dry-run]`. Each expired report
goes through `analytics.intake.forget_report`, so an incident already folded into the
analysis is removed as well and the organizations are recomputed. Cron-friendly; idempotent.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path

from qorgan.analytics.intake import forget_report
from qorgan.config import get_config
from qorgan.reports.store import REPORTS_FILENAME, as_aware, load_reports


def expired_receipts(reports_path: Path, *, retention_days: int, now: datetime) -> list[str]:
    if retention_days <= 0:
        raise ValueError(f"retention_days must be > 0, got {retention_days}")
    cutoff = now - timedelta(days=retention_days)
    return [r.receipt_id for r in load_reports(reports_path) if as_aware(r.timestamp, now) < cutoff]


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
