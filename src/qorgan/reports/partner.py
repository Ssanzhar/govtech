"""Partner-side views over loaded reports (PLAN_2026-09 C5): the rolling quota count and
the idempotency lookup. Both are pure over the sequence they are given, so the API loads
the shared reports file once per request. The quota anchors on `received_at` -- the
server clock at storage time -- never on the partner-supplied `timestamp`/`occurred_at`,
which would let a backdated report escape the budget."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime

from qorgan.reports.model import StoredReport
from qorgan.reports.store import as_aware


def partner_reports_since(reports: Sequence[StoredReport], *, partner_id: str, since: datetime) -> int:
    """How many of `reports` `partner_id` stored (by server receipt time) at or after `since`."""
    return sum(
        1
        for report in reports
        if report.partner_id == partner_id and as_aware(report.received_at or report.timestamp, since) >= since
    )


def find_partner_report(reports: Sequence[StoredReport], *, partner_id: str, reference: str | None) -> StoredReport | None:
    """The report `partner_id` already filed under `reference`, if any (retries are no-ops)."""
    if reference is None:
        return None
    return next((r for r in reports if r.partner_id == partner_id and r.partner_reference == reference), None)
