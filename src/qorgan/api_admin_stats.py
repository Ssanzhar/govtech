"""Analyst results/statistics HTTP API — the dashboard's "how is the work going" view.

Thin wrapper over the pure `analytics.stats` helpers plus the same `_load_analysis`
degrade point the rest of the admin API uses: absent or corrupt analysis data yields
`available: false`, never a 500.
"""

from __future__ import annotations

from datetime import date, datetime

from fastapi import APIRouter
from pydantic import BaseModel

from qorgan.analytics.intake import load_report_drafts, report_incident_id
from qorgan.analytics.presentation import org_display_name
from qorgan.analytics.stats import activity_series, weekly_trend
from qorgan.api_admin import Locale, _load_analysis
from qorgan.config import get_config
from qorgan.reports.store import REPORTS_FILENAME

router = APIRouter(prefix="/api/admin", tags=["admin"])

_ACTIVITY_DAYS = 30
_TOP_ORGANIZATIONS = 6


class ActivityPointOut(BaseModel):
    date: str
    count: int


class TrendOut(BaseModel):
    this_week: int
    last_week: int
    delta_pct: float | None


class TopOrgOut(BaseModel):
    id: str
    name: str
    incidents: int
    is_novel: bool


class ReportCountsOut(BaseModel):
    submitted: int
    ingested: int  # folded into an incident
    pending: int  # has a transcript, not yet ingested
    signals_only: int  # partner reports without a transcript: stored, not clustered (PLAN C9)


class StatsResponse(BaseModel):
    available: bool
    activity: list[ActivityPointOut]
    trend: TrendOut
    top_organizations: list[TopOrgOut]
    reports: ReportCountsOut
    novel_schemes: int


_EMPTY_STATS = StatsResponse(
    available=False,
    activity=[],
    trend=TrendOut(this_week=0, last_week=0, delta_pct=None),
    top_organizations=[],
    reports=ReportCountsOut(submitted=0, ingested=0, pending=0, signals_only=0),
    novel_schemes=0,
)


@router.get("/stats", response_model=StatsResponse)
def stats(locale: Locale = "ru") -> StatsResponse:
    analysis = _load_analysis()
    if analysis is None:
        return _EMPTY_STATS

    now = datetime.now()
    series = activity_series(analysis.incidents, days=_ACTIVITY_DAYS, today=date.today())
    this_week, last_week, delta_pct = weekly_trend(analysis.incidents, now=now)

    by_id = {incident.id: incident for incident in analysis.incidents}
    ranked = sorted(analysis.organizations, key=lambda org: -len(org.members))
    top = [
        TopOrgOut(
            id=org.id,
            name=org_display_name(org, by_id, locale=locale),
            incidents=len(org.members),
            is_novel=org.is_novel,
        )
        for org in ranked[:_TOP_ORGANIZATIONS]
    ]

    return StatsResponse(
        available=True,
        activity=[ActivityPointOut(date=d.isoformat(), count=c) for d, c in series],
        trend=TrendOut(this_week=this_week, last_week=last_week, delta_pct=delta_pct),
        top_organizations=top,
        reports=_report_counts(by_id, pending=analysis.pending),
        novel_schemes=sum(1 for org in analysis.organizations if org.is_novel),
    )


def _report_counts(incidents_by_id: dict, *, pending: int) -> ReportCountsOut:
    """Distinct reports (citizen + partner, by content-addressed id) split by what
    happened to them; `ingested` is actual incident membership, never `submitted - pending`."""
    reports_path = get_config().data_dir / "processed" / REPORTS_FILENAME
    try:
        drafts = load_report_drafts(reports_path)
    except ValueError:  # corrupt line — the stats view must not take the page down
        return ReportCountsOut(submitted=0, ingested=0, pending=0, signals_only=0)
    by_report_id = {report_incident_id(draft): draft for draft in drafts}
    ingested = sum(1 for rid in by_report_id if rid in incidents_by_id)
    signals_only = sum(1 for draft in by_report_id.values() if draft.is_signals_only)
    return ReportCountsOut(submitted=len(by_report_id), ingested=ingested, pending=pending, signals_only=signals_only)
