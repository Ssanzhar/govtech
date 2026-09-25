"""Partner export: `GET /api/v1/organizations` -- aggregates only (PLAN_2026-09 C5).

What a partner gets back is what an analyst's overview shows *minus* anything that could
identify a call or a caller: organization ids, localized tactic-derived names, incident
counts, tactic counts, last-activity dates, novelty and priority. No numbers (not even
digests), no transcripts, no excerpts. Every export is audited. Kept separate from
`api_partner.py` so the ingress module never imports the analytics read paths.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from qorgan.analytics.pipeline import load_organizations_jsonl
from qorgan.analytics.presentation import last_activity, org_display_name, org_tactic_profile
from qorgan.api_partner import record_partner_action, require_partner
from qorgan.config import get_config
from qorgan.data.incident_seed import load_incidents_jsonl
from qorgan.data.schema import Incident, Organization
from qorgan.partners import Partner

router = APIRouter(prefix="/api/v1", tags=["partner"])

Locale = Literal["ru", "kk"]


class TacticCountOut(BaseModel):
    id: str
    count: int


class OrganizationAggregateOut(BaseModel):
    id: str
    name: str
    incidents: int
    tactics: list[TacticCountOut]
    last_activity: str | None
    is_novel: bool
    priority: float


class OrganizationsExport(BaseModel):
    available: bool
    generated_at: datetime
    organizations: list[OrganizationAggregateOut]


@router.get("/organizations", response_model=OrganizationsExport)
def organizations(locale: Locale = "ru", partner: Partner = Depends(require_partner)) -> OrganizationsExport:
    cfg = get_config()
    now = datetime.now(UTC)
    loaded = _load_analysis()
    if loaded is None:
        record_partner_action(partner, "organizations.export", subject=None, outcome="unavailable", cfg=cfg, now=now)
        return OrganizationsExport(available=False, generated_at=now, organizations=[])

    orgs, incidents = loaded
    by_id = {incident.id: incident for incident in incidents}
    rows = [
        OrganizationAggregateOut(
            id=org.id,
            name=org_display_name(org, by_id, locale=locale),
            incidents=len(org.members),
            tactics=[TacticCountOut(id=tid, count=n) for tid, n in org_tactic_profile(org, by_id).items()],
            last_activity=_iso_date(last_activity(org, by_id)),
            is_novel=org.is_novel,
            priority=org.priority,
        )
        for org in orgs
    ]
    record_partner_action(partner, "organizations.export", subject=None, outcome="ok", cfg=cfg, now=now)
    return OrganizationsExport(available=True, generated_at=now, organizations=rows)


def _load_analysis() -> tuple[list[Organization], list[Incident]] | None:
    processed = get_config().data_dir / "processed"
    try:
        organizations = load_organizations_jsonl(processed / "organizations.jsonl")
    except (FileNotFoundError, ValueError):
        return None
    try:
        incidents = load_incidents_jsonl(processed / "incidents.jsonl")
    except FileNotFoundError:
        incidents = []
    except ValueError:
        return None
    return organizations, incidents


def _iso_date(value: datetime | None) -> str | None:
    return value.date().isoformat() if value is not None else None
