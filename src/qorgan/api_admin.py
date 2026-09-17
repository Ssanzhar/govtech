"""Analyst-dashboard HTTP API (Level 2) — a thin wrapper over `analytics.pipeline` +
`analytics.presentation`, the same read path `app/analyst_view.py` uses.

Degrades honestly: if the precomputed analysis (`organizations.jsonl` / `incidents.jsonl`)
is missing or corrupt, endpoints return `available: false` / 404 instead of a 500 — the
demo must never crash just because `scripts/demo_seed.py` +
`python -m qorgan.analytics.pipeline` haven't been run yet (CLAUDE.md SS9).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Literal

from datetime import UTC, datetime

from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from qorgan.analytics.intake import ingest_pending, pending_reports
from qorgan.audit import AUDIT_FILENAME, AuditEntry, append_audit
from qorgan.classifier import predict
from qorgan.explain.explainer import ExplainerError, explain
from qorgan.analytics.pipeline import load_organizations_jsonl
from qorgan.analytics.presentation import (
    dashboard_kpis,
    last_activity,
    org_display_name,
    org_tactic_profile,
    short_tactic_name,
)
from qorgan.config import get_config
from qorgan.data.incident_seed import load_incidents_jsonl
from qorgan.data.schema import Incident, Organization
from qorgan.reports.store import REPORTS_FILENAME

router = APIRouter(prefix="/api/admin", tags=["admin"])

_LOGGER = logging.getLogger(__name__)

# Drill-down call rows: enough for the analyst to search/sort within an organization
# while keeping the payload modest (60 × ~250-char excerpts ≈ 16 KB).
_MAX_SAMPLE_INCIDENTS = 60
_EXCERPT_CHARS = 200
_ELLIPSIS = "…"
# Analysts are not authenticated in this demo; the header only names who opened a case in
# the audit line (PLAN C4). A real deployment puts SSO in front of /api/admin.
_ANALYST_ID_HEADER = "X-Analyst-Id"
_DEFAULT_ANALYST_ID = "anonymous-analyst"
_MAX_OPEN_REASON_CHARS = 160
# Test seam: a deterministic fake embedder is injected here; None means the real
# sentence-transformers model (downloaded/cached on first ingest).
_EMBEDDER_OVERRIDE: Any = None
Locale = Literal["ru", "kk"]


@dataclass(frozen=True)
class _Analysis:
    organizations: list[Organization]
    incidents: list[Incident]
    pending: int


class KpiOut(BaseModel):
    incidents: int
    organizations: int
    novel_schemes: int
    pending_reports: int


class OrgSummaryOut(BaseModel):
    id: str
    name: str
    priority: float
    incidents: int
    numbers: list[str]  # full list so the queue is searchable by caller number
    last_activity: str | None
    is_novel: bool


class OverviewResponse(BaseModel):
    available: bool
    kpis: KpiOut
    organizations: list[OrgSummaryOut]


class TacticOut(BaseModel):
    id: str
    name: str
    count: int


class SampleIncidentOut(BaseModel):
    id: str
    date: str | None
    number: str | None
    risk: float
    excerpt: str


class OrgDetailResponse(BaseModel):
    id: str
    name: str
    priority: float
    is_novel: bool
    numbers: list[str]
    tactics: list[TacticOut]
    representative_script: str | None
    sample_incidents: list[SampleIncidentOut]


class RankedTagOut(BaseModel):
    id: str
    name: str
    weight: float


class AnalysisSpanOut(BaseModel):
    text: str
    start: int
    end: int


class IncidentAnalysisResponse(BaseModel):
    """The live model verdict for one call — same `score()` contract as /api/analyze,
    computed on demand so the analyst always sees the current model, never a cached label.
    Carries an excerpt only; the full transcript needs an explicit, audited `open` (C4)."""

    incident_id: str
    excerpt: str
    risk: float
    threshold: float
    flagged: bool
    backend: str
    fallback: bool
    tags: list[RankedTagOut]
    spans: list[AnalysisSpanOut]
    reason: str
    caveat: str


class OpenCaseRequest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    reason: str | None = Field(default=None, max_length=_MAX_OPEN_REASON_CHARS)


class OpenCaseResponse(IncidentAnalysisResponse):
    """The analysis plus the full (scrubbed) transcript — returned only by `open`."""

    transcript: str


class PlacementOut(BaseModel):
    incident_id: str
    org_id: str
    org_name: str
    org_is_novel: bool


class IngestResponse(BaseModel):
    ingested: int
    organizations_total: int
    placements: list[PlacementOut]


def _load_analysis() -> _Analysis | None:
    """Load organizations + incidents for the configured data dir, or None if the
    analysis is absent or corrupt — the sole degrade point every endpoint routes through."""
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
    try:
        pending = len(pending_reports(processed / REPORTS_FILENAME, incidents))
    except ValueError:
        pending = 0
    return _Analysis(organizations=organizations, incidents=incidents, pending=pending)


@router.get("/overview", response_model=OverviewResponse)
def overview(locale: Locale = "ru") -> OverviewResponse:
    analysis = _load_analysis()
    if analysis is None:
        return OverviewResponse(
            available=False,
            kpis=KpiOut(incidents=0, organizations=0, novel_schemes=0, pending_reports=0),
            organizations=[],
        )

    by_id = {incident.id: incident for incident in analysis.incidents}
    kpis = dashboard_kpis(
        analysis.organizations, analysis.incidents, pending_reports=analysis.pending
    )
    orgs_out = [
        OrgSummaryOut(
            id=org.id,
            name=org_display_name(org, by_id, locale=locale),
            priority=org.priority,
            incidents=len(org.members),
            numbers=list(org.numbers),
            last_activity=_iso_date(last_activity(org, by_id)),
            is_novel=org.is_novel,
        )
        for org in analysis.organizations
    ]
    return OverviewResponse(
        available=True,
        kpis=KpiOut(
            incidents=kpis.incidents_total,
            organizations=kpis.organizations_total,
            novel_schemes=kpis.novel_schemes,
            pending_reports=kpis.pending_reports,
        ),
        organizations=orgs_out,
    )


@router.get("/organizations/{org_id}", response_model=OrgDetailResponse)
def organization_detail(org_id: str, locale: Locale = "ru") -> OrgDetailResponse:
    analysis = _load_analysis()
    if analysis is None:
        raise HTTPException(status_code=404, detail="analysis unavailable")

    org = next((o for o in analysis.organizations if o.id == org_id), None)
    if org is None:
        raise HTTPException(status_code=404, detail=f"unknown organization {org_id!r}")

    by_id = {incident.id: incident for incident in analysis.incidents}
    profile = org_tactic_profile(org, by_id)
    tactics = [
        TacticOut(id=tactic_id, name=name, count=count)
        for tactic_id, count in profile.items()
        if (name := short_tactic_name(tactic_id, locale=locale)) is not None
    ]
    sample_incidents = [
        SampleIncidentOut(
            id=incident.id,
            date=incident.timestamp.strftime("%Y-%m-%d %H:%M") if incident.timestamp else None,
            number=incident.number_prefix,
            risk=incident.label.risk,
            excerpt=_excerpt(incident.transcript),
        )
        for incident in (
            by_id[member] for member in org.members[:_MAX_SAMPLE_INCIDENTS] if member in by_id
        )
    ]

    return OrgDetailResponse(
        id=org.id,
        name=org_display_name(org, by_id, locale=locale),
        priority=org.priority,
        is_novel=org.is_novel,
        numbers=list(org.numbers),
        tactics=tactics,
        representative_script=_excerpt(org.representative_script) if org.representative_script else None,
        sample_incidents=sample_incidents,
    )


@router.get("/incidents/{incident_id}/analysis", response_model=IncidentAnalysisResponse)
def incident_analysis(
    incident_id: str, locale: Locale = "ru", backend: str | None = None
) -> IncidentAnalysisResponse:
    """Score one call with the classifier, on demand (the drill-down's rank graph).

    Same backend-resolution contract as /api/analyze: an explicitly unknown backend is
    a 422; a configured-but-unavailable one degrades honestly to `mock` and says so.
    """
    return _analyse(_find_incident(incident_id), locale, backend)


@router.post("/incidents/{incident_id}/open", response_model=OpenCaseResponse)
def open_case(
    incident_id: str,
    body: OpenCaseRequest | None = None,
    locale: Locale = "ru",
    backend: str | None = None,
    analyst_id: str = Header(default=_DEFAULT_ANALYST_ID, alias=_ANALYST_ID_HEADER),
) -> OpenCaseResponse:
    """The explicit "open case" action (PLAN C4): the only way an analyst sees a full
    transcript, and every call leaves a content-free audit line naming who opened what."""
    incident = _find_incident(incident_id)
    reason = body.reason if body is not None else None
    try:
        entry = AuditEntry(
            timestamp=datetime.now(UTC),
            actor_kind="analyst",
            actor_id=analyst_id.strip() or _DEFAULT_ANALYST_ID,
            action="case.open",
            subject=f"incident:{incident.id}",
            outcome=f"ok: {reason.strip()}" if reason and reason.strip() else "ok",
        )
    except ValidationError as exc:  # the reason carried a number / content
        raise HTTPException(status_code=422, detail="reason must not carry call content or numbers") from exc
    append_audit(entry, get_config().data_dir / "processed" / AUDIT_FILENAME)
    analysed = _analyse(incident, locale, backend)
    return OpenCaseResponse(**analysed.model_dump(), transcript=incident.transcript)


def _find_incident(incident_id: str) -> Incident:
    analysis = _load_analysis()
    if analysis is None:
        raise HTTPException(status_code=404, detail="analysis unavailable")
    incident = next((i for i in analysis.incidents if i.id == incident_id), None)
    if incident is None:
        raise HTTPException(status_code=404, detail=f"unknown incident {incident_id!r}")
    return incident


def _analyse(incident: Incident, locale: Locale, backend: str | None) -> IncidentAnalysisResponse:
    fallback = False
    try:
        result = predict.score(incident.transcript, backend=backend)
    except (predict.UnknownBackendError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception:  # configured backend unavailable here (weights/keys) — degrade honestly
        result = predict.score(incident.transcript, backend="mock")
        fallback = True

    try:
        explanation = explain(result, incident.transcript, locale)
    except ExplainerError as exc:
        raise HTTPException(status_code=500, detail=f"explainer failed: {exc}") from exc

    cfg = get_config()
    ranked = sorted(result.tags, key=lambda tag: -tag.weight)
    return IncidentAnalysisResponse(
        incident_id=incident.id,
        excerpt=_excerpt(incident.transcript),
        risk=result.risk,
        threshold=cfg.risk_threshold,
        flagged=result.risk >= cfg.risk_threshold,
        backend=result.backend,
        fallback=fallback,
        tags=[
            RankedTagOut(
                id=tag.id,
                name=short_tactic_name(tag.id, locale=locale) or tag.id,
                weight=tag.weight,
            )
            for tag in ranked
        ],
        spans=[
            AnalysisSpanOut(text=span.text, start=span.start, end=span.end)
            for span in result.attributions
        ],
        reason=explanation.reason,
        caveat=explanation.caveat,
    )


@router.post("/ingest", response_model=IngestResponse)
def ingest(locale: Locale = "ru") -> IngestResponse:
    """Ingest pending citizen reports into the analysis (the analyst's explicit click —
    never automatic, per the human-decides principle). Embeds only the new transcripts
    via the cached matrix; the first ingest on a fresh clone may download the embedding
    model, hence the 503 (not 500) when that is impossible here."""
    from qorgan.analytics.pipeline import EMBEDDINGS_FILENAME

    processed = get_config().data_dir / "processed"
    try:
        summary = ingest_pending(
            reports_path=processed / REPORTS_FILENAME,
            incidents_path=processed / "incidents.jsonl",
            organizations_path=processed / "organizations.jsonl",
            embeddings_path=processed / EMBEDDINGS_FILENAME,
            embedder=_EMBEDDER_OVERRIDE,
        )
    except Exception as exc:
        _LOGGER.exception("citizen-report ingest failed")
        raise HTTPException(status_code=503, detail=f"ingest failed: {exc}") from exc

    org_names = _org_display_names(locale)
    return IngestResponse(
        ingested=summary.ingested,
        organizations_total=summary.organizations_total,
        placements=[
            PlacementOut(
                incident_id=placement.incident_id,
                org_id=placement.org_id,
                org_name=org_names.get(placement.org_id, placement.org_id),
                org_is_novel=placement.org_is_novel,
            )
            for placement in summary.placements
        ],
    )


def _excerpt(text: str) -> str:
    return text if len(text) <= _EXCERPT_CHARS else text[:_EXCERPT_CHARS] + _ELLIPSIS


def _org_display_names(locale: Locale) -> dict[str, str]:
    """Tactic-derived display names for the refreshed analysis (raw ids as fallback)."""
    analysis = _load_analysis()
    if analysis is None:
        return {}
    by_id = {incident.id: incident for incident in analysis.incidents}
    return {
        org.id: org_display_name(org, by_id, locale=locale) for org in analysis.organizations
    }


def _iso_date(value) -> str | None:
    return value.date().isoformat() if value is not None else None
