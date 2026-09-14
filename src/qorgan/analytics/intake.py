"""Citizen-report intake: `citizen_reports.jsonl` → incidents → refreshed analysis.

Closes the loop between the Live-call tab's consent-gated report flow and the analyst
dashboard. Ingest is **idempotent**: each report gets a deterministic content-addressed
incident id, so re-running ingests nothing new. Only the NEW reports are embedded (the
expensive step); the cached matrix from `pipeline.EMBEDDINGS_FILENAME` covers the rest,
with a full re-embed fallback when the cache is missing or stale. Placement falls out of
the existing number-graph clustering: a report whose caller number matches a known
organization joins it; an unknown number forms a new (novelty-candidate) organization.

Never called automatically — the analyst clicks "Ingest into analysis" (human-decides).
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from qorgan.analytics.embed import embed_incidents
from qorgan.analytics.pipeline import (
    analyze_incidents,
    load_embeddings_npz,
    load_organizations_jsonl,
    save_embeddings_npz,
    write_organizations_jsonl,
)
from qorgan.classifier.embed import embed_texts
from qorgan.data.incident_seed import load_incidents_jsonl, write_incidents_jsonl
from qorgan.data.schema import Incident
from qorgan.reports.model import StoredReport, report_to_incident
from qorgan.reports.store import load_reports, remove_report

_ID_HASH_CHARS = 10


class Placement(BaseModel):
    """Where one ingested report landed in the refreshed analysis."""

    model_config = ConfigDict(frozen=True)

    incident_id: str
    org_id: str
    org_is_novel: bool


class IngestSummary(BaseModel):
    """Outcome of one ingest run (what the analyst sees after clicking Ingest)."""

    model_config = ConfigDict(frozen=True)

    ingested: int = Field(ge=0)
    placements: tuple[Placement, ...] = ()
    organizations_total: int = Field(ge=0)


def report_incident_id(report: StoredReport) -> str:
    """Deterministic, content-addressed incident id for a stored report.

    Same (scrubbed) transcript + timestamp → same id, which is what makes ingest idempotent.
    """
    digest = hashlib.sha1(
        f"{report.transcript}|{report.timestamp.isoformat()}".encode()
    ).hexdigest()
    return f"report-{digest[:_ID_HASH_CHARS]}"


def load_report_drafts(reports_path: Path) -> list[StoredReport]:
    """Parse the submitted-reports JSONL; missing file means no reports yet."""
    return load_reports(reports_path)


def pending_reports(reports_path: Path, incidents: Sequence[Incident]) -> list[StoredReport]:
    """Stored reports not yet ingested (by deterministic id), deduplicated within the file."""
    existing = {incident.id for incident in incidents}
    seen: set[str] = set()
    pending: list[StoredReport] = []
    for report in load_reports(reports_path):
        report_id = report_incident_id(report)
        if report_id in existing or report_id in seen:
            continue
        seen.add(report_id)
        pending.append(report)
    return pending


def ingest_pending(
    *,
    reports_path: Path,
    incidents_path: Path,
    organizations_path: Path,
    embeddings_path: Path,
    embedder: Any = None,
    now: datetime | None = None,
) -> IngestSummary:
    """Ingest all pending reports and rewrite the analysis artifacts.

    Embeds only the new transcripts when the cache aligns with the current incident
    stream; otherwise re-embeds everything (correctness over speed on a stale cache).
    """
    incidents = load_incidents_jsonl(incidents_path) if incidents_path.exists() else []
    drafts = pending_reports(reports_path, incidents)
    if not drafts:
        organizations = (
            load_organizations_jsonl(organizations_path) if organizations_path.exists() else []
        )
        return IngestSummary(ingested=0, placements=(), organizations_total=len(organizations))

    new_incidents = [
        _normalize_timestamp(report_to_incident(draft, incident_id=report_incident_id(draft)))
        for draft in drafts
    ]
    all_incidents = [*incidents, *new_incidents]

    embeddings = _extend_or_rebuild_embeddings(
        incidents, new_incidents, embeddings_path, embedder=embedder
    )
    organizations = analyze_incidents(
        all_incidents, embeddings=embeddings, now=now or datetime.now()
    )

    write_incidents_jsonl(all_incidents, incidents_path)
    write_organizations_jsonl(organizations, organizations_path)
    save_embeddings_npz([i.id for i in all_incidents], embeddings, embeddings_path)

    org_of = {member: org for org in organizations for member in org.members}
    placements = tuple(
        Placement(
            incident_id=incident.id,
            org_id=org_of[incident.id].id,
            org_is_novel=org_of[incident.id].is_novel,
        )
        for incident in new_incidents
    )
    return IngestSummary(
        ingested=len(new_incidents),
        placements=placements,
        organizations_total=len(organizations),
    )


class ForgetSummary(BaseModel):
    """Outcome of a citizen's deletion (PLAN_2026-09 C3): the report is gone; if it had
    already been folded into the analysis, so is its incident, and the organizations were
    recomputed without it."""

    model_config = ConfigDict(frozen=True)

    receipt_id: str
    incident_removed: bool
    organizations_total: int = Field(ge=0)


def forget_report(
    receipt_id: str,
    *,
    reports_path: Path,
    incidents_path: Path,
    organizations_path: Path,
    embeddings_path: Path,
    now: datetime | None = None,
) -> ForgetSummary | None:
    """Delete the report with `receipt_id` everywhere it reached. `None` if unknown.

    The derived incident (content-addressed id) is removed from the incident stream and
    the embedding cache, and organizations are re-analysed from the cached rows -- no
    re-embedding is needed, because deletion only ever shrinks the stream.
    """
    report = remove_report(receipt_id, reports_path)
    if report is None:
        return None
    incident_id = report_incident_id(report)
    incidents = load_incidents_jsonl(incidents_path) if incidents_path.exists() else []
    remaining = [incident for incident in incidents if incident.id != incident_id]
    organizations = load_organizations_jsonl(organizations_path) if organizations_path.exists() else []
    if len(remaining) == len(incidents):
        return ForgetSummary(receipt_id=receipt_id, incident_removed=False, organizations_total=len(organizations))

    embeddings = _drop_cached_row(incidents, remaining, embeddings_path)
    organizations = analyze_incidents(remaining, embeddings=embeddings, now=now or datetime.now()) if remaining else []
    write_incidents_jsonl(remaining, incidents_path)
    write_organizations_jsonl(organizations, organizations_path)
    if embeddings is not None:
        save_embeddings_npz([i.id for i in remaining], embeddings, embeddings_path)
    return ForgetSummary(receipt_id=receipt_id, incident_removed=True, organizations_total=len(organizations))


def _drop_cached_row(
    incidents: Sequence[Incident], remaining: Sequence[Incident], embeddings_path: Path
) -> np.ndarray | None:
    """Cached embedding rows for `remaining`, or `None` when the cache is absent/misaligned
    (the next ingest then rebuilds it -- correctness over speed)."""
    if not embeddings_path.exists():
        return None
    cached_ids, cached = load_embeddings_npz(embeddings_path)
    if cached_ids != [incident.id for incident in incidents] or len(cached) != len(incidents):
        return None
    keep = {incident.id for incident in remaining}
    rows = [row for incident_id, row in zip(cached_ids, cached) if incident_id in keep]
    return np.vstack(rows) if rows else cached[:0]


def _normalize_timestamp(incident: Incident) -> Incident:
    """Convert a tz-aware report timestamp to the incident store's naive-UTC convention.

    Seeded incidents are naive; live report drafts are tz-aware UTC. Mixing the two
    crashes datetime arithmetic in priority ranking, so intake normalizes at the boundary.
    """
    ts = incident.timestamp
    if ts is None or ts.tzinfo is None:
        return incident
    return incident.model_copy(update={"timestamp": ts.astimezone(UTC).replace(tzinfo=None)})


def _extend_or_rebuild_embeddings(
    incidents: Sequence[Incident],
    new_incidents: Sequence[Incident],
    embeddings_path: Path,
    *,
    embedder: Any = None,
) -> np.ndarray:
    """Cached rows + embeddings of only the new transcripts, or a full re-embed when the
    cache is absent or does not match the current incident stream exactly."""
    if embeddings_path.exists():
        cached_ids, cached = load_embeddings_npz(embeddings_path)
        if cached_ids == [incident.id for incident in incidents] and len(cached) == len(incidents):
            new_vectors = embed_texts(
                [incident.transcript for incident in new_incidents], embedder=embedder
            )
            return np.vstack([cached, new_vectors]) if len(cached) else new_vectors
    return embed_incidents([*incidents, *new_incidents], embedder=embedder)
