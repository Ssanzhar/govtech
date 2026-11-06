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

import functools
import hashlib
import logging
import zipfile
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TypeVar

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from qorgan.analytics.embed import embed_incidents, embed_transcripts
from qorgan.analytics.feedback import FEEDBACK_FILENAME, forget_in_feedback
from qorgan.analytics.pipeline import (
    analyze_incidents,
    load_embeddings_npz,
    load_organizations_jsonl,
    save_embeddings_npz,
    write_organizations_jsonl,
)
from qorgan.data.incident_seed import load_incidents_jsonl, write_incidents_jsonl
from qorgan.data.schema import Incident, Organization
from qorgan.jsonl import count_records
from qorgan.reports.model import StoredReport, report_to_incident
from qorgan.reports.store import REPORTS_LOCK, load_reports, remove_report

_ID_HASH_CHARS = 10
_REPORT_INCIDENT_PREFIX = "report-"
_LOGGER = logging.getLogger(__name__)
_T = TypeVar("_T")


def _serialised(fn: Callable[..., _T]) -> Callable[..., _T]:
    """Run `fn` under the reports lock (re-entrant, shared with report writes and the purge).

    Ingest reads the pending reports, embeds them (seconds on the server's ONNX model) and only
    then writes the incident stream; a deletion that landed in between removed the report,
    found no incident yet, and was undone when ingest wrote the incident back (QA 2026-09-29).
    Single worker (see Dockerfile): an in-process lock is the whole story.
    """

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> _T:
        with REPORTS_LOCK:
            return fn(*args, **kwargs)

    return wrapper


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
    skipped: int = Field(default=0, ge=0)  # rows that could not become incidents; they stay pending
    placements: tuple[Placement, ...] = ()
    organizations_total: int = Field(ge=0)


def report_incident_id(report: StoredReport) -> str:
    """Deterministic, content-addressed incident id for a stored report.

    Same (scrubbed) transcript + timestamp → same id, which is what makes ingest idempotent.
    A signals-only report (no transcript) is addressed by its tactic set + number digest.
    """
    content = report.transcript or f"{','.join(report.tactic_ids)}|{report.number_hash or ''}"
    # An id, not a security function (bandit B324): same digest, flagged as non-cryptographic.
    digest = hashlib.sha1(f"{content}|{report.timestamp.isoformat()}".encode(), usedforsecurity=False).hexdigest()
    return f"{_REPORT_INCIDENT_PREFIX}{digest[:_ID_HASH_CHARS]}"


def load_report_drafts(reports_path: Path) -> list[StoredReport]:
    """Parse the submitted-reports JSONL; missing file means no reports yet."""
    return load_reports(reports_path)


def pending_reports(reports_path: Path, incidents: Sequence[Incident]) -> list[StoredReport]:
    """Stored reports not yet ingested (by deterministic id), deduplicated within the file.

    Signals-only partner reports (no transcript) are pending too: they are placed through
    the number graph alone, with a zero embedding row (PLAN_2026-09 C9).
    """
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


@_serialised
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

    new_incidents: list[Incident] = []
    skipped = 0
    for draft in drafts:
        try:
            new_incidents.append(_normalize_timestamp(report_to_incident(draft, incident_id=report_incident_id(draft))))
        except ValueError as exc:  # one bad row stays pending; it must not hold back the others
            skipped += 1
            _LOGGER.warning("a report could not become an incident (%s); it stays pending", type(exc).__name__)
    if not new_incidents:
        organizations = (
            load_organizations_jsonl(organizations_path) if organizations_path.exists() else []
        )
        return IngestSummary(ingested=0, skipped=skipped, placements=(), organizations_total=len(organizations))
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
        skipped=skipped,
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


@_serialised
def forget_report(
    receipt_id: str,
    *,
    reports_path: Path,
    incidents_path: Path,
    organizations_path: Path,
    embeddings_path: Path,
    now: datetime | None = None,
    feedback_path: Path | None = None,
) -> ForgetSummary | None:
    """Delete the report with `receipt_id` everywhere it reached. `None` if unknown.

    The derived incident (content-addressed id) is removed from the incident stream and
    the embedding cache, and organizations are re-analysed from the cached rows -- no
    re-embedding is needed, because deletion only ever shrinks the stream. Analyst feedback
    (`org_feedback.jsonl`, next to the organizations by default) loses the incident id and,
    unless another incident or report still carries it, the number digest.

    The report row goes last: if anything before it fails, the receipt still works and the
    deletion can simply be repeated. Every copy of the report from the same sender goes with it
    (a retried POST stored a second row whose receipt the citizen never saw).
    """
    reports = load_reports(reports_path, skip_corrupt=True)
    report = next((r for r in reports if r.receipt_id == receipt_id), None)
    if report is None:
        return None
    incident_id = report_incident_id(report)
    copies = [
        r for r in reports
        if r.source == report.source and r.partner_id == report.partner_id and report_incident_id(r) == incident_id
    ]
    removed, organizations_total = _forget_incident(
        incident_id,
        number_hashes=[r.number_hash for r in copies],
        surviving_reports=[r for r in reports if r not in copies],
        incidents_path=incidents_path,
        organizations_path=organizations_path,
        embeddings_path=embeddings_path,
        feedback_path=feedback_path,
        now=now,
    )
    for copy in copies:
        remove_report(copy.receipt_id, reports_path)
    return ForgetSummary(receipt_id=receipt_id, incident_removed=removed, organizations_total=organizations_total)


@_serialised
def forget_orphan_incidents(
    *,
    reports_path: Path,
    incidents_path: Path,
    organizations_path: Path,
    embeddings_path: Path,
    now: datetime | None = None,
    feedback_path: Path | None = None,
) -> list[str]:
    """Remove incidents that came from a report which no longer exists. No receipt can reach
    them, so retention has to (an older build could strand one when a deletion failed halfway)."""
    if not incidents_path.exists():
        return []
    reports = load_reports(reports_path)
    backed = {report_incident_id(report) for report in reports}
    orphans = [i for i in load_incidents_jsonl(incidents_path) if i.id.startswith(_REPORT_INCIDENT_PREFIX) and i.id not in backed]
    for orphan in orphans:
        _forget_incident(
            orphan.id,
            number_hashes=[],
            surviving_reports=reports,
            incidents_path=incidents_path,
            organizations_path=organizations_path,
            embeddings_path=embeddings_path,
            feedback_path=feedback_path,
            now=now,
        )
    return [orphan.id for orphan in orphans]


def _forget_incident(
    incident_id: str,
    *,
    number_hashes: Sequence[str | None],
    surviving_reports: Sequence[StoredReport],
    incidents_path: Path,
    organizations_path: Path,
    embeddings_path: Path,
    feedback_path: Path | None,
    now: datetime | None,
) -> tuple[bool, int]:
    """Take one incident out of every derived store. Returns (it was present, organizations left).
    Safe to repeat: each step is a no-op once done. A line that cannot be parsed is left out rather
    than stopping the deletion. `number_hashes`: those of the reports it came from; the incident's
    own goes too."""
    incidents = load_incidents_jsonl(incidents_path, skip_corrupt=True) if incidents_path.exists() else []
    remaining = [incident for incident in incidents if incident.id != incident_id]
    live_numbers = {i.number_hash for i in remaining if i.number_hash} | {
        r.number_hash for r in surviving_reports if r.number_hash
    }
    own = [i.number_hash for i in incidents if i.id == incident_id]
    # Dropping the id once is enough; each digest needs its own pass. At least one pass runs for the id.
    for number_hash in dict.fromkeys([*number_hashes, *own]) or [None]:
        forget_in_feedback(
            feedback_path or organizations_path.parent / FEEDBACK_FILENAME,
            incident_id=incident_id,
            number_hash=number_hash,
            live_numbers=live_numbers,
        )
    organizations = load_organizations_jsonl(organizations_path, skip_corrupt=True) if organizations_path.exists() else []
    present = len(remaining) != len(incidents)
    if not present and not any(incident_id in org.members for org in organizations):
        # Nothing to take out; the stores are still rewritten if a corrupt line had to be left out.
        _rewrite_if_corrupt(incidents, incidents_path, organizations, organizations_path)
        return False, len(organizations)

    embeddings = _drop_cached_row(incidents, remaining, embeddings_path)
    organizations = _reanalysed_without(incident_id, organizations, remaining, embeddings, now=now or datetime.now())
    write_incidents_jsonl(remaining, incidents_path)
    write_organizations_jsonl(organizations, organizations_path)
    if embeddings is not None:
        save_embeddings_npz([i.id for i in remaining], embeddings, embeddings_path)
    return True, len(organizations)


def _rewrite_if_corrupt(
    incidents: Sequence[Incident], incidents_path: Path, organizations: Sequence[Organization], organizations_path: Path
) -> None:
    """Drop what `skip_corrupt` left out from the files too: it may hold what was being forgotten."""
    if incidents_path.exists() and len(incidents) != count_records(incidents_path):
        write_incidents_jsonl(incidents, incidents_path)
    if organizations_path.exists() and len(organizations) != count_records(organizations_path):
        write_organizations_jsonl(organizations, organizations_path)


def _reanalysed_without(
    incident_id: str,
    organizations: Sequence[Organization],
    remaining: Sequence[Incident],
    embeddings: np.ndarray | None,
    *,
    now: datetime,
) -> list[Organization]:
    """The analysis without the incident. Without cached rows that takes the embedding model; a
    deletion does not wait for one, so if it cannot run the incident simply leaves its organization
    and the next ingest re-clusters."""
    if not remaining:
        return []
    try:
        return analyze_incidents(remaining, embeddings=embeddings, now=now)
    except Exception as exc:  # noqa: BLE001 -- any embedder failure; the promise is the deletion
        _LOGGER.warning("re-clustering after a deletion failed (%s); removing the member only", type(exc).__name__)
    by_id = {incident.id: incident for incident in remaining}
    kept: list[Organization] = []
    for org in organizations:
        if incident_id not in org.members:
            kept.append(org)
            continue
        members = tuple(member for member in org.members if member != incident_id)
        if members:
            kept.append(org.model_copy(update={
                "members": members,
                "numbers": tuple(sorted({n for m in members if m in by_id and (n := by_id[m].number_hash)})),
                "representative_script": by_id[members[0]].transcript if members[0] in by_id else None,
            }))
    return kept


def _readable_cache(embeddings_path: Path) -> tuple[list[str], np.ndarray] | None:
    """The embedding cache, or `None` if there is none worth reading. It is only a cache: a damaged
    file (a crash mid-write on an older build) or a zero-width matrix (a first ingest of
    signals-only reports) is deleted and counts as a miss, never as a failure."""
    if not embeddings_path.exists():
        return None
    try:
        ids, rows = load_embeddings_npz(embeddings_path)
    except (OSError, ValueError, EOFError, KeyError, zipfile.BadZipFile) as exc:
        _LOGGER.warning("the embedding cache is unreadable (%s); it will be rebuilt", type(exc).__name__)
        embeddings_path.unlink(missing_ok=True)
        return None
    if rows.ndim != 2 or (len(rows) and rows.shape[1] == 0):
        embeddings_path.unlink(missing_ok=True)
        return None
    return ids, rows


def _drop_cached_row(
    incidents: Sequence[Incident], remaining: Sequence[Incident], embeddings_path: Path
) -> np.ndarray | None:
    """Cached embedding rows for `remaining`, or `None` when the cache is absent or misaligned. A
    misaligned file is deleted, because it may still hold the row being forgotten (the next ingest
    rebuilds it -- correctness over speed)."""
    cache = _readable_cache(embeddings_path)
    if cache is None:
        return None
    cached_ids, cached = cache
    if cached_ids != [incident.id for incident in incidents] or len(cached) != len(incidents):
        embeddings_path.unlink(missing_ok=True)
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
    cache = _readable_cache(embeddings_path)
    if cache is not None:
        cached_ids, cached = cache
        if cached_ids == [incident.id for incident in incidents] and len(cached) == len(incidents):
            new_vectors = embed_transcripts(
                [incident.transcript for incident in new_incidents], embedder=embedder
            )
            if len(cached) and new_vectors.shape[1] == 0:  # only signals-only rows are new
                new_vectors = np.zeros((len(new_incidents), cached.shape[1]), dtype=np.float32)
            return np.vstack([cached, new_vectors]) if len(cached) else new_vectors
    return embed_incidents([*incidents, *new_incidents], embedder=embedder)
