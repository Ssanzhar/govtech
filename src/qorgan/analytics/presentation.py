"""Pure presentation helpers for the analyst dashboard (no Streamlit imports).

Turns clustered `Organization`s + their member `Incident`s into human-readable material:
display names derived from dominant tactics (never raw cluster ids), tactic profiles and
activity series for charts, and KPI totals. Kept Streamlit-free so every function is unit
testable and reusable by any future view (CLI report, PDF export).
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from qorgan.data.schema import Incident, Organization
from qorgan.taxonomy import get_taxonomy

# Queue labels stay scannable: at most this many tactic names joined into a display name.
_TOP_TACTICS_IN_NAME = 2


class DashboardKpis(BaseModel):
    """Headline totals for the analyst dashboard's metric row."""

    model_config = ConfigDict(frozen=True)

    incidents_total: int = Field(ge=0)
    organizations_total: int = Field(ge=0)
    novel_schemes: int = Field(ge=0)
    pending_reports: int = Field(ge=0)


def org_display_name(
    org: Organization, incidents_by_id: Mapping[str, Incident], *, locale: str = "ru"
) -> str:
    """A human label for an organization: its dominant tactic names, localized.

    Falls back to the most common member `script_family`, then to the raw org id —
    the id is a last resort, never the normal case. Unknown tactic ids are skipped
    (same convention as the explainer: a future-taxonomy tag must never crash a view).
    """
    taxonomy = get_taxonomy()
    names: list[str] = []
    for tactic_id in org_tactic_profile(org, incidents_by_id):
        try:
            short = _short_name(taxonomy.display_name(tactic_id, locale))
        except KeyError:
            continue
        if short not in names:
            names.append(short)
        if len(names) == _TOP_TACTICS_IN_NAME:
            break
    if names:
        return " + ".join(names)

    families = Counter(
        incident.script_family
        for incident in _member_incidents(org, incidents_by_id)
        if incident.script_family
    )
    if families:
        return families.most_common(1)[0][0]
    return org.id


def org_tactic_profile(
    org: Organization, incidents_by_id: Mapping[str, Incident]
) -> dict[str, int]:
    """Tactic-id → count across the org's member incidents, most common first."""
    counts = Counter(
        tag.id
        for incident in _member_incidents(org, incidents_by_id)
        for tag in incident.label.tactic_tags
    )
    return dict(counts.most_common())


def org_activity_by_day(
    org: Organization, incidents_by_id: Mapping[str, Incident]
) -> dict[str, int]:
    """ISO date → incident count for the org, ascending by date (chart-ready).

    Incidents without a timestamp are skipped rather than guessed.
    """
    counts = Counter(
        incident.timestamp.date().isoformat()
        for incident in _member_incidents(org, incidents_by_id)
        if incident.timestamp is not None
    )
    return dict(sorted(counts.items()))


def last_activity(
    org: Organization, incidents_by_id: Mapping[str, Incident]
) -> datetime | None:
    """The org's most recent incident timestamp, or None if none are dated."""
    timestamps = [
        incident.timestamp
        for incident in _member_incidents(org, incidents_by_id)
        if incident.timestamp is not None
    ]
    return max(timestamps) if timestamps else None


def dashboard_kpis(
    organizations: Sequence[Organization],
    incidents: Sequence[Incident],
    *,
    pending_reports: int,
) -> DashboardKpis:
    """Headline totals for the metric row."""
    return DashboardKpis(
        incidents_total=len(incidents),
        organizations_total=len(organizations),
        novel_schemes=sum(1 for org in organizations if org.is_novel),
        pending_reports=pending_reports,
    )


def short_tactic_name(tactic_id: str, *, locale: str = "ru") -> str | None:
    """Shortened localized tactic name for chart axes; None for unknown ids."""
    try:
        return _short_name(get_taxonomy().display_name(tactic_id, locale))
    except KeyError:
        return None


def _member_incidents(
    org: Organization, incidents_by_id: Mapping[str, Incident]
) -> list[Incident]:
    """The org's member incidents that are actually present in the map (ghosts skipped)."""
    return [incidents_by_id[m] for m in org.members if m in incidents_by_id]


def _short_name(display_name: str) -> str:
    """Truncate a long taxonomy display name at its first slash/parenthesis clause."""
    return display_name.split("/")[0].split("(")[0].strip()
