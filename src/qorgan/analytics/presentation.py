"""Pure presentation helpers for the analyst dashboard (no Streamlit imports).

Turns clustered `Organization`s + their member `Incident`s into human-readable material:
display names derived from dominant tactics (never raw cluster ids), tactic profiles and
activity series for charts, and KPI totals. Kept Streamlit-free so every function is unit
testable and reusable by any future view (CLI report, PDF export).
"""

from __future__ import annotations

import hashlib
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from qorgan.data.schema import Incident, Organization
from qorgan.taxonomy import get_taxonomy

# Queue labels stay scannable: at most this many tactic names joined into a display name.
_TOP_TACTICS_IN_NAME = 2
_FINGERPRINT_CHARS = 16
_HEX_TO_LETTERS = str.maketrans("0123456789abcdef", "ghijklmnopqrstuv")


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
    names: list[str] = []
    for tactic_id in org_tactic_profile(org, incidents_by_id):
        short = short_tactic_name(tactic_id, locale=locale)
        if short is None:
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


def org_display_names(
    organizations: Sequence[Organization],
    incidents_by_id: Mapping[str, Incident],
    *,
    locale: str = "ru",
) -> dict[str, str]:
    """`org_display_name` for a whole queue, with no label used twice.

    Two operations can share their dominant tactics (the seeded bank and police scripts both lead
    with intimidation and fake verification), and an analyst cannot merge or compare what reads the
    same. A clash gets the tactic that most sets the operation apart; identical profiles get the id.
    """
    base = {org.id: org_display_name(org, incidents_by_id, locale=locale) for org in organizations}
    names = dict(base)
    for org in organizations:
        rivals = [o for o in organizations if o.id != org.id and base[o.id] == base[org.id]]
        if rivals and (extra := _distinguishing_tactic(org, rivals, incidents_by_id, base[org.id], locale)):
            names[org.id] = f"{base[org.id]} · {extra}"
    uses = Counter(names.values())
    return {org_id: f"{name} · {org_id}" if uses[name] > 1 else name for org_id, name in names.items()}


def _tactic_shares(org: Organization, incidents_by_id: Mapping[str, Incident]) -> dict[str, float]:
    """Tactic-id -> the share of the org's incidents carrying it."""
    members = max(len(_member_incidents(org, incidents_by_id)), 1)
    return {tactic_id: count / members for tactic_id, count in org_tactic_profile(org, incidents_by_id).items()}


def _distinguishing_tactic(
    org: Organization,
    rivals: Sequence[Organization],
    incidents_by_id: Mapping[str, Incident],
    name: str,
    locale: str,
) -> str | None:
    """The short name of the tactic this org uses most beyond its rivals, or None if none does."""
    rival_shares = [_tactic_shares(rival, incidents_by_id) for rival in rivals]
    best: tuple[float, str] | None = None
    for tactic_id, share in _tactic_shares(org, incidents_by_id).items():
        short = short_tactic_name(tactic_id, locale=locale)
        if short is None or short in name:
            continue
        lead = share - max(shares.get(tactic_id, 0.0) for shares in rival_shares)
        if lead > 0 and (best is None or lead > best[0]):
            best = (lead, short)
    return best[1] if best else None


def org_fingerprint(org: Organization) -> str:
    """Which operation this is, independent of the positional `org_<n>`: its members. Content-free,
    and in letters only: it goes into the audit log, whose scrubber reads a run of digits in a hex
    digest as a phone number and would refuse every verdict on that operation."""
    joined = "\n".join(sorted(org.members))
    digest = hashlib.sha256(joined.encode("utf-8")).hexdigest()[:_FINGERPRINT_CHARS]
    return digest.translate(_HEX_TO_LETTERS)


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
    """Short localized tactic name for queue labels and chart axes; None for unknown ids."""
    taxonomy = get_taxonomy()
    try:
        written = taxonomy.get(tactic_id).short.get(locale)
        return written or _short_name(taxonomy.display_name(tactic_id, locale))
    except KeyError:
        return None


def _member_incidents(
    org: Organization, incidents_by_id: Mapping[str, Incident]
) -> list[Incident]:
    """The org's member incidents that are actually present in the map (ghosts skipped)."""
    return [incidents_by_id[m] for m in org.members if m in incidents_by_id]


def _short_name(display_name: str) -> str:
    """Truncate a taxonomy display name at its first slash/parenthesis clause, closing any
    quotation the cut left open. For a tactic without a written `short` name."""
    short = display_name.split("/")[0].split("(")[0].strip()
    if short.count('"') % 2:
        short += '"'
    return short + "»" * (short.count("«") - short.count("»"))
