"""Cluster incidents into scam **organizations** (D5-3): HDBSCAN on transcript embeddings,
overlaid with a phone-number co-occurrence graph.

Two incidents join the same organization if they cluster together on text **or** share a
phone number (union of both signals). The number overlay is what makes this robust: a small
or stylistically-varied group that HDBSCAN drops as noise is still held together by its
reused numbers -- exactly how a real operation is linked. Organizations come back with
`priority` / `is_novel` unset (0.0 / False); `rank.py` and `novelty.py` fill those in.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

import numpy as np

from qorgan.analytics.numbers import link_by_shared_numbers, numbers_for_group
from qorgan.data.schema import Incident, Organization

DEFAULT_MIN_CLUSTER_SIZE = 8
# A text (HDBSCAN) cluster only *merges* number-components if it covers at most this
# fraction of all incidents. Scam scripts across families are semantically near-identical
# (cosine 0.97+, see docs/eval_report.md), so HDBSCAN tends to lump most incidents into one
# giant cluster -- merging on that would collapse every organization into one. The phone
# number graph stays the reliable primary link; text only merges genuine tight sub-clusters.
_MAX_TEXT_MERGE_FRACTION = 0.5


def cluster_incidents(
    incidents: Sequence[Incident],
    embeddings: np.ndarray | None = None,
    *,
    min_cluster_size: int = DEFAULT_MIN_CLUSTER_SIZE,
    use_text_merge: bool = False,
) -> list[Organization]:
    """Group `incidents` into organizations.

    The **phone-number co-occurrence graph is the reliable primary link** (incidents sharing
    a number are the same operation). HDBSCAN text clustering is available as an opt-in
    overlay (`use_text_merge=True`) but is OFF by default: scam scripts across families are
    semantically near-identical (cosine 0.97+, see `docs/eval_report.md`), so text clusters
    span families and over-merge every organization into one. Text embeddings are instead
    used downstream for novelty detection, where the signal is reliable.
    """
    if embeddings is not None and len(incidents) != len(embeddings):
        raise ValueError(f"incidents/embeddings length mismatch: {len(incidents)} vs {len(embeddings)}")
    if not incidents:
        return []

    ids = [incident.id for incident in incidents]
    parent = {incident_id: incident_id for incident_id in ids}

    # Primary signal: union incidents sharing a phone number (transitively).
    incident_numbers = {
        incident.id: ([incident.number_hash] if incident.number_hash else []) for incident in incidents
    }
    for component in link_by_shared_numbers(incident_numbers):
        _union_all(parent, list(component))

    # Optional text overlay (guarded against the "everything is a scam" mega-blob).
    if use_text_merge and embeddings is not None:
        _apply_text_merge(parent, ids, _hdbscan_labels(embeddings, min_cluster_size))

    return _build_organizations(incidents, parent, incident_numbers)


def _apply_text_merge(parent: dict[str, str], ids: Sequence[str], labels: np.ndarray) -> None:
    max_merge_size = _MAX_TEXT_MERGE_FRACTION * len(ids)
    label_members: dict[int, list[str]] = {}
    for incident_id, label in zip(ids, labels, strict=True):
        if label != -1:
            label_members.setdefault(label, []).append(incident_id)
    for members in label_members.values():
        if len(members) <= max_merge_size:
            _union_all(parent, members)


def _hdbscan_labels(embeddings: np.ndarray, min_cluster_size: int) -> np.ndarray:
    import hdbscan

    count = len(embeddings)
    effective = max(2, min(min_cluster_size, count))
    clusterer = hdbscan.HDBSCAN(min_cluster_size=effective, metric="euclidean")
    return clusterer.fit_predict(np.ascontiguousarray(embeddings, dtype=np.float64))


def _find(parent: dict[str, str], node: str) -> str:
    while parent[node] != node:
        parent[node] = parent[parent[node]]
        node = parent[node]
    return node


def _union_all(parent: dict[str, str], members: Sequence[str]) -> None:
    for other in members[1:]:
        parent[_find(parent, other)] = _find(parent, members[0])


def _build_organizations(
    incidents: Sequence[Incident], parent: dict[str, str], incident_numbers: dict[str, list[str]]
) -> list[Organization]:
    transcripts = {incident.id: incident.transcript for incident in incidents}
    groups: dict[str, list[str]] = {}
    for incident_id in parent:
        groups.setdefault(_find(parent, incident_id), []).append(incident_id)

    ordered = sorted(groups.values(), key=lambda members: (-len(members), min(members)))
    organizations: list[Organization] = []
    for index, members in enumerate(ordered):
        members_sorted = tuple(sorted(members))
        representative = Counter(transcripts[m] for m in members_sorted).most_common(1)[0][0]
        organizations.append(
            Organization(
                id=f"org_{index}",
                members=members_sorted,
                numbers=numbers_for_group(members_sorted, incident_numbers),
                representative_script=representative,
            )
        )
    return organizations
