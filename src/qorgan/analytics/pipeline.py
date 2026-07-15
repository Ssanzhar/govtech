"""Level-2 pipeline (D5-6): incidents -> ranked scam organizations with novelty flags, plus
cluster-quality metrics against the ground-truth script families.

`analyze_incidents` chains embed -> cluster (HDBSCAN + number graph) -> novelty -> priority,
returning organizations sorted most-urgent-first. The CLI precomputes the analysis to
`organizations.jsonl` so the Streamlit panel reads it instantly (degrade-ready).
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from qorgan.analytics import cluster as cluster_mod
from qorgan.analytics import novelty as novelty_mod
from qorgan.analytics import rank as rank_mod
from qorgan.analytics.embed import embed_incidents
from qorgan.config import get_config
from qorgan.data.incident_seed import load_incidents_jsonl
from qorgan.data.schema import Incident, Organization
from qorgan.eval import metrics

_RECENT_WINDOW_DAYS = 7.0
# Novelty tuned to real e5 scale: established scam families sit ~0.03 cosine-distance apart,
# a genuinely new scheme ~0.10+ -- so a small org past this distance is a new scheme.
_NOVELTY_MAX_SIZE = 10
_NOVELTY_MIN_DISTANCE = 0.06


def analyze_incidents(
    incidents: Sequence[Incident],
    *,
    embeddings: np.ndarray | None = None,
    embedder: Any = None,
    now: datetime | None = None,
    min_cluster_size: int = cluster_mod.DEFAULT_MIN_CLUSTER_SIZE,
    novelty_max_size: int = _NOVELTY_MAX_SIZE,
    novelty_min_distance: float = _NOVELTY_MIN_DISTANCE,
) -> list[Organization]:
    """Cluster -> flag novelty -> score priority; return orgs sorted by priority desc."""
    if not incidents:
        return []
    if embeddings is None:
        embeddings = embed_incidents(incidents, embedder=embedder)
    organizations = cluster_mod.cluster_incidents(incidents, embeddings, min_cluster_size=min_cluster_size)
    organizations = novelty_mod.flag_novel_organizations(
        organizations, incidents, embeddings,
        max_novel_size=novelty_max_size, min_distance=novelty_min_distance,
    )
    organizations = assign_priority(organizations, incidents, now=now or datetime.now())
    return sorted(organizations, key=lambda org: org.priority, reverse=True)


def assign_priority(
    organizations: Sequence[Organization],
    incidents: Sequence[Incident],
    *,
    now: datetime,
    recent_window_days: float = _RECENT_WINDOW_DAYS,
) -> list[Organization]:
    """Set each org's `priority` from size + recency + recent-growth of its incidents."""
    timestamps = {incident.id: incident.timestamp for incident in incidents}
    scored: list[Organization] = []
    for org in organizations:
        member_ts = [timestamps[m] for m in org.members if timestamps.get(m) is not None]
        recency = growth = 0.0
        if member_ts:
            recency = rank_mod.recency_score(max(member_ts), now)
            window_seconds = recent_window_days * rank_mod.SECONDS_PER_DAY
            recent = sum(1 for t in member_ts if (now - t).total_seconds() <= window_seconds)
            growth = rank_mod.growth_score(recent, len(member_ts))
        priority = rank_mod.priority_score(size=len(org.members), recency=recency, growth=growth)
        scored.append(org.model_copy(update={"priority": priority}))
    return scored


def cluster_quality(organizations: Sequence[Organization], incidents: Sequence[Incident]) -> dict:
    """Purity + Adjusted Rand of the organizations vs the ground-truth `script_family`."""
    org_of = {member: org.id for org in organizations for member in org.members}
    labels_true = [incident.script_family or "unknown" for incident in incidents]
    labels_pred = [org_of.get(incident.id, "unassigned") for incident in incidents]
    return {
        "purity": metrics.purity(labels_true, labels_pred),
        "ari": metrics.adjusted_rand(labels_true, labels_pred),
        "n_organizations": len(organizations),
        "n_novel": sum(1 for org in organizations if org.is_novel),
    }


# Embedding cache written next to organizations.jsonl so `intake.py` can extend the
# analysis with only the NEW reports embedded (the expensive step) instead of re-embedding
# the whole incident stream on every ingest.
EMBEDDINGS_FILENAME = "incident_embeddings.npz"


def save_embeddings_npz(ids: Sequence[str], embeddings: np.ndarray, path: Path) -> None:
    """Persist the incident-embedding matrix keyed by incident ids."""
    if len(ids) != len(embeddings):
        raise ValueError(f"ids/embeddings length mismatch: {len(ids)} vs {len(embeddings)}")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, ids=np.array(list(ids)), embeddings=embeddings)


def load_embeddings_npz(path: Path) -> tuple[list[str], np.ndarray]:
    """Load an embedding cache written by `save_embeddings_npz`."""
    if not path.exists():
        raise FileNotFoundError(f"Embedding cache not found: {path}")
    data = np.load(path, allow_pickle=False)
    return [str(incident_id) for incident_id in data["ids"]], data["embeddings"]


def write_organizations_jsonl(organizations: Sequence[Organization], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(o.model_dump_json() for o in organizations) + "\n", encoding="utf-8")


def load_organizations_jsonl(path: Path) -> list[Organization]:
    if not path.exists():
        raise FileNotFoundError(f"Organizations not found: {path}. Run `python -m qorgan.analytics.pipeline`.")
    return [Organization.model_validate_json(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main(argv: Sequence[str] | None = None) -> None:  # pragma: no cover - CLI (real embeddings)
    cfg = get_config()
    parser = argparse.ArgumentParser(description="Analyze L2 incidents into ranked organizations.")
    parser.add_argument("--incidents", type=Path, default=cfg.data_dir / "processed" / "incidents.jsonl")
    parser.add_argument("--out", type=Path, default=cfg.data_dir / "processed" / "organizations.jsonl")
    args = parser.parse_args(argv)

    incidents = load_incidents_jsonl(args.incidents)
    embeddings = embed_incidents(incidents)
    organizations = analyze_incidents(incidents, embeddings=embeddings)
    write_organizations_jsonl(organizations, args.out)
    embeddings_path = args.out.with_name(EMBEDDINGS_FILENAME)
    save_embeddings_npz([i.id for i in incidents], embeddings, embeddings_path)
    quality = cluster_quality(organizations, incidents)
    print(json.dumps(quality, indent=2))
    print(f"wrote {len(organizations)} organizations -> {args.out}")
    print(f"wrote embedding cache -> {embeddings_path}")


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
