"""Novelty detection for Level-2 (D5-5): flag a scam **organization** as a *new scheme*.

Heuristic (distance-to-established-cluster): an organization is novel if it is still small
(`size <= max_novel_size`) AND its transcript centroid sits far (cosine distance
`>= min_distance`) from every *established* (large) organization AND it has **support** --
a linkable caller number or at least `min_support` incidents. A brand-new scheme starts
small and looks unlike anything seen before -- exactly this signature; a single call with
no number is an anomaly, not a scheme (PLAN_2026-09 C10: with half the numbers rotated,
29 such singletons were flagged before the support rule; `python -m qorgan.eval.cluster`).
Pure `novelty_flags` is deterministic and tested; `flag_novel_organizations` wraps it over
embeddings.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from qorgan.data.schema import Incident, Organization

_DEFAULT_MAX_NOVEL_SIZE = 5
_DEFAULT_MIN_DISTANCE = 0.4
# A number-less organization needs this many incidents before it can be called a scheme.
_DEFAULT_MIN_SUPPORT = 2


def novelty_flags(
    centroids: np.ndarray, sizes: Sequence[int], *, max_novel_size: int, min_distance: float
) -> list[bool]:
    """Per-organization novelty flags.

    An org is novel iff `sizes[i] <= max_novel_size` and its centroid's minimum cosine
    distance to any *large* org (`size > max_novel_size`) is `>= min_distance`. If there are
    no large orgs to compare against, nothing is flagged. Raises `ValueError` on length
    mismatch.
    """
    if len(centroids) != len(sizes):
        raise ValueError(f"centroids/sizes length mismatch: {len(centroids)} vs {len(sizes)}")

    large = [i for i, size in enumerate(sizes) if size > max_novel_size]
    flags: list[bool] = []
    for index, size in enumerate(sizes):
        if size > max_novel_size or not large:
            flags.append(False)
            continue
        nearest = min(_cosine_distance(centroids[index], centroids[j]) for j in large)
        flags.append(nearest >= min_distance)
    return flags


def flag_novel_organizations(
    organizations: Sequence[Organization],
    incidents: Sequence[Incident],
    embeddings: np.ndarray,
    *,
    max_novel_size: int = _DEFAULT_MAX_NOVEL_SIZE,
    min_distance: float = _DEFAULT_MIN_DISTANCE,
    min_support: int = _DEFAULT_MIN_SUPPORT,
) -> list[Organization]:
    """Return copies of `organizations` with `is_novel` set from `novelty_flags`, gated by
    support: an org with no number and fewer than `min_support` incidents is never novel."""
    # Zero rows are signals-only incidents (PLAN C9): no text evidence, so they neither
    # pull a centroid nor let an org be "far from everything".
    id_to_vector = {incident.id: embeddings[i] for i, incident in enumerate(incidents) if embeddings[i].any()}
    centroids = []
    sizes = []
    for org in organizations:
        vectors = [id_to_vector[m] for m in org.members if m in id_to_vector]
        centroids.append(np.mean(vectors, axis=0) if vectors else np.zeros(embeddings.shape[1]))
        sizes.append(len(org.members))

    flags = novelty_flags(
        np.array(centroids), sizes, max_novel_size=max_novel_size, min_distance=min_distance
    )
    has_text = [bool(centroid.any()) for centroid in centroids]
    return [
        org.model_copy(update={"is_novel": flag and text and _supported(org, min_support)})
        for org, flag, text in zip(organizations, flags, has_text, strict=True)
    ]


def _supported(org: Organization, min_support: int) -> bool:
    return bool(org.numbers) or len(org.members) >= min_support


def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    norm = float(np.linalg.norm(a) * np.linalg.norm(b))
    if norm == 0.0:
        return 1.0
    return 1.0 - float(np.dot(a, b)) / norm
