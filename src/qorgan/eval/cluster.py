"""Cluster-quality evaluation with intervals (PLAN_2026-09 C8).

Purity / ARI of the recovered organizations against the seeded `script_family` ground
truth, under **number-availability stress**: the seeds give every family its own reused
number, so the number graph recovers them perfectly and a point estimate of 1.00 says
nothing about the realistic case where callers rotate SIMs. Each condition drops the number
from a seeded fraction of incidents (and optionally enables the HDBSCAN text overlay), and
every metric carries a percentile interval over 80 % subsamples -- a *stability* interval,
which is the right notion for a clustering.

Run: `python -m qorgan.eval.cluster [--processed-dir data/processed] [--resamples 50]`
Caveat printed with the table: the seeds are synthetic (`scripts/demo_seed.py`).
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from qorgan.analytics import cluster as cluster_mod
from qorgan.analytics import novelty as novelty_mod
from qorgan.analytics import pipeline as pipeline_mod
from qorgan.analytics.pipeline import load_embeddings_npz
from qorgan.config import get_config
from qorgan.data.incident_seed import load_incidents_jsonl
from qorgan.data.schema import Incident
from qorgan.eval import metrics
from qorgan.eval.intervals import DEFAULT_CONFIDENCE, Interval

_SUBSAMPLE_FRACTION = 0.8
_DEFAULT_RESAMPLES = 50
_DEFAULT_SEED = 42
_METHOD = "subsample-percentile"
_UNKNOWN_FAMILY = "unknown"


@dataclass(frozen=True)
class Condition:
    name: str
    number_keep_fraction: float
    text_merge: bool


DEFAULT_CONDITIONS: tuple[Condition, ...] = (
    Condition("numbers as seeded (shipped)", 1.0, False),
    Condition("numbers rotated for 50 % of calls", 0.5, False),
    Condition("numbers rotated for 50 % + text overlay", 0.5, True),
    Condition("text only (no numbers)", 0.0, True),
)


# --- building blocks -------------------------------------------------------------------------


def drop_numbers(incidents: Sequence[Incident], *, keep_fraction: float, seed: int) -> tuple[Incident, ...]:
    """Copies of `incidents` where a seeded random `1 - keep_fraction` share lost its number."""
    if not 0.0 <= keep_fraction <= 1.0:
        raise ValueError(f"keep_fraction must be in [0, 1], got {keep_fraction}")
    if keep_fraction >= 1.0:
        return tuple(incidents)
    rng = np.random.default_rng(seed)
    n_drop = int(round(len(incidents) * (1.0 - keep_fraction)))
    dropped = set(rng.choice(len(incidents), size=n_drop, replace=False).tolist()) if n_drop else set()
    return tuple(
        incident.model_copy(update={"number_hash": None, "number_prefix": None}) if i in dropped else incident
        for i, incident in enumerate(incidents)
    )


def quality(incidents: Sequence[Incident], embeddings: np.ndarray, *, text_merge: bool) -> dict:
    """Purity, ARI, org count, share of incidents in multi-member orgs, novel-flag count."""
    organizations = cluster_mod.cluster_incidents(incidents, embeddings, use_text_merge=text_merge)
    # The shipped pipeline's novelty knobs, so "novel flagged" means what the analyst sees.
    organizations = novelty_mod.flag_novel_organizations(
        organizations, incidents, embeddings,
        max_novel_size=pipeline_mod._NOVELTY_MAX_SIZE, min_distance=pipeline_mod._NOVELTY_MIN_DISTANCE,
    )
    org_of = {member: org.id for org in organizations for member in org.members}
    labels_true = [incident.script_family or _UNKNOWN_FAMILY for incident in incidents]
    labels_pred = [org_of.get(incident.id, incident.id) for incident in incidents]
    multi = sum(len(org.members) for org in organizations if len(org.members) > 1)
    return {
        "purity": metrics.purity(labels_true, labels_pred),
        "ari": metrics.adjusted_rand(labels_true, labels_pred),
        "n_organizations": len(organizations),
        "multi_member_share": multi / len(incidents) if incidents else 0.0,
        "n_novel": sum(1 for org in organizations if org.is_novel),
    }


def subsample_intervals(
    incidents: Sequence[Incident],
    embeddings: np.ndarray,
    *,
    text_merge: bool,
    n_resamples: int = _DEFAULT_RESAMPLES,
    seed: int = _DEFAULT_SEED,
    fraction: float = _SUBSAMPLE_FRACTION,
    confidence: float = DEFAULT_CONFIDENCE,
) -> dict[str, Interval]:
    """Percentile intervals of purity / ARI / multi-member share over `fraction` subsamples
    (without replacement -- duplicated points would distort HDBSCAN densities)."""
    if n_resamples <= 0:
        raise ValueError(f"n_resamples must be > 0, got {n_resamples}")
    rng = np.random.default_rng(seed)
    n = len(incidents)
    size = max(2, int(round(n * fraction)))
    samples: dict[str, list[float]] = {"purity": [], "ari": [], "multi_member_share": []}
    for _ in range(n_resamples):
        idx = np.sort(rng.choice(n, size=size, replace=False))
        q = quality([incidents[i] for i in idx], embeddings[idx], text_merge=text_merge)
        for name in samples:
            samples[name].append(float(q[name]))
    alpha = 1.0 - confidence
    return {
        name: Interval(
            low=float(np.percentile(values, 100 * alpha / 2)),
            high=float(np.percentile(values, 100 * (1 - alpha / 2))),
            confidence=confidence,
            method=_METHOD,
        )
        for name, values in samples.items()
    }


def evaluate_conditions(
    incidents: Sequence[Incident],
    embeddings: np.ndarray,
    conditions: Sequence[Condition] = DEFAULT_CONDITIONS,
    *,
    n_resamples: int = _DEFAULT_RESAMPLES,
    seed: int = _DEFAULT_SEED,
) -> dict[str, dict]:
    results: dict[str, dict] = {}
    for condition in conditions:
        stressed = drop_numbers(incidents, keep_fraction=condition.number_keep_fraction, seed=seed)
        results[condition.name] = {
            "point": quality(stressed, embeddings, text_merge=condition.text_merge),
            "intervals": subsample_intervals(
                stressed, embeddings, text_merge=condition.text_merge, n_resamples=n_resamples, seed=seed
            ),
        }
    return results


def render_table(results: dict[str, dict]) -> str:
    lines = [
        "| Condition | Purity [95% CI] | ARI [95% CI] | Orgs | Multi-member share | Novel flagged |",
        "|---|---|---|---|---|---|",
    ]
    for name, row in results.items():
        point, ci = row["point"], row["intervals"]
        lines.append(
            f"| {name} | {point['purity']:.3f} [{ci['purity'].low:.3f}, {ci['purity'].high:.3f}] "
            f"| {point['ari']:.3f} [{ci['ari'].low:.3f}, {ci['ari'].high:.3f}] "
            f"| {point['n_organizations']} | {point['multi_member_share']:.2f} | {point['n_novel']} |"
        )
    lines.append(
        "\nIntervals: 2.5-97.5 percentiles over 80 % subsamples (clustering *stability*); a full-set "
        "point estimate can sit outside them when HDBSCAN merges differently on fewer points."
    )
    return "\n".join(lines)


# --- CLI -------------------------------------------------------------------------------------


def load_seeded(processed_dir: Path) -> tuple[list[Incident], np.ndarray]:
    """Incidents + their cached embeddings, aligned by id (raises if the cache is stale)."""
    incidents = load_incidents_jsonl(processed_dir / "incidents.jsonl")
    ids, matrix = load_embeddings_npz(processed_dir / "incident_embeddings.npz")
    position = {incident_id: i for i, incident_id in enumerate(ids)}
    missing = [incident.id for incident in incidents if incident.id not in position]
    if missing:
        raise ValueError(f"embedding cache lacks {len(missing)} incidents; run python -m qorgan.analytics.pipeline")
    return incidents, matrix[[position[incident.id] for incident in incidents]]


def main(argv: Sequence[str] | None = None) -> None:  # pragma: no cover - CLI
    parser = argparse.ArgumentParser(description="Cluster quality vs seeded families, with stability intervals.")
    parser.add_argument("--processed-dir", type=Path, default=get_config().data_dir / "processed")
    parser.add_argument("--resamples", type=int, default=_DEFAULT_RESAMPLES)
    parser.add_argument("--seed", type=int, default=_DEFAULT_SEED)
    args = parser.parse_args(argv)
    incidents, embeddings = load_seeded(args.processed_dir)
    families = {incident.script_family for incident in incidents}
    print(f"{len(incidents)} seeded incidents, {len(families)} script families; "
          f"intervals = {int(100 * _SUBSAMPLE_FRACTION)} % subsamples x {args.resamples}")
    print(render_table(evaluate_conditions(incidents, embeddings, n_resamples=args.resamples, seed=args.seed)))
    print("\nCaveat: the seeds are synthetic (scripts/demo_seed.py); every family reuses one number by "
          "construction, so the 'as seeded' row is an upper bound, not a field result.")


if __name__ == "__main__":
    main()
