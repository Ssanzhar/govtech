"""TDD tests for `qorgan.eval.cluster` (PLAN_2026-09 C8): cluster quality vs seeded
script families under number-availability stress, with subsample intervals."""

from __future__ import annotations

import numpy as np
import pytest

from qorgan.data.schema import Incident, Label
from qorgan.eval.cluster import (
    DEFAULT_CONDITIONS,
    Condition,
    drop_numbers,
    evaluate_conditions,
    quality,
    render_table,
    subsample_intervals,
)
from support.numbers import hashed, prefix

FAMILIES = {"bank": 12, "police": 12, "prize": 10, "novel": 2}
DIM = 8


def _seeded() -> tuple[list[Incident], np.ndarray]:
    rng = np.random.default_rng(0)
    incidents: list[Incident] = []
    rows: list[np.ndarray] = []
    for f_index, (family, count) in enumerate(FAMILIES.items()):
        centre = np.zeros(DIM)
        centre[f_index] = 1.0
        number = f"+7 700 {100 + f_index:03d} 00 00"
        for k in range(count):
            iid = f"{family}-{k}"
            incidents.append(
                Incident(
                    id=iid, dialogue_id=iid, transcript=f"{family} script {k}", label=Label(risk=0.9),
                    number_hash=hashed(number), number_prefix=prefix(number), script_family=family,
                )
            )
            vector = centre + 0.05 * rng.standard_normal(DIM)
            rows.append(vector / np.linalg.norm(vector))
    return incidents, np.vstack(rows)


def test_quality_as_seeded_recovers_every_family():
    incidents, embeddings = _seeded()
    q = quality(incidents, embeddings, text_merge=False)
    assert q["purity"] == 1.0 and q["ari"] == 1.0
    assert q["n_organizations"] == len(FAMILIES) and q["multi_member_share"] == 1.0


def test_drop_numbers_is_deterministic_and_immutable():
    incidents, _ = _seeded()
    a = drop_numbers(incidents, keep_fraction=0.5, seed=1)
    b = drop_numbers(incidents, keep_fraction=0.5, seed=1)
    assert [i.number_hash for i in a] == [i.number_hash for i in b]
    assert sum(1 for i in a if i.number_hash is None) == len(incidents) // 2
    assert all(i.number_hash is not None for i in incidents)  # inputs untouched
    assert all(i.number_hash is None and i.number_prefix is None for i in drop_numbers(incidents, keep_fraction=0.0, seed=1))
    assert drop_numbers(incidents, keep_fraction=1.0, seed=1) == tuple(incidents)
    with pytest.raises(ValueError):
        drop_numbers(incidents, keep_fraction=1.5, seed=1)


def test_without_numbers_and_without_text_everything_is_a_singleton():
    incidents, embeddings = _seeded()
    q = quality(drop_numbers(incidents, keep_fraction=0.0, seed=1), embeddings, text_merge=False)
    assert q["multi_member_share"] == 0.0 and q["ari"] == pytest.approx(0.0) and q["purity"] == 1.0


def test_text_overlay_recovers_well_separated_families_without_numbers():
    pytest.importorskip("hdbscan")  # the optional text overlay (off by default)
    incidents, embeddings = _seeded()
    q = quality(drop_numbers(incidents, keep_fraction=0.0, seed=1), embeddings, text_merge=True)
    assert q["purity"] >= 0.9 and q["ari"] >= 0.8


def test_subsample_intervals_bracket_the_point_estimate():
    incidents, embeddings = _seeded()
    intervals = subsample_intervals(incidents, embeddings, text_merge=False, n_resamples=10, seed=3)
    for name in ("purity", "ari"):
        assert intervals[name].low <= 1.0 <= intervals[name].high
    with pytest.raises(ValueError):
        subsample_intervals(incidents, embeddings, text_merge=False, n_resamples=0, seed=3)


def test_evaluate_conditions_and_render_table():
    pytest.importorskip("hdbscan")  # the optional text overlay (off by default)
    incidents, embeddings = _seeded()
    conditions = (Condition("as seeded", 1.0, False), Condition("text only", 0.0, True))
    results = evaluate_conditions(incidents, embeddings, conditions, n_resamples=5, seed=0)
    assert list(results) == ["as seeded", "text only"]
    table = render_table(results)
    assert "| Condition | Purity [95% CI] | ARI [95% CI] | Orgs | Multi-member share | Novel flagged |" in table
    assert "as seeded" in table and "1.000 [" in table
    assert len(DEFAULT_CONDITIONS) == 4
