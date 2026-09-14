"""SMOKE tests for `qorgan.analytics.pipeline` — synthetic embeddings, offline."""

from datetime import datetime, timedelta

import numpy as np

from qorgan.analytics.pipeline import (
    analyze_incidents,
    cluster_quality,
    load_organizations_jsonl,
    write_organizations_jsonl,
)
from support.numbers import hashed, prefix, stored_report

from qorgan.data.schema import Incident, Label

_NOW = datetime(2026, 7, 31, 12, 0, 0)


def _incident(iid, text, number, family, days_ago):
    return Incident(
        id=iid, dialogue_id=iid, transcript=text, label=Label(risk=0.9),
        number_hash=hashed(number), number_prefix=prefix(number), timestamp=_NOW - timedelta(days=days_ago), script_family=family,
    )


def _fixture():
    rng = np.random.default_rng(0)
    incidents, rows = [], []
    for i in range(12):
        incidents.append(_incident(f"a{i}", "банк код", "+7 700 000 00 01", "bank", 20))
        rows.append([1.0, 0.0, 0.0] + rng.normal(0, 0.02, 3).tolist())
    for i in range(12):
        incidents.append(_incident(f"b{i}", "следователь дело", "+7 701 000 00 02", "police", 15))
        rows.append([0.0, 1.0, 0.0] + rng.normal(0, 0.02, 3).tolist())
    for i in range(2):
        incidents.append(_incident(f"x{i}", "крипто раздача", "+7 702 000 00 03", "crypto_new", 1))
        rows.append([0.0, 0.0, 1.0] + rng.normal(0, 0.02, 3).tolist())
    return incidents, np.array(rows, dtype=np.float32)


def test_analyze_returns_ranked_orgs_with_novel_flag():
    incidents, embeddings = _fixture()
    orgs = analyze_incidents(incidents, embeddings=embeddings, now=_NOW, min_cluster_size=3)

    assert len(orgs) >= 3
    # sorted by priority descending
    priorities = [o.priority for o in orgs]
    assert priorities == sorted(priorities, reverse=True)
    # the small recent crypto pair is flagged novel
    novel = [o for o in orgs if o.is_novel]
    assert any(set(o.members) == {"x0", "x1"} for o in novel)


def test_cluster_quality_recovers_families():
    incidents, embeddings = _fixture()
    orgs = analyze_incidents(incidents, embeddings=embeddings, now=_NOW, min_cluster_size=3)
    quality = cluster_quality(orgs, incidents)
    assert quality["purity"] >= 0.9
    assert quality["ari"] > 0.5
    assert quality["n_novel"] >= 1


def test_write_and_load_organizations_round_trips(tmp_path):
    incidents, embeddings = _fixture()
    orgs = analyze_incidents(incidents, embeddings=embeddings, now=_NOW, min_cluster_size=3)
    path = tmp_path / "organizations.jsonl"
    write_organizations_jsonl(orgs, path)
    loaded = load_organizations_jsonl(path)
    assert [o.id for o in loaded] == [o.id for o in orgs]
