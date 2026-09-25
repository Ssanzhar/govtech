"""SMOKE test for `qorgan.analytics.cluster` — synthetic embeddings, offline."""

import numpy as np

from qorgan.analytics.cluster import cluster_incidents
from support.numbers import hashed, prefix, stored_report

from qorgan.data.schema import Incident, Label


def _incident(iid, text, number):
    return Incident(id=iid, dialogue_id=iid, transcript=text, label=Label(risk=0.9), number_hash=hashed(number), number_prefix=prefix(number))


def _fixture():
    # Each family operates from one number -> the number graph gives one org per family
    # (scam scripts are text-inseparable, so numbers are the reliable link).
    incidents = []
    for i in range(8):
        incidents.append(_incident(f"a{i}", "банк код из смс", "+7 700 000 00 01"))
    for i in range(8):
        incidents.append(_incident(f"b{i}", "следователь дело финпол", "+7 701 000 00 02"))
    for i in range(2):
        incidents.append(_incident(f"x{i}", "крипто раздача удвоим баланс", "+7 702 000 00 03"))
    return incidents


def test_number_graph_recovers_one_organization_per_operating_number():
    incidents = _fixture()
    orgs = cluster_incidents(incidents)  # number-primary, no embeddings needed

    members_of = {org.id: set(org.members) for org in orgs}
    a_ids = {f"a{i}" for i in range(8)}
    b_ids = {f"b{i}" for i in range(8)}
    x_ids = {"x0", "x1"}

    assert any(a_ids == m for m in members_of.values())
    assert any(b_ids == m for m in members_of.values())
    novel_org = next(org for org in orgs if x_ids == set(org.members))
    assert novel_org.numbers == (hashed("+7 702 000 00 03"),)  # linked by digest, never raw
    assert novel_org.representative_script == "крипто раздача удвоим баланс"


def test_incidents_without_numbers_are_singletons():
    incidents = [_incident("s1", "текст один", None), _incident("s2", "текст два", None)]
    orgs = cluster_incidents(incidents)
    assert len(orgs) == 2


def test_empty_incidents_returns_empty():
    assert cluster_incidents([], np.zeros((0, 3), dtype=np.float32)) == []


def test_length_mismatch_raises():
    import pytest

    with pytest.raises(ValueError):
        cluster_incidents([_incident("a", "x", "n")], np.zeros((2, 3), dtype=np.float32))
