"""Tests for `qorgan.analytics.novelty` — pure novelty flags + org wrapper."""

import numpy as np
import pytest

from qorgan.analytics.novelty import flag_novel_organizations, novelty_flags
from qorgan.data.schema import Organization


def test_small_far_org_is_flagged_novel():
    centroids = np.array([[1.0, 0.0, 0.0], [0.98, 0.02, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    sizes = [50, 40, 3]
    flags = novelty_flags(centroids, sizes, max_novel_size=5, min_distance=0.5)
    assert flags == [False, False, True]  # only the small, far one


def test_small_but_close_org_not_novel():
    centroids = np.array([[1.0, 0.0, 0.0], [0.99, 0.01, 0.0]], dtype=np.float32)
    flags = novelty_flags(centroids, [50, 3], max_novel_size=5, min_distance=0.5)
    assert flags == [False, False]  # small one is near the big one


def test_no_large_orgs_means_nothing_novel():
    centroids = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    flags = novelty_flags(centroids, [2, 3], max_novel_size=5, min_distance=0.5)
    assert flags == [False, False]


def test_length_mismatch_raises():
    with pytest.raises(ValueError):
        novelty_flags(np.zeros((2, 3), dtype=np.float32), [1], max_novel_size=5, min_distance=0.5)


def test_flag_novel_organizations_sets_is_novel_on_the_new_scheme():
    # two big established orgs + a small far "new scheme"
    embeddings = np.array(
        [[1.0, 0.0, 0.0]] * 6 + [[0.0, 1.0, 0.0]] * 6 + [[0.0, 0.0, 1.0]] * 2, dtype=np.float32
    )
    ids = [f"a{i}" for i in range(6)] + [f"b{i}" for i in range(6)] + ["x0", "x1"]
    from qorgan.data.schema import Incident, Label

    incidents = [Incident(id=i, dialogue_id=i, transcript="t", label=Label(risk=0.9)) for i in ids]
    orgs = [
        Organization(id="org_0", members=tuple(f"a{i}" for i in range(6))),
        Organization(id="org_1", members=tuple(f"b{i}" for i in range(6))),
        Organization(id="org_2", members=("x0", "x1")),
    ]
    flagged = flag_novel_organizations(orgs, incidents, embeddings, max_novel_size=3, min_distance=0.5)
    by_id = {o.id: o for o in flagged}
    assert by_id["org_2"].is_novel is True
    assert by_id["org_0"].is_novel is False
    assert by_id["org_1"].is_novel is False


def test_a_number_less_singleton_is_never_a_novel_scheme():
    """PLAN C10 (from the C8 stress table): one incident with no linkable number is an
    anomaly, not a scheme -- novelty needs support: a number or >= 2 incidents."""
    from qorgan.data.schema import Incident, Label
    from support.numbers import hashed

    embeddings = np.array(
        [[1.0, 0.0, 0.0]] * 6 + [[0.0, 0.0, 1.0]] + [[0.0, 1.0, 0.0]] + [[0.0, 0.0, 1.0]] * 2, dtype=np.float32
    )
    ids = [f"a{i}" for i in range(6)] + ["lone", "numbered", "pair0", "pair1"]
    incidents = [Incident(id=i, dialogue_id=i, transcript="t", label=Label(risk=0.9)) for i in ids]
    orgs = [
        Organization(id="big", members=tuple(f"a{i}" for i in range(6))),
        Organization(id="lone", members=("lone",)),  # far, but one call and no number
        Organization(id="numbered", members=("numbered",), numbers=(hashed("+7 700 111 22 33"),)),  # far, one call, a number
        Organization(id="pair", members=("pair0", "pair1")),  # far, two calls
    ]
    by_id = {o.id: o for o in flag_novel_organizations(orgs, incidents, embeddings, max_novel_size=3, min_distance=0.5)}
    assert by_id["lone"].is_novel is False
    assert by_id["numbered"].is_novel is True
    assert by_id["pair"].is_novel is True
    assert by_id["big"].is_novel is False


def test_an_org_with_no_text_evidence_is_never_novel_even_with_a_number():
    """A zero embedding row (signals-only incident, PLAN C9) carries no text evidence."""
    from qorgan.data.schema import Incident, Label
    from support.numbers import hashed

    embeddings = np.array([[1.0, 0.0, 0.0]] * 6 + [[0.0, 0.0, 0.0]], dtype=np.float32)
    ids = [f"a{i}" for i in range(6)] + ["signals"]
    incidents = [Incident(id=i, dialogue_id=i, transcript="t" if i != "signals" else "", label=Label(risk=0.9)) for i in ids]
    orgs = [
        Organization(id="big", members=tuple(f"a{i}" for i in range(6))),
        Organization(id="signals", members=("signals",), numbers=(hashed("+7 700 111 22 33"),)),
    ]
    by_id = {o.id: o for o in flag_novel_organizations(orgs, incidents, embeddings, max_novel_size=3, min_distance=0.5)}
    assert by_id["signals"].is_novel is False
