"""`shift_local_4` (ADR D73): the judge for the v3 retrain. The local model writes it from the
`shift_local` briefs PLUS `shift_local_briefs_v3.yaml` (the five v3 tactics and their legitimate
twins), under a new seed and prefix, disjoint from every earlier batch, never published.
`shift_local_3` was looked at once (v2.2), so it cannot judge v3."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest

from qorgan.data import shift_local_plan as plan
from qorgan.data.shift_local_set import checked_splits

_REPO = Path(__file__).resolve().parents[2]
_CONFIGS = _REPO / "configs"
_FOURTH = "shift_local_4"
_EARLIER = ("shift_local", "shift_local_2", "shift_local_3")
_V3 = {"minor_targeting", "blackmail_extortion", "survey_pretext", "romance_trust", "charity_appeal"}


def test_fourth_batch_is_registered_with_its_own_seed_prefix_and_briefs():
    fourth = plan.LOCAL_SPLITS[_FOURTH]
    others = [plan.LOCAL_SPLITS[n] for n in _EARLIER]
    assert fourth.plan_seed not in {o.plan_seed for o in others}
    assert fourth.slot_prefix == "sl4" and fourth.slot_prefix not in {o.slot_prefix for o in others}
    assert set(fourth.prior) == set(_EARLIER)
    assert fourth.briefs_files == ("shift_local_briefs.yaml", "shift_local_briefs_v3.yaml")
    assert all(o.briefs_files == ("shift_local_briefs.yaml",) for o in others)  # earlier plans unchanged
    assert dict(fourth.language_targets) == {"kk": 96, "ru": 64, "mixed": 0, "en": 0}


def test_v3_briefs_cover_every_v3_tactic_with_scams_and_twins():
    briefs = plan.load_briefs(_CONFIGS / "shift_local_briefs_v3.yaml")
    scams = [b for b in briefs if b.label == "scam"]
    covered = Counter(t for b in scams for t in b.tactics if t in _V3)
    assert set(covered) == _V3 and min(covered.values()) >= 3
    assert sum(1 for b in briefs if b.label == "legit") >= 10
    from qorgan.taxonomy import get_taxonomy

    known = set(get_taxonomy().all_tactic_ids())
    assert all(set(b.tactics) <= known for b in briefs)


def test_split_briefs_concatenate_without_duplicate_ids():
    briefs = plan.load_split_briefs(_FOURTH, _CONFIGS)
    base = plan.load_briefs(_CONFIGS / "shift_local_briefs.yaml")
    assert briefs[: len(base)] == base and len({b.id for b in briefs}) == len(briefs)
    assert plan.load_split_briefs("shift_local_3", _CONFIGS) == base


def test_a_brief_persona_pins_the_callee_without_moving_other_slots():
    base = plan.load_briefs(_CONFIGS / "shift_local_briefs.yaml")
    pinned = tuple(b if b.label == "legit" else plan.Brief(**{**b.__dict__, "persona": "a 13-year-old schoolgirl"}) for b in base)
    a = plan.build_plan(base, seed=5, targets={"kk": 10, "ru": 0, "mixed": 0, "en": 0}, prefix="x")
    b = plan.build_plan(pinned, seed=5, targets={"kk": 10, "ru": 0, "mixed": 0, "en": 0}, prefix="x")
    for left, right in zip(a, b):
        assert left.seed == right.seed and left.turns == right.turns
        assert right.persona == ("a 13-year-old schoolgirl" if right.brief.label == "scam" else left.persona)


def test_minor_briefs_pin_a_child_callee():
    briefs = plan.load_briefs(_CONFIGS / "shift_local_briefs_v3.yaml")
    minors = [b for b in briefs if "minor_targeting" in b.tactics]
    assert minors and all(b.persona and "year-old" in b.persona for b in minors)


def test_fourth_batch_plan_mixes_old_and_v3_briefs_and_is_disjoint():
    fourth = plan.plan_for(_FOURTH, plan.load_split_briefs(_FOURTH, _CONFIGS))
    v3_ids = {b.id for b in plan.load_briefs(_CONFIGS / "shift_local_briefs_v3.yaml")}
    used = Counter(s.brief.id in v3_ids for s in fourth)
    assert len(fourth) == 160 and used[True] >= 40 and used[False] >= 80
    earlier = {s.slot_id for n in _EARLIER for s in plan.plan_for(n, plan.load_split_briefs(n, _CONFIGS))}
    assert not {s.slot_id for s in fourth} & earlier


def test_fourth_batch_is_checked_against_every_earlier_batch():
    assert set(_EARLIER) <= set(checked_splits(_FOURTH))


def test_shift_local_4_is_never_published():
    from qorgan.data.publish_guard import NEVER_PUBLISHED, PublishRefused, assert_publishable

    assert {f"{_FOURTH}.jsonl", f"{_FOURTH}.manifest.json"} <= set(NEVER_PUBLISHED)
    with pytest.raises(PublishRefused, match=_FOURTH):
        assert_publishable([f"{_FOURTH}.jsonl"])


def test_shift_local_4_raw_and_processed_files_are_gitignored():
    import subprocess

    for path in (f"data/raw/{_FOURTH}/kk.jsonl", f"data/processed/{_FOURTH}.jsonl"):
        assert subprocess.run(["git", "check-ignore", "-q", path], cwd=_REPO, check=False).returncode == 0, path
