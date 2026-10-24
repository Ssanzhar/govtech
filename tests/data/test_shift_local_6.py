"""`shift_local_6` (ADR D75): the judge for the v1 + v3.1 risk ensemble. `shift_local_5` was
looked at once (v3.1), so it cannot judge the ensemble. Same briefs and size as the fifth batch
under a new seed and prefix, disjoint from every earlier batch, never published."""

from __future__ import annotations

from pathlib import Path

import pytest

from qorgan.data import shift_local_plan as plan
from qorgan.data.shift_local_set import checked_splits

_REPO = Path(__file__).resolve().parents[2]
_CONFIGS = _REPO / "configs"
_SIXTH = "shift_local_6"
_EARLIER = ("shift_local", "shift_local_2", "shift_local_3", "shift_local_4", "shift_local_5")


def test_sixth_batch_is_registered_with_its_own_seed_prefix_and_the_fifth_batch_briefs():
    sixth = plan.LOCAL_SPLITS[_SIXTH]
    others = [plan.LOCAL_SPLITS[n] for n in _EARLIER]
    assert sixth.plan_seed not in {o.plan_seed for o in others}
    assert sixth.slot_prefix == "sl6" and sixth.slot_prefix not in {o.slot_prefix for o in others}
    assert set(sixth.prior) == set(_EARLIER)
    assert sixth.briefs_files == plan.LOCAL_SPLITS["shift_local_5"].briefs_files
    assert dict(sixth.language_targets) == {"kk": 96, "ru": 64, "mixed": 0, "en": 0}
    assert sixth.adr == "D75"


def test_sixth_batch_plan_is_new_and_disjoint():
    sixth = plan.plan_for(_SIXTH, plan.load_split_briefs(_SIXTH, _CONFIGS))
    fifth = plan.plan_for("shift_local_5", plan.load_split_briefs("shift_local_5", _CONFIGS))
    assert len(sixth) == 160
    assert [s.brief.id for s in sixth] != [s.brief.id for s in fifth]  # a new draw, not a relabel
    earlier = {s.slot_id for n in _EARLIER for s in plan.plan_for(n, plan.load_split_briefs(n, _CONFIGS))}
    assert not {s.slot_id for s in sixth} & earlier


def test_sixth_batch_is_checked_against_every_earlier_batch():
    assert set(_EARLIER) <= set(checked_splits(_SIXTH))


def test_shift_local_6_is_never_published():
    from qorgan.data.publish_guard import NEVER_PUBLISHED, PublishRefused, assert_publishable

    assert {f"{_SIXTH}.jsonl", f"{_SIXTH}.manifest.json"} <= set(NEVER_PUBLISHED)
    with pytest.raises(PublishRefused, match=_SIXTH):
        assert_publishable([f"{_SIXTH}.jsonl"])


def test_shift_local_6_raw_and_processed_files_are_gitignored():
    import subprocess

    for path in (f"data/raw/{_SIXTH}/kk.jsonl", f"data/processed/{_SIXTH}.jsonl"):
        assert subprocess.run(["git", "check-ignore", "-q", path], cwd=_REPO, check=False).returncode == 0, path
