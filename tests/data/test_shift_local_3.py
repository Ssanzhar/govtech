"""`shift_local_3` (D63 v2.1 A6 verdict, PLAN_2026-10 A7): the judge for v2.2. Same briefs, model
and targets as `shift_local_2`, a new seed and prefix, disjoint from BOTH earlier batches, never
published. `shift_local_2` was looked at once (v2.1 A5), so it cannot judge v2.2."""

from __future__ import annotations

from pathlib import Path

import pytest

from qorgan.data import shift_local_plan as plan
from qorgan.data.shift_local_set import checked_splits

_REPO = Path(__file__).resolve().parents[2]
_THIRD = "shift_local_3"


def test_third_batch_is_registered_with_its_own_seed_and_prefix():
    third = plan.LOCAL_SPLITS[_THIRD]
    others = [plan.LOCAL_SPLITS[n] for n in ("shift_local", "shift_local_2")]
    assert third.plan_seed not in {o.plan_seed for o in others}
    assert third.slot_prefix == "sl3" and third.slot_prefix not in {o.slot_prefix for o in others}
    assert set(third.prior) == {"shift_local", "shift_local_2"}
    assert third.near_duplicate_jaccard == plan.LOCAL_SPLITS["shift_local_2"].near_duplicate_jaccard


def test_third_batch_has_the_second_batch_design():
    assert dict(plan.LOCAL_SPLITS[_THIRD].language_targets) == dict(plan.LOCAL_SPLITS["shift_local_2"].language_targets)
    briefs = plan.load_briefs(_REPO / "configs" / "shift_local_briefs.yaml")
    third, second = plan.plan_for(_THIRD, briefs), plan.plan_for("shift_local_2", briefs)
    assert len(third) == len(second)
    assert not {s.slot_id for s in third} & {s.slot_id for s in second}
    assert [s.seed for s in third] != [s.seed for s in second]


def test_third_batch_is_checked_against_both_earlier_batches():
    assert {"shift_local", "shift_local_2"} <= set(checked_splits(_THIRD))


def test_shift_local_3_is_never_published():
    from qorgan.data.publish_guard import NEVER_PUBLISHED, PublishRefused, assert_publishable

    assert {f"{_THIRD}.jsonl", f"{_THIRD}.manifest.json"} <= set(NEVER_PUBLISHED)
    with pytest.raises(PublishRefused, match=_THIRD):
        assert_publishable([f"{_THIRD}.jsonl"])


def test_shift_local_3_raw_and_processed_files_are_gitignored():
    import subprocess

    for path in (f"data/raw/{_THIRD}/kk.jsonl", f"data/processed/{_THIRD}.jsonl"):
        assert subprocess.run(["git", "check-ignore", "-q", path], cwd=_REPO, check=False).returncode == 0, path
