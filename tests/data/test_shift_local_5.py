"""`shift_local_5` (ADR D74): the judge for the v3.1 retrain. The local model writes it from the
`shift_local_4` briefs PLUS `shift_local_briefs_v31.yaml` (ordinary family and everyday calls,
the classes A6 (c) failed on), under a new seed and prefix, disjoint from every earlier batch,
never published. `shift_local_4` was looked at once (v3), so it cannot judge v3.1."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest

from qorgan.data import shift_local_plan as plan
from qorgan.data.shift_local_set import checked_splits

_REPO = Path(__file__).resolve().parents[2]
_CONFIGS = _REPO / "configs"
_FIFTH = "shift_local_5"
_EARLIER = ("shift_local", "shift_local_2", "shift_local_3", "shift_local_4")


def test_fifth_batch_is_registered_with_its_own_seed_prefix_and_briefs():
    fifth = plan.LOCAL_SPLITS[_FIFTH]
    others = [plan.LOCAL_SPLITS[n] for n in _EARLIER]
    assert fifth.plan_seed not in {o.plan_seed for o in others}
    assert fifth.slot_prefix == "sl5" and fifth.slot_prefix not in {o.slot_prefix for o in others}
    assert set(fifth.prior) == set(_EARLIER)
    assert fifth.briefs_files == plan.LOCAL_SPLITS["shift_local_4"].briefs_files + ("shift_local_briefs_v31.yaml",)
    assert dict(fifth.language_targets) == {"kk": 96, "ru": 64, "mixed": 0, "en": 0}


def test_v31_briefs_are_legitimate_family_and_everyday_calls():
    briefs = plan.load_briefs(_CONFIGS / "shift_local_briefs_v31.yaml")
    assert len(briefs) >= 12 and all(b.label == "legit" and not b.tactics for b in briefs)
    kinds = Counter(b.id.split("_")[1] for b in briefs)
    assert kinds["family"] >= 5 and kinds["chat"] >= 5


def test_fifth_batch_plan_uses_the_new_briefs_and_is_disjoint():
    fifth = plan.plan_for(_FIFTH, plan.load_split_briefs(_FIFTH, _CONFIGS))
    v31_ids = {b.id for b in plan.load_briefs(_CONFIGS / "shift_local_briefs_v31.yaml")}
    assert len(fifth) == 160 and sum(s.brief.id in v31_ids for s in fifth) >= 20
    earlier = {s.slot_id for n in _EARLIER for s in plan.plan_for(n, plan.load_split_briefs(n, _CONFIGS))}
    assert not {s.slot_id for s in fifth} & earlier


def test_fifth_batch_is_checked_against_every_earlier_batch():
    assert set(_EARLIER) <= set(checked_splits(_FIFTH))


def test_shift_local_5_is_never_published():
    from qorgan.data.publish_guard import NEVER_PUBLISHED, PublishRefused, assert_publishable

    assert {f"{_FIFTH}.jsonl", f"{_FIFTH}.manifest.json"} <= set(NEVER_PUBLISHED)
    with pytest.raises(PublishRefused, match=_FIFTH):
        assert_publishable([f"{_FIFTH}.jsonl"])


def test_shift_local_5_raw_and_processed_files_are_gitignored():
    import subprocess

    for path in (f"data/raw/{_FIFTH}/kk.jsonl", f"data/processed/{_FIFTH}.jsonl"):
        assert subprocess.run(["git", "check-ignore", "-q", path], cwd=_REPO, check=False).returncode == 0, path
