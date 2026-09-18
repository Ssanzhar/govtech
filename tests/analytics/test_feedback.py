"""TDD tests for `qorgan.analytics.feedback` (PLAN_2026-09 C6): analyst confirm / dismiss /
merge events, stored append-only, applied at read time, following the *operation* (its
numbers, else its members) rather than the re-assigned `org_<n>` id."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from qorgan.analytics.feedback import (
    DISMISSED_PRIORITY_FACTOR,
    FeedbackEvent,
    append_feedback,
    apply_feedback,
    load_feedback,
    snapshot_for,
)
from qorgan.data.schema import Organization
from support.numbers import hashed

N1, N2, N3 = hashed("+7 700 111 11 11"), hashed("+7 700 222 22 22"), hashed("+7 700 333 33 33")
NOW = datetime(2026, 9, 18, 9, 0, tzinfo=UTC)


def _orgs():
    return [
        Organization(id="org_0", members=("a0", "a1", "a2"), numbers=(N1,), priority=0.9, is_novel=False),
        Organization(id="org_1", members=("b0", "b1"), numbers=(N2,), priority=0.8, is_novel=True),
        Organization(id="org_2", members=("c0",), numbers=(), priority=0.5, is_novel=False),
    ]


def _event(action, org, *, target=None, at=NOW, analyst="analyst-1", note=None):
    return FeedbackEvent(
        timestamp=at, analyst_id=analyst, action=action, org=snapshot_for(org),
        target=snapshot_for(target) if target is not None else None, note=note,
    )


def test_snapshot_records_numbers_and_members_not_the_index_id():
    snap = snapshot_for(_orgs()[0])
    assert snap.numbers == (N1,) and snap.members == ("a0", "a1", "a2")


def test_dismiss_decays_priority_and_clears_novelty():
    orgs = _orgs()
    applied = {o.id: o for o in apply_feedback(orgs, [_event("dismiss", orgs[1])])}
    assert applied["org_1"].priority == pytest.approx(0.8 * DISMISSED_PRIORITY_FACTOR)
    assert applied["org_1"].is_novel is False and applied["org_1"].feedback == "dismissed"
    assert applied["org_0"].priority == 0.9 and applied["org_0"].feedback is None
    assert orgs[1].priority == 0.8  # inputs untouched


def test_confirm_marks_the_org_and_keeps_its_priority():
    orgs = _orgs()
    applied = {o.id: o for o in apply_feedback(orgs, [_event("confirm", orgs[1])])}
    assert applied["org_1"].feedback == "confirmed" and applied["org_1"].priority == 0.8 and applied["org_1"].is_novel is True


def test_latest_event_wins():
    orgs = _orgs()
    events = [
        _event("dismiss", orgs[0], at=datetime(2026, 9, 17, 9, 0, tzinfo=UTC)),
        _event("confirm", orgs[0], at=NOW),
    ]
    applied = {o.id: o for o in apply_feedback(orgs, events)}
    assert applied["org_0"].feedback == "confirmed" and applied["org_0"].priority == 0.9


def test_feedback_follows_the_operation_after_reclustering():
    """Ingest re-assigns org ids and grows members; the number digest still identifies it."""
    orgs = _orgs()
    event = _event("dismiss", orgs[1])  # org_1 = the N2 operation
    reclustered = [
        Organization(id="org_5", members=("b0", "b1", "b9"), numbers=(N2,), priority=0.85, is_novel=True),
        Organization(id="org_1", members=("a0", "a1"), numbers=(N1,), priority=0.9),  # a different operation now wears the old id
    ]
    applied = {o.id: o for o in apply_feedback(reclustered, [event])}
    assert applied["org_5"].feedback == "dismissed" and applied["org_5"].is_novel is False
    assert applied["org_1"].feedback is None


def test_feedback_without_numbers_matches_by_member_overlap():
    orgs = _orgs()
    event = _event("confirm", orgs[2])  # org_2 has no number, one member
    grown = [Organization(id="org_9", members=("c0", "c1"), priority=0.5)]
    unrelated = [Organization(id="org_9", members=("z0", "z1"), priority=0.5)]
    assert apply_feedback(grown, [event])[0].feedback == "confirmed"
    assert apply_feedback(unrelated, [event])[0].feedback is None


def test_merge_unions_the_two_operations():
    orgs = _orgs()
    applied = apply_feedback(orgs, [_event("merge", orgs[1], target=orgs[0])])
    by_id = {o.id: o for o in applied}
    assert len(applied) == 2
    merged = by_id["org_0"]
    assert set(merged.members) == {"a0", "a1", "a2", "b0", "b1"} and set(merged.numbers) == {N1, N2}
    assert merged.priority == 0.9 and merged.feedback == "merged"
    assert "org_1" not in by_id and by_id["org_2"].feedback is None


def test_merge_needs_a_target_and_actions_are_validated():
    orgs = _orgs()
    with pytest.raises(ValidationError):
        _event("merge", orgs[1])
    with pytest.raises(ValidationError):
        _event("promote", orgs[1])
    with pytest.raises(ValidationError):
        _event("dismiss", orgs[1], note="caller +7 700 555 66 77")  # content-free like the audit log


def test_events_round_trip_through_the_file(tmp_path):
    orgs = _orgs()
    path = tmp_path / "org_feedback.jsonl"
    first, second = _event("dismiss", orgs[1]), _event("confirm", orgs[0], note="matches case file")
    append_feedback(first, path)
    append_feedback(second, path)
    assert load_feedback(path) == [first, second]
    assert load_feedback(tmp_path / "missing.jsonl") == []
