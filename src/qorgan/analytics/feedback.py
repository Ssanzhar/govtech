"""Analyst feedback on organizations: confirm / dismiss / merge (PLAN_2026-09 C6).

Events are **append-only** (`org_feedback.jsonl`) and applied at **read time** by
`apply_feedback`, a pure function over the current organizations. Because every ingest
re-clusters and re-assigns `org_<n>` ids, an event never refers to an id: it carries a
snapshot of the operation -- its number digests, else its members -- and is matched to
whichever current organization shares a number (or most of its members). Latest event per
operation wins. Effects: `dismissed` decays priority and clears the novelty flag;
`confirmed` marks the org; `merge` unions the source into the target. Notes are
content-free like the audit log (`scrub_text` must be a fixed point).
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from qorgan.data.schema import Organization
from qorgan.data.scrub import scrub_text

FEEDBACK_FILENAME = "org_feedback.jsonl"
DISMISSED_PRIORITY_FACTOR = 0.2
_MEMBER_OVERLAP_MIN = 0.5
_MAX_NOTE_CHARS = 200

FeedbackAction = Literal["confirm", "dismiss", "merge"]


class OrgSnapshot(BaseModel):
    """What identifies an operation across re-clustering."""

    model_config = ConfigDict(frozen=True)

    numbers: tuple[str, ...] = ()
    members: tuple[str, ...] = Field(min_length=1)


class FeedbackEvent(BaseModel):
    model_config = ConfigDict(frozen=True)

    timestamp: datetime
    analyst_id: str = Field(min_length=1, max_length=64)
    action: FeedbackAction
    org: OrgSnapshot
    target: OrgSnapshot | None = None
    note: str | None = Field(default=None, max_length=_MAX_NOTE_CHARS)

    @field_validator("note", "analyst_id")
    @classmethod
    def _content_free(cls, value: str | None) -> str | None:
        if value is not None and scrub_text(value) != value:
            raise ValueError("feedback must not carry numbers or other call content")
        return value

    @model_validator(mode="after")
    def _merge_has_target(self) -> "FeedbackEvent":
        if (self.action == "merge") != (self.target is not None):
            raise ValueError("merge needs a target organization; other actions must not carry one")
        return self


def snapshot_for(org: Organization) -> OrgSnapshot:
    return OrgSnapshot(numbers=tuple(org.numbers), members=tuple(org.members))


# --- persistence -----------------------------------------------------------------------------


def append_feedback(event: FeedbackEvent, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(event.model_dump_json() + "\n")
    return path


def load_feedback(path: Path) -> list[FeedbackEvent]:
    if not path.exists():
        return []
    return [FeedbackEvent.model_validate_json(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def forget_in_feedback(path: Path, *, incident_id: str, number_hash: str | None, live_numbers: set[str]) -> int:
    """Remove a deleted report's traces from the feedback snapshots (legal review F8 / M4).

    The incident id leaves every snapshot's members; the number digest leaves every snapshot
    unless `live_numbers` (digests still carried by remaining incidents or reports) holds it.
    An event whose snapshot (or merge target) no longer names any member is dropped: it was
    about data that no longer exists. Rewrites the file only if something changed; returns
    how many events were changed or dropped.
    """
    events = load_feedback(path)
    if not events:
        return 0
    drop_number = number_hash is not None and number_hash not in live_numbers

    def _clean(snapshot: OrgSnapshot) -> OrgSnapshot | None:
        members = tuple(m for m in snapshot.members if m != incident_id)
        numbers = tuple(n for n in snapshot.numbers if not (drop_number and n == number_hash))
        return OrgSnapshot(numbers=numbers, members=members) if members else None

    kept: list[FeedbackEvent] = []
    changed = 0
    for event in events:
        org = _clean(event.org)
        target = _clean(event.target) if event.target is not None else None
        if org is None or (event.target is not None and target is None):
            changed += 1
            continue
        cleaned = event.model_copy(update={"org": org, "target": target})
        changed += cleaned != event
        kept.append(cleaned)
    if changed:
        path.write_text("".join(e.model_dump_json() + "\n" for e in kept), encoding="utf-8")
    return changed


# --- application -----------------------------------------------------------------------------


def apply_feedback(organizations: Sequence[Organization], events: Sequence[FeedbackEvent]) -> list[Organization]:
    """Organizations with the latest feedback per operation applied; merges union the
    source into the target (the target keeps its id and priority). Confirm / dismiss are
    computed from the organization's *base* state, so a later confirm fully undoes an
    earlier dismiss rather than compounding with it."""
    base = list(organizations)  # what clustering produced (or a merge), before confirm/dismiss
    current = list(organizations)
    for event in sorted(events, key=lambda e: e.timestamp):
        index = _match(current, event.org)
        if index is None:
            continue
        if event.action == "merge":
            target_index = _match(current, event.target) if event.target is not None else None
            if target_index is None or target_index == index:
                continue
            base = _merge(base, source=index, target=target_index)
            current = list(base)
        elif event.action == "dismiss":
            org = base[index]
            current[index] = org.model_copy(
                update={"priority": org.priority * DISMISSED_PRIORITY_FACTOR, "is_novel": False, "feedback": "dismissed"}
            )
        else:
            current[index] = base[index].model_copy(update={"feedback": "confirmed"})
    return current


def _match(organizations: Sequence[Organization], snapshot: OrgSnapshot) -> int | None:
    """Index of the organization that shares a number with `snapshot`, else the one with
    the largest member overlap (>= half of the snapshot's members)."""
    numbers = set(snapshot.numbers)
    if numbers:
        for i, org in enumerate(organizations):
            if numbers & set(org.numbers):
                return i
    members = set(snapshot.members)
    best, best_overlap = None, 0.0
    for i, org in enumerate(organizations):
        overlap = len(members & set(org.members)) / len(members)
        if overlap > best_overlap:
            best, best_overlap = i, overlap
    return best if best_overlap >= _MEMBER_OVERLAP_MIN else None


def _merge(organizations: list[Organization], *, source: int, target: int) -> list[Organization]:
    src, dst = organizations[source], organizations[target]
    merged = dst.model_copy(
        update={
            "members": tuple(sorted({*dst.members, *src.members})),
            "numbers": tuple(sorted({*dst.numbers, *src.numbers})),
            "priority": max(dst.priority, src.priority),
            "is_novel": dst.is_novel and src.is_novel,
            "feedback": "merged",
        }
    )
    return [merged if i == target else org for i, org in enumerate(organizations) if i != source]
