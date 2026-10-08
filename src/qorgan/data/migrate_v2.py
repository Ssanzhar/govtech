"""Deterministic v1 -> v2 tactic-label migration (ADR D63).

Taxonomy v2 replaces the three v1 `impersonation_*` ids with six `imp_*` ids. Which one a v1
row gets is decided by the Layer-2 role lexicon over its transcript, constrained to the old
id's declared successors (`tactics.yaml::superseded_by`, first = default):

* the transcript claims one or more roles whose `imp_tactic` is a successor -> those ids;
* no compatible role -> the default successor, and the row goes on the REVIEW list
  (reason `no_role` or `role_outside_family`, with the roles that were found).

Every other v1 id is kept as is (they are all in v2). The five new v2 tactics are never
inferred: a v1 row is UNLABELLED for them, not negative -- the phase-2 trainer masks old rows
out of those five heads (ADR D63, "label masking") instead of guessing labels.

    python -m qorgan.data.migrate_v2 --split train --split val   # counts + review list, no rewrite

Held-out splits are migrated by the same function in phase 2 with aggregate counts only;
their review rows are never opened (held-out hygiene, ADR D35/D43).
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from qorgan.data.schema import Dialogue, Label, TacticTag

REVIEW_NO_ROLE = "no_role"
REVIEW_OUTSIDE_FAMILY = "role_outside_family"
# Splits whose rows a developer may read for the review list. Everything else is counts-only.
REVIEWABLE_SPLITS = ("train", "val")


@dataclass(frozen=True)
class TagMapping:
    """The v2 ids one v1 id became, and why it may need a human look."""

    old_id: str
    new_ids: tuple[str, ...]
    review_reason: str | None = None


@dataclass(frozen=True)
class RowMigration:
    dialogue_id: str
    mappings: tuple[TagMapping, ...]
    roles_found: tuple[str, ...] = field(default=())

    @property
    def needs_review(self) -> bool:
        return any(m.review_reason for m in self.mappings)


def map_v1_tag(old_id: str, role_imps: Sequence[str | None]) -> TagMapping:
    """Map one tag id given the impersonation tactics of the roles the transcript claims
    (in claim order). Ids that are not retired map to themselves."""
    from qorgan.taxonomy import get_taxonomy

    taxonomy = get_taxonomy()
    try:
        successors = taxonomy.get(old_id).superseded_by
    except KeyError:
        return TagMapping(old_id, (old_id,))
    if not successors:
        return TagMapping(old_id, (old_id,))
    compatible = tuple(dict.fromkeys(imp for imp in role_imps if imp in successors))
    if compatible:
        return TagMapping(old_id, compatible)
    reason = REVIEW_OUTSIDE_FAMILY if any(role_imps) else REVIEW_NO_ROLE
    return TagMapping(old_id, (successors[0],), reason)


def migrate_tag_ids(tag_ids: Iterable[str], role_imps: Sequence[str | None]) -> tuple[str, ...]:
    """v1 (or mixed) tag ids -> v2 ids, de-duplicated, order kept."""
    out: list[str] = []
    for tag_id in tag_ids:
        out.extend(map_v1_tag(tag_id, role_imps).new_ids)
    return tuple(dict.fromkeys(out))


def migrate_dialogue(dialogue: Dialogue) -> tuple[Dialogue, RowMigration]:
    """A new `Dialogue` with v2 tags (weights carried over) and the record of what changed."""
    from qorgan.explain.roles import detect_roles, get_role_lexicon

    lexicon = get_role_lexicon()
    claims = detect_roles(dialogue.transcript(), lexicon)
    role_imps = [lexicon.get(c.role_id).imp_tactic for c in claims]
    mappings = tuple(map_v1_tag(t.id, role_imps) for t in dialogue.label.tactic_tags)
    weights: dict[str, float] = {}
    for tag, mapping in zip(dialogue.label.tactic_tags, mappings):
        for new_id in mapping.new_ids:
            weights[new_id] = max(weights.get(new_id, 0.0), tag.weight)
    label = Label(
        risk=dialogue.label.risk,
        tactic_tags=tuple(TacticTag(id=i, weight=w) for i, w in weights.items()),
        trigger_spans=dialogue.label.trigger_spans,
        is_hard_negative=dialogue.label.is_hard_negative,
    )
    migrated = Dialogue(id=dialogue.id, language=dialogue.language, utterances=dialogue.utterances, label=label)
    record = RowMigration(dialogue.id, tuple(m for m in mappings if m.new_ids != (m.old_id,)), tuple(c.role_id for c in claims))
    return migrated, record


def summarize(records: Sequence[RowMigration]) -> dict:
    """Aggregate counts only -- safe to print for any split."""
    transitions: Counter[str] = Counter()
    reasons: Counter[str] = Counter()
    for record in records:
        for m in record.mappings:
            transitions[f"{m.old_id} -> {'+'.join(m.new_ids)}"] += 1
            if m.review_reason:
                reasons[f"{m.old_id}: {m.review_reason}"] += 1
    return {
        "rows": len(records),
        "rows_with_impersonation": sum(1 for r in records if r.mappings),
        "rows_needing_review": sum(1 for r in records if r.needs_review),
        "transitions": dict(sorted(transitions.items())),
        "review_reasons": dict(sorted(reasons.items())),
    }


def review_list(records: Sequence[RowMigration]) -> list[dict]:
    """Ids + reasons + roles found -- never transcript text."""
    return [
        {
            "id": r.dialogue_id,
            "mapped": {m.old_id: list(m.new_ids) for m in r.mappings},
            "reasons": sorted({m.review_reason for m in r.mappings if m.review_reason}),
            "roles_found": list(r.roles_found),
        }
        for r in records
        if r.needs_review
    ]


def _load(path: Path) -> list[Dialogue]:
    return [Dialogue.model_validate_json(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main(argv: list[str] | None = None) -> None:  # pragma: no cover - CLI
    from qorgan.config import get_config

    parser = argparse.ArgumentParser(description="Dry-run the v1 -> v2 label migration (counts + review list).")
    parser.add_argument("--split", action="append", default=[], help="processed split name (repeatable)")
    parser.add_argument("--review-out", type=Path, default=None, help="write the review list (reviewable splits only)")
    args = parser.parse_args(argv)
    processed = get_config().data_dir / "processed"
    report: dict = {}
    review: dict = {}
    for split in args.split or list(REVIEWABLE_SPLITS):
        records = [migrate_dialogue(d)[1] for d in _load(processed / f"{split}.jsonl")]
        report[split] = summarize(records)
        if split in REVIEWABLE_SPLITS:
            review[split] = review_list(records)
    print(json.dumps(report, ensure_ascii=False, indent=1))
    if args.review_out is not None:
        args.review_out.parent.mkdir(parents=True, exist_ok=True)
        args.review_out.write_text(json.dumps(review, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        print(f"review list -> {args.review_out}")


if __name__ == "__main__":  # pragma: no cover
    main()
