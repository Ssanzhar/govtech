"""Generator-shift evaluation split (PLAN A12): a hand-authored set from a *second* generator.

`test` shares its generator (Gemini) and prompts with `train`, so its recall carries the
generator's house style. The rows under `data/authored/shift/` were written by a different
model with different prompting and no sight of the corpus or the lexicons, so scoring them
measures how much of the headline depends on one generator's style. Like the anchors, rows
are authored as plain turns + verbatim trigger phrases and grounded here; like the augment
files, the set is proven disjoint from every other split before anything is written.

    python -m qorgan.data.shift_set          # -> data/processed/shift.jsonl + shift.manifest.json
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path

from qorgan.data.build_corpus import content_hash, deduplicate, normalize_for_dedup, scrub_dialogue, split_counts
from qorgan.data.clean import clean_dialogue
from qorgan.data.schema import Dialogue, Label, TacticTag, Utterance, spans_from_phrases

SPLIT_NAME = "shift"
RAW_SUBDIR = Path("authored") / "shift"
RAW_ROW_KEYS = frozenset({"id", "language", "scenario", "utterances", "risk", "is_hard_negative", "tactic_tags", "trigger_phrases"})
# Every split the shift set must be disjoint from (train included: a leaked row would be memorised).
EVAL_SPLITS_CHECKED = ("train", "val", "test", "authored_heldout", "ood", "adversarial", "adversarial_legit")
GENERATOR_NOTE = "authored 2026-09-21 by Claude (Opus 5) from its own knowledge of KZ call patterns; no access to the corpus, prompts or lexicons"


def parse_raw_row(row: Mapping) -> Dialogue:
    """A raw authored row -> schema-valid `Dialogue`; every trigger phrase must be a verbatim
    substring (a phrase that is not is an authoring error, never a silently dropped span)."""
    unknown = set(row) ^ RAW_ROW_KEYS
    if unknown:
        raise ValueError(f"row {row.get('id')!r}: unexpected or missing keys {sorted(unknown)}")
    utterances = tuple(Utterance(speaker=u["speaker"], text=u["text"]) for u in row["utterances"])
    transcript = "\n".join(u.text for u in utterances)
    phrases = tuple(row["trigger_phrases"])
    spans = spans_from_phrases(phrases, transcript)
    if len(spans) != len(phrases):
        raise ValueError(f"row {row['id']!r}: {len(phrases) - len(spans)} trigger phrase(s) are not verbatim substrings of the transcript")
    label = Label(
        risk=float(row["risk"]),
        tactic_tags=tuple(TacticTag(id=t["id"], weight=float(t.get("weight", 1.0))) for t in row["tactic_tags"]),
        trigger_spans=spans,
        is_hard_negative=bool(row["is_hard_negative"]),
    )
    return Dialogue(id=row["id"], language=row["language"], utterances=utterances, label=label)


def load_raw_shift_rows(raw_dir: Path) -> tuple[Dialogue, ...]:
    """Every `*.jsonl` under `raw_dir`, files in name order, rows in file order."""
    rows: list[Dialogue] = []
    for path in sorted(raw_dir.glob("*.jsonl")):
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                rows.append(parse_raw_row(json.loads(line)))
            except (ValueError, KeyError, TypeError) as exc:
                raise ValueError(f"{path.name}:{number}: {exc}") from exc
    return tuple(rows)


def _scenarios(raw_dir: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for path in sorted(raw_dir.glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                out[row["id"]] = row.get("scenario", "")
    return out


def _overlaps(dialogues: Sequence[Dialogue], processed_dir: Path) -> dict[str, list[str]]:
    """`{split: [shift ids]}` for shift rows whose normalised transcript exists in that split."""
    own = {normalize_for_dedup(d.transcript()): d.id for d in dialogues}
    found: dict[str, list[str]] = {}
    for split in EVAL_SPLITS_CHECKED:
        path = processed_dir / f"{split}.jsonl"
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            key = normalize_for_dedup(Dialogue.model_validate_json(line).transcript())
            if key in own:
                found.setdefault(split, []).append(own[key])
    return found


def build_shift_split(*, raw_dir: Path, processed_dir: Path) -> dict:
    """Ground, clean, scrub and deduplicate the raw rows, refuse any overlap with an existing
    split, then write `processed_dir/shift.jsonl` + `shift.manifest.json`. Returns the manifest."""
    raw_rows = load_raw_shift_rows(raw_dir)
    cleaned = [(d.id, clean_dialogue(d)) for d in raw_rows]
    corrupted = [dialogue_id for dialogue_id, d in cleaned if d is None]
    if corrupted:  # hand-written data is fixed at the source, never silently dropped
        raise ValueError(f"corrupted shift dialogue(s): {corrupted}")
    dialogues = deduplicate(tuple(scrub_dialogue(d) for _, d in cleaned if d is not None))
    overlaps = _overlaps(dialogues, processed_dir)
    if overlaps:
        detail = "; ".join(f"overlaps {split}: {ids}" for split, ids in overlaps.items())
        raise ValueError(f"shift set is not disjoint -- {detail}")
    processed_dir.mkdir(parents=True, exist_ok=True)
    (processed_dir / f"{SPLIT_NAME}.jsonl").write_text(
        "".join(d.model_dump_json() + "\n" for d in dialogues), encoding="utf-8"
    )
    manifest = {
        "split": SPLIT_NAME,
        "generator": GENERATOR_NOTE,
        "source_files": [p.name for p in sorted(raw_dir.glob("*.jsonl"))],
        "counts": split_counts(dialogues),
        "content_hash": content_hash(dialogues),
        "disjoint_from": list(EVAL_SPLITS_CHECKED),
        "scenarios": _scenarios(raw_dir),
    }
    (processed_dir / f"{SPLIT_NAME}.manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest


def main() -> None:  # pragma: no cover - CLI
    from qorgan.config import get_config

    cfg = get_config()
    manifest = build_shift_split(raw_dir=cfg.data_dir / RAW_SUBDIR, processed_dir=cfg.data_dir / "processed")
    print(json.dumps(manifest["counts"], ensure_ascii=False))
    print(f"wrote {cfg.data_dir / 'processed' / (SPLIT_NAME + '.jsonl')}")


if __name__ == "__main__":  # pragma: no cover
    main()
