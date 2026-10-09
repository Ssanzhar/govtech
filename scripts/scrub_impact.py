"""Measure what re-scrubbing the corpus with the current `scrub_text` would change (ADR D66).

For every published split (and `data/augment/`) it reports, per split:
- rows / utterances whose text changes, and which placeholders appear;
- hard-signal cue evidence: rows whose cue-tactic set changes (a feature flip), rows whose cue
  span text changes, and how many cue spans survive verbatim / in scrubbed form / not at all;
- the reassurance flag (the 6th interpretable feature): rows where it flips;
- gold trigger spans (`label.trigger_spans`): verbatim / in scrubbed form / dropped.

Held-out splits print counts only, never row text or ids (CLAUDE.md: never read held-out rows);
`train` / `val` / augment rows are listed by id with the placeholders that appeared.

Run: `python scripts/scrub_impact.py [--processed-dir DIR] [--ids]`.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

from qorgan.classifier.cue_lexicon import load_cue_lexicon
from qorgan.classifier.features import match_cues
from qorgan.classifier.reassurance import load_reassurance_patterns, reassures
from qorgan.data.publish_guard import PUBLISHED_SPLITS
from qorgan.data.schema import UTTERANCE_JOIN, Dialogue
from qorgan.data.scrub import PLACEHOLDERS, scrub_text
from qorgan.data.span_grounding import ground_flagged_phrases

_REPO = Path(__file__).resolve().parents[1]
_TRAINING_SPLITS = ("train.jsonl", "val.jsonl")


def _load(path: Path) -> list[Dialogue]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [Dialogue.model_validate_json(line) for line in lines if line.strip()]


def _survival(phrases: Sequence[str], before: str, after: str) -> Counter:
    tally: Counter = Counter()
    for phrase in phrases:
        if phrase in after:
            tally["verbatim"] += 1
        elif ground_flagged_phrases([phrase], after, before):
            tally["scrubbed_form"] += 1
        else:
            tally["dropped"] += 1
    return tally


def _placeholders_added(before: str, after: str) -> Counter:
    return Counter({p: after.count(p) - before.count(p) for p in PLACEHOLDERS if after.count(p) > before.count(p)})


def measure_split(dialogues: Sequence[Dialogue], lexicon, patterns) -> dict:
    stats: Counter = Counter(rows=len(dialogues))
    placeholders: Counter = Counter()
    cue_spans: Counter = Counter()
    gold_spans: Counter = Counter()
    changed: list[tuple[str, dict]] = []
    for dialogue in dialogues:
        before = dialogue.transcript()
        utterances_after = [scrub_text(u.text) for u in dialogue.utterances]
        after = UTTERANCE_JOIN.join(utterances_after)
        stats["utterances"] += len(utterances_after)
        stats["utterances_changed"] += sum(a != u.text for a, u in zip(utterances_after, dialogue.utterances))
        cues_before, cues_after = match_cues(before, lexicon), match_cues(after, lexicon)
        cue_spans += _survival([m.span.text for m in cues_before], before, after)
        gold_spans += _survival([s.text for s in dialogue.label.trigger_spans], before, after)
        if after == before:
            continue
        added = _placeholders_added(before, after)
        placeholders += added
        stats["rows_changed"] += 1
        stats["scam_rows_changed" if dialogue.label.risk >= 0.5 else "legit_rows_changed"] += 1
        if {m.tactic_id for m in cues_before} != {m.tactic_id for m in cues_after}:
            stats["rows_cue_tactics_changed"] += 1
        if [m.span.text for m in cues_before] != [m.span.text for m in cues_after]:
            stats["rows_cue_span_text_changed"] += 1
        if reassures(before, patterns) != reassures(after, patterns):
            stats["rows_reassurance_flipped"] += 1
        changed.append((dialogue.id, dict(added)))
    return {
        "stats": dict(stats),
        "placeholders_added": dict(placeholders),
        "cue_spans": dict(cue_spans),
        "gold_trigger_spans": dict(gold_spans),
        "changed": changed,
    }


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--processed-dir", type=Path, default=_REPO / "data" / "processed")
    parser.add_argument("--ids", action="store_true", help="list changed train/val/augment row ids")
    args = parser.parse_args(argv)
    lexicon, patterns = load_cue_lexicon(), load_reassurance_patterns()
    sources = [args.processed_dir / name for name in PUBLISHED_SPLITS]
    sources += sorted((_REPO / "data" / "augment").glob("*.jsonl"))
    for path in sources:
        if not path.exists():
            continue
        result = measure_split(_load(path), lexicon, patterns)
        listed = path.name in _TRAINING_SPLITS or path.parent.name == "augment"
        print(f"== {path.parent.name}/{path.name}")
        for key in ("stats", "placeholders_added", "cue_spans", "gold_trigger_spans"):
            print(f"   {key}: {json.dumps(result[key], ensure_ascii=False, sort_keys=True)}")
        if args.ids and listed:
            for dialogue_id, added in result["changed"]:
                print(f"     {dialogue_id}: {json.dumps(added, ensure_ascii=False, sort_keys=True)}")


if __name__ == "__main__":
    main()
