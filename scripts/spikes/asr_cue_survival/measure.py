"""Cue survival under the real recogniser: exact matching vs the bounded-edit matcher.

    python scripts/spikes/asr_cue_survival/measure.py [--pairs data/asr_capture/pairs.jsonl]

Three numbers decide whether the matcher ships:
  1. RECOVERY   — cue-bearing utterances whose tactic is still detected in the hypothesis.
  2. FALSE FIRE — cue-free utterances whose hypothesis now trips a cue that was never said.
  3. CLEAN DRIFT— corpus rows whose cue features change on clean text (must be ~0, or the
                  trained model's inputs move and the retrain carries the D27 KK-boundary risk).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))

from qorgan.classifier.cue_match import find_cue  # noqa: E402


def load_cues() -> dict[str, list[str]]:
    import yaml

    return yaml.safe_load((REPO / "data/lexicon/hard_signal_cues.yaml").read_text(encoding="utf-8"))["cues"]


def tactics_exact(text: str, cues: dict[str, list[str]]) -> set[str]:
    low = text.lower()
    return {t for t, cue_list in cues.items() if any(c.lower() in low for c in cue_list)}


def tactics_fuzzy(text: str, cues: dict[str, list[str]]) -> set[str]:
    return {t for t, cue_list in cues.items() if any(find_cue(text, c) for c in cue_list)}


def _rate(hit: int, total: int) -> str:
    return f"{hit}/{total} ({hit / total:.0%})" if total else "0/0"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, default=REPO / "data/asr_capture/pairs.jsonl")
    parser.add_argument("--show", type=int, default=12, help="how many recovered/false examples to print")
    args = parser.parse_args()
    cues = load_cues()
    rows = [json.loads(l) for l in args.pairs.read_text(encoding="utf-8").splitlines() if l.strip()]

    positives = [r for r in rows if r["cue_tactics_reference"]]
    negatives = [r for r in rows if not r["cue_tactics_reference"]]

    print(f"# Cue survival on {len(rows)} real recogniser outputs "
          f"({len(positives)} cue-bearing, {len(negatives)} cue-free)\n")

    print("## 1. Recovery — the tactic the reference carried, still found in the hypothesis\n")
    print("| set | n | exact | bounded-edit |")
    print("|---|---|---|---|")
    recovered_examples, by_language = [], {}
    for label, subset in [("all", positives)] + [(f"  {lang}", [r for r in positives if r["language"] == lang]) for lang in ("ru", "kk", "mixed")]:
        exact_hit = fuzzy_hit = 0
        for row in subset:
            truth = set(row["cue_tactics_reference"])
            got_exact = tactics_exact(row["hypothesis"], cues) & truth
            got_fuzzy = tactics_fuzzy(row["hypothesis"], cues) & truth
            exact_hit += bool(got_exact)
            fuzzy_hit += bool(got_fuzzy)
            if got_fuzzy and not got_exact:
                recovered_examples.append((row, sorted(got_fuzzy)))
        by_language[label] = (exact_hit, fuzzy_hit, len(subset))
        print(f"| {label} | {len(subset)} | {_rate(exact_hit, len(subset))} | {_rate(fuzzy_hit, len(subset))} |")

    print("\n## 2. False fire — a cue detected in a hypothesis whose reference had none\n")
    print("| set | n | exact | bounded-edit |")
    print("|---|---|---|---|")
    false_examples = []
    for label, subset in [("all", negatives)] + [(f"  {lang}", [r for r in negatives if r["language"] == lang]) for lang in ("ru", "kk", "mixed")]:
        exact_bad = fuzzy_bad = 0
        for row in subset:
            got_exact = tactics_exact(row["hypothesis"], cues)
            got_fuzzy = tactics_fuzzy(row["hypothesis"], cues)
            exact_bad += bool(got_exact)
            fuzzy_bad += bool(got_fuzzy)
            if got_fuzzy - got_exact:
                false_examples.append((row, sorted(got_fuzzy - got_exact)))
        print(f"| {label} | {len(subset)} | {_rate(exact_bad, len(subset))} | {_rate(fuzzy_bad, len(subset))} |")

    print("\n## 3. Clean drift — cue features on the corpus's own (clean) text\n")
    drift = []
    scanned = 0
    for split in ("train", "val", "test", "authored_heldout", "ood", "adversarial", "adversarial_legit", "shift"):
        path = REPO / "data/processed" / f"{split}.jsonl"
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            dialogue = json.loads(line)
            text = "\n".join(u["text"] for u in dialogue["utterances"])
            scanned += 1
            before, after = tactics_exact(text, cues), tactics_fuzzy(text, cues)
            if before != after:
                drift.append((split, dialogue["id"], sorted(after - before), sorted(before - after)))
    print(f"{len(drift)} of {scanned} dialogues change ({len(drift) / max(scanned, 1):.2%})")
    for split, dialogue_id, gained, lost in drift[:20]:
        print(f"  {split:18s} {dialogue_id:45s} +{gained} -{lost}")

    print(f"\n## Recovered examples (first {args.show})\n")
    for row, tactics in recovered_examples[: args.show]:
        print(f"- [{row['language']}] {tactics}\n    REF: {row['reference'][:105]}\n    HYP: {row['hypothesis'][:105]}")
    print(f"\n## New false fires (first {args.show})\n")
    for row, tactics in false_examples[: args.show]:
        print(f"- [{row['language']}] {tactics}\n    REF: {row['reference'][:105]}\n    HYP: {row['hypothesis'][:105]}")


if __name__ == "__main__":
    main()
