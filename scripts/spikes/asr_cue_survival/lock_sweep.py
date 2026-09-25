"""Replay language-locking policies over the captured dual-recogniser decisions (STT Tier B).

    python scripts/spikes/asr_cue_survival/lock_sweep.py [--dual data/asr_capture/dual.jsonl]

Microphone mode runs a Kazakh AND a Russian recogniser for the whole call (ADR D26). Two WASM
instances is the memory half of the phone gate (ADR D25). A locking policy switches the loser
off once the language is settled; the question is what that costs. Decode once, simulate many
-- the A6 meter-sweep pattern.

Policies compared against `both` (ship today):
  both            vote every utterance, both recognisers always running
  lock@K          after K utterances, keep only the language that won most of them
  lock@K+conf<T   as lock@K, but reopen both for one utterance whenever the locked
                  recogniser's confidence drops below T (a language switch looks like this)
  lock@K+every N  as lock@K, with both reopened every Nth utterance regardless

Reported per policy: utterances whose chosen TEXT differs from `both`, cue detections lost or
gained versus `both`, and the share of utterance-decodes saved (the compute/memory win).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))

from qorgan.classifier.cue_match import find_cue  # noqa: E402

TIE_EPSILON = 0.02          # asr.js: confidences this close are a tie
KAZAKH_SHARE = 0.08         # asr.js: a tie goes to Kazakh if its share of Kazakh letters clears this
KAZAKH_LETTERS = set("әғқңөұүһі")


def load_cues() -> dict[str, list[str]]:
    import yaml

    return yaml.safe_load((REPO / "data/lexicon/hard_signal_cues.yaml").read_text(encoding="utf-8"))["cues"]


def kazakh_share(text: str) -> float:
    letters = [c for c in text.lower() if c.isalpha()]
    if not letters:
        return 0.0
    return sum(1 for c in letters if c in KAZAKH_LETTERS) / len(letters)


def vote(row: dict) -> tuple[str, str]:
    """The shipped rule: higher mean word confidence wins; a near-tie goes to Kazakh when the
    Kazakh hypothesis carries enough Kazakh letters (ADR D26)."""
    kk, ru = row["kk"], row["ru"]
    if not kk["text"] and not ru["text"]:
        return "", ""
    if not kk["text"]:
        return "ru", ru["text"]
    if not ru["text"]:
        return "kk", kk["text"]
    if abs(kk["confidence"] - ru["confidence"]) <= TIE_EPSILON:
        return ("kk", kk["text"]) if kazakh_share(kk["text"]) >= KAZAKH_SHARE else ("ru", ru["text"])
    return ("kk", kk["text"]) if kk["confidence"] > ru["confidence"] else ("ru", ru["text"])


def simulate(rows: list[dict], *, lock_after: int | None, conf_floor: float | None, recheck_every: int | None):
    """Return per-utterance (language, text, decodes_used) for one dialogue under a policy."""
    out = []
    locked: str | None = None
    winners: Counter[str] = Counter()
    for position, row in enumerate(rows):
        reopen = locked is not None and recheck_every and position % recheck_every == 0
        if locked is None or reopen:
            language, text = vote(row)
            decodes = 2
            if language:
                winners[language] += 1
        else:
            language, text = locked, row[locked]["text"]
            decodes = 1
            if conf_floor is not None and row[locked]["confidence"] < conf_floor:
                language, text = vote(row)   # a drop looks like a language switch: look again
                decodes = 2
                if language:
                    winners[language] += 1
        if locked is None and lock_after is not None and position + 1 >= lock_after and winners:
            locked = winners.most_common(1)[0][0]
        out.append((language, text, decodes))
    return out


def cue_tactics(text: str, cues: dict[str, list[str]]) -> set[str]:
    return {t for t, cue_list in cues.items() if any(find_cue(text, c) for c in cue_list)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dual", type=Path, default=REPO / "data/asr_capture/dual.jsonl")
    args = parser.parse_args()
    cues = load_cues()

    by_dialogue: dict[str, list[dict]] = defaultdict(list)
    meta: dict[str, dict] = {}
    for line in args.dual.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            by_dialogue[row["dialogue_id"]].append(row)
            meta[row["dialogue_id"]] = row
    for rows in by_dialogue.values():
        rows.sort(key=lambda r: r["index"])

    policies = [
        ("both (ships today)", dict(lock_after=None, conf_floor=None, recheck_every=None)),
        ("lock@2", dict(lock_after=2, conf_floor=None, recheck_every=None)),
        ("lock@3", dict(lock_after=3, conf_floor=None, recheck_every=None)),
        ("lock@4", dict(lock_after=4, conf_floor=None, recheck_every=None)),
        ("lock@3 + conf<0.80", dict(lock_after=3, conf_floor=0.80, recheck_every=None)),
        ("lock@3 + conf<0.85", dict(lock_after=3, conf_floor=0.85, recheck_every=None)),
        ("lock@3 + every 5th", dict(lock_after=3, conf_floor=None, recheck_every=5)),
        ("lock@3 + conf<0.85 + every 5th", dict(lock_after=3, conf_floor=0.85, recheck_every=5)),
    ]

    languages = sorted({m["language"] for m in meta.values()})
    baseline = {d: simulate(rows, lock_after=None, conf_floor=None, recheck_every=None) for d, rows in by_dialogue.items()}
    total_utterances = sum(len(r) for r in by_dialogue.values())
    print(f"# Language locking over {len(by_dialogue)} dialogues / {total_utterances} utterances\n")
    print("| policy | decodes saved | text differs | cues lost | cues gained | " + " | ".join(f"differs ({lang})" for lang in languages) + " |")
    print("|---|---|---|---|---|" + "---|" * len(languages))

    for label, kwargs in policies:
        differs = lost = gained = 0
        decodes = 0
        per_language: Counter[str] = Counter()
        for dialogue_id, rows in by_dialogue.items():
            simulated = simulate(rows, **kwargs)
            for (_, text, used), (_, base_text, _) in zip(simulated, baseline[dialogue_id]):
                decodes += used
                if text != base_text:
                    differs += 1
                    per_language[meta[dialogue_id]["language"]] += 1
            got = cue_tactics("\n".join(t for _, t, _ in simulated), cues)
            base = cue_tactics("\n".join(t for _, t, _ in baseline[dialogue_id]), cues)
            lost += len(base - got)
            gained += len(got - base)
        saved = 1 - decodes / (2 * total_utterances)
        cells = " | ".join(str(per_language.get(lang, 0)) for lang in languages)
        print(f"| {label} | {saved:.0%} | {differs} ({differs / total_utterances:.1%}) | {lost} | {gained} | {cells} |")


if __name__ == "__main__":
    main()
