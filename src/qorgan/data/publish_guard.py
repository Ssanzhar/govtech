"""Pre-publish PII gate for the public dataset (ADR D45).

`scripts/hf_upload.py` publishes the processed splits and `data/augment/`. Some splits are
built by `build_corpus` (which scrubs), others are generated or repaired by separate tools
(`ood`, `adversarial*`, `shift`) -- one of them reached the Hub with an unscrubbed IIN. The
guard makes the invariant explicit: every published utterance must already be a
`scrub_text` fixed point, or nothing is uploaded.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from qorgan.data.scrub import scrub_text

PUBLISHED_SPLITS: tuple[str, ...] = (
    "train.jsonl",
    "val.jsonl",
    "test.jsonl",
    "authored_heldout.jsonl",
    "ood.jsonl",
    "adversarial.jsonl",
    "adversarial_legit.jsonl",
    "shift.jsonl",
)


@dataclass(frozen=True)
class Finding:
    """Where PII was found -- deliberately without the offending text."""

    file: str
    dialogue_id: str
    utterance_index: int


def find_unscrubbed(paths: Iterable[Path]) -> list[Finding]:
    findings: list[Finding] = []
    for path in paths:
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            dialogue = json.loads(line)
            for index, utterance in enumerate(dialogue.get("utterances", [])):
                text = utterance.get("text", "")
                if scrub_text(text) != text:
                    findings.append(Finding(path.name, str(dialogue.get("id")), index))
    return findings
