"""Golden fixture for the JS cue matcher: every (hypothesis, cue) pair Python resolves.

    python scripts/export_cue_match_fixture.py   ->  tests_js/fixtures/cue_match.json

The JS port (`site/core/cue-match.js`) must return byte-identical offsets; `tests_js/
cue-match.test.mjs` asserts it. Cases come from the real recogniser capture (ADR D39), so
the fixture covers the boundary shifts and manglings the matcher exists for.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from qorgan.classifier.cue_match import MATCHER_VERSION, find_cue  # noqa: E402

CAPTURE = REPO / "data/asr_capture/pairs.jsonl"
OUT = REPO / "tests_js/fixtures/cue_match.json"
MAX_TEXTS = 80


def main() -> None:
    import yaml

    cues = yaml.safe_load((REPO / "data/lexicon/hard_signal_cues.yaml").read_text(encoding="utf-8"))["cues"]
    phrases = sorted({c for cue_list in cues.values() for c in cue_list})
    texts: list[str] = []
    for line in CAPTURE.read_text(encoding="utf-8").splitlines():
        if line.strip() and len(texts) < MAX_TEXTS:
            row = json.loads(line)
            texts.extend(t for t in (row["reference"], row["hypothesis"]) if t.strip())
    cases = [
        {"text": text, "cue": cue, "span": list(span) if (span := find_cue(text, cue)) else None}
        for text in texts
        for cue in phrases
    ]
    hits = sum(1 for c in cases if c["span"])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"matcher_version": MATCHER_VERSION, "cases": cases}, ensure_ascii=False), encoding="utf-8")
    print(f"{len(cases)} cases ({hits} hits) -> {OUT}")


if __name__ == "__main__":
    main()
