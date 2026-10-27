"""Pre-publish PII gate for the public dataset (ADR D45).

`scripts/hf_upload.py` publishes the processed splits and `data/augment/`. Some splits are
built by `build_corpus` (which scrubs), others are generated or repaired by separate tools
(`ood`, `adversarial*`, `shift`) -- one of them reached the Hub with an unscrubbed IIN. The
guard makes the invariant explicit: every published utterance must already be a
`scrub_text` fixed point, or nothing is uploaded.

`scrub_file` is the repair that establishes that invariant, so finding and fixing cannot
drift apart. It reuses `build_corpus.scrub_dialogue`, which also re-grounds trigger spans
and drops any span that *was* the PII -- a split must never ship a highlight pointing at
`[IIN]`.
"""

from __future__ import annotations

import fnmatch
import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from qorgan.data.schema import Dialogue
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

# What `scripts/hf_upload.py` uploads from data/processed -- one explicit allow list.
DATASET_ALLOW_PATTERNS: tuple[str, ...] = (*PUBLISHED_SPLITS, "manifest.json", "shift.manifest.json")

# Held-out splits that must NEVER reach the Hub (ADR D65). `shift_local` and its later batches
# `shift_local_2` (PLAN_2026-10 A3), `shift_local_3` (A7, the v2.2 judge), `shift_local_4` (D73,
# the v3 judge) `shift_local_5` (D74, the v3.1 judge) and `shift_local_6`
# (D75, the ensemble judge) are the independent held-outs written by a local model: once
# public, the next training generator or a crawler can see them, and they stop being independent.
# Kept in data/processed only for the harness.
NEVER_PUBLISHED: tuple[str, ...] = (
    "shift_local.jsonl", "shift_local.manifest.json", "shift_local_2.jsonl", "shift_local_2.manifest.json",
    "shift_local_3.jsonl", "shift_local_3.manifest.json",
    "shift_local_4.jsonl", "shift_local_4.manifest.json",
    "shift_local_5.jsonl", "shift_local_5.manifest.json",
    "shift_local_6.jsonl", "shift_local_6.manifest.json",
)


class PublishRefused(ValueError):
    """An upload pattern would publish a never-published held-out file."""


def assert_publishable(patterns: Sequence[str]) -> None:
    """Raise `PublishRefused` if any allow pattern matches a `NEVER_PUBLISHED` file."""
    blocked = sorted({name for pattern in patterns for name in NEVER_PUBLISHED if fnmatch.fnmatch(name, pattern)})
    if blocked:
        raise PublishRefused(f"refusing to publish never-published held-out file(s): {blocked} (ADR D65)")


# Real court-verdict quotes (ADR D68) are local-only: the Hub is a public, cross-border
# distribution channel (PD Law Arts. 7(3), 7(6), 16; docs/DATA_SOURCES.md SS4.4), and the legal
# basis for redistributing them is unsettled. No allow pattern may ever name them.
_NEVER_PUBLISH_MARKERS: tuple[str, ...] = ("court_quotes",)


def assert_not_publishing_court_quotes(names: Iterable[str]) -> None:
    """Exit (refuse the upload) if any path or allow pattern names the court_quotes split."""
    offending = [str(name) for name in names if any(marker in str(name) for marker in _NEVER_PUBLISH_MARKERS)]
    if offending:
        raise SystemExit(f"refusing to publish court_quotes (ADR D68, local-only): {offending}")


@dataclass(frozen=True)
class Finding:
    """Where PII was found -- deliberately without the offending text."""

    file: str
    dialogue_id: str
    utterance_index: int


def corpus_scrubber_version(manifest_path: Path) -> int:
    """The scrubber version the corpus at `manifest_path` was built with (absent = 1, ADR D66)."""
    if not manifest_path.exists():
        return 1
    return int(json.loads(manifest_path.read_text(encoding="utf-8")).get("scrubber_version", 1))


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


def scrub_file(path: Path) -> list[Finding]:
    """Rewrite `path` so every utterance is a `scrub_text` fixed point; return what was fixed.

    A missing file, or one that is already a fixed point, is left completely untouched (no
    rewrite, so a clean corpus never churns and the call is idempotent). Rows are re-serialised
    canonically, exactly as `build_corpus` writes a split.

    Raises `ValueError` if a line is not a valid `Dialogue`: a split we cannot parse is a split
    we must not silently publish. The message names the file and line, never their content.
    """
    from qorgan.data.build_corpus import scrub_dialogue  # local: keeps the gate's imports light

    findings = find_unscrubbed([path])
    if not findings:
        return []

    repaired: list[str] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            dialogue = Dialogue.model_validate(json.loads(line))
        except Exception as exc:  # noqa: BLE001 -- re-raised with position, without content
            raise ValueError(f"{path.name} line {number} is not a valid Dialogue") from exc
        repaired.append(scrub_dialogue(dialogue).model_dump_json())

    path.write_text("\n".join(repaired) + "\n", encoding="utf-8")
    return findings


def main(argv: list[str] | None = None) -> None:
    """Report, or with `--fix` repair, the published splits (ADR D45; default: the real ones)."""
    import argparse

    parser = argparse.ArgumentParser(description="PII gate for the published dataset (ADR D45).")
    parser.add_argument("paths", nargs="*", type=Path, help="defaults to data/processed + data/augment")
    parser.add_argument("--fix", action="store_true", help="rewrite offending rows via scrub_dialogue")
    args = parser.parse_args(argv)

    paths = args.paths or (
        [Path("data/processed") / name for name in PUBLISHED_SPLITS]
        + sorted(Path("data/augment").glob("*.jsonl"))
    )
    findings = find_unscrubbed(paths)
    for finding in findings:
        verb = "fixed" if args.fix else "is not scrubbed"
        print(f"{finding.file} {finding.dialogue_id} utterance {finding.utterance_index} {verb}")
    if args.fix:
        for path in paths:
            scrub_file(path)
        print(f"repaired {len(findings)} utterance(s); re-run the eval for any split that changed")
    elif findings:
        raise SystemExit(f"{len(findings)} unscrubbed utterance(s); re-run with --fix")
    else:
        print(f"{len(paths)} path(s) checked, all clean")


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
