"""Assemble the shippable corpus (D2-4): read generated synthetic dialogues + the curated
`real_heldout` anchors, PII-scrub them, deduplicate, deterministically split the synthetic
set into train/val/test, keep `real_heldout` entirely separate, and write each split plus a
reproducibility `manifest.json` (counts + content hash) to `data/processed/`.

Determinism is the whole point (`docs/DECISIONS.md` D9 -- seeded build + manifest instead
of DVC): given the same inputs, seed, and fractions, the partition and the manifest hash are
byte-stable, so judges can reproduce the exact corpus. Every function here is pure except
`build_corpus`, which does the file I/O at the edges.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path

from qorgan.config import get_config
from qorgan.data.anchors import build_anchor_dialogues
from qorgan.data.generate import load_corpus_config
from qorgan.data.schema import Dialogue, Label, Utterance, spans_from_phrases
from qorgan.data.scrub import scrub_text

_SPLIT_NAMES = ("train", "val", "test")
# Resolution of the deterministic hash-to-fraction mapping used by `assign_split`.
_HASH_BUCKETS = 1_000_000
_WHITESPACE = re.compile(r"\s+")
_MANIFEST_SCHEMA_NOTE = "records validated against src/qorgan/data/schema.py:Dialogue"


def normalize_for_dedup(text: str) -> str:
    """Canonical form for duplicate detection: lowercased, whitespace-collapsed, stripped."""
    return _WHITESPACE.sub(" ", text).strip().lower()


def deduplicate(dialogues: Sequence[Dialogue]) -> tuple[Dialogue, ...]:
    """Drop dialogues whose normalized transcript was already seen; keep first, stable order."""
    seen: set[str] = set()
    kept: list[Dialogue] = []
    for dialogue in dialogues:
        key = normalize_for_dedup(dialogue.transcript())
        if key in seen:
            continue
        seen.add(key)
        kept.append(dialogue)
    return tuple(kept)


def scrub_dialogue(dialogue: Dialogue) -> Dialogue:
    """Return a new `Dialogue` with PII scrubbed from every utterance and trigger spans
    re-grounded against the scrubbed transcript.

    A trigger span whose text was itself altered by scrubbing (e.g. it *was* a card number)
    is dropped rather than left dangling -- we never ship a highlighted span that no longer
    matches the text (the schema's verbatim invariant, CLAUDE.md SS6).
    """
    scrubbed_utterances = tuple(
        Utterance(speaker=u.speaker, text=scrub_text(u.text)) for u in dialogue.utterances
    )
    new_transcript = "\n".join(u.text for u in scrubbed_utterances)
    regrounded_spans = spans_from_phrases(
        [span.text for span in dialogue.label.trigger_spans], new_transcript
    )
    new_label = Label(
        risk=dialogue.label.risk,
        tactic_tags=dialogue.label.tactic_tags,
        trigger_spans=regrounded_spans,
        is_hard_negative=dialogue.label.is_hard_negative,
    )
    return Dialogue(
        id=dialogue.id,
        language=dialogue.language,
        utterances=scrubbed_utterances,
        label=new_label,
    )


def assign_split(dialogue_id: str, *, seed: int, train_fraction: float, val_fraction: float) -> str:
    """Deterministically map a dialogue id to `train`/`val`/`test` via a seeded hash.

    Hashing the id (not a running counter) means the split is stable under reordering,
    additions, and reruns -- only the id, seed, and fractions matter.
    """
    digest = hashlib.sha256(f"{seed}:{dialogue_id}".encode("utf-8")).hexdigest()
    ratio = (int(digest, 16) % _HASH_BUCKETS) / _HASH_BUCKETS
    if ratio < train_fraction:
        return "train"
    if ratio < train_fraction + val_fraction:
        return "val"
    return "test"


def split_dialogues(
    dialogues: Sequence[Dialogue], *, seed: int, train_fraction: float, val_fraction: float
) -> dict[str, tuple[Dialogue, ...]]:
    """Partition `dialogues` into train/val/test by `assign_split`, preserving input order."""
    buckets: dict[str, list[Dialogue]] = {name: [] for name in _SPLIT_NAMES}
    for dialogue in dialogues:
        bucket = assign_split(
            dialogue.id, seed=seed, train_fraction=train_fraction, val_fraction=val_fraction
        )
        buckets[bucket].append(dialogue)
    return {name: tuple(buckets[name]) for name in _SPLIT_NAMES}


def content_hash(dialogues: Sequence[Dialogue]) -> str:
    """Order-independent SHA-256 over the canonical JSON of `dialogues` (sorted by id)."""
    canonical = [d.model_dump(mode="json") for d in sorted(dialogues, key=lambda d: d.id)]
    blob = json.dumps(canonical, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _split_counts(dialogues: Sequence[Dialogue]) -> dict:
    by_language: dict[str, int] = {}
    hard_negatives = 0
    for dialogue in dialogues:
        by_language[dialogue.language] = by_language.get(dialogue.language, 0) + 1
        if dialogue.label.is_hard_negative:
            hard_negatives += 1
    return {
        "total": len(dialogues),
        "hard_negatives": hard_negatives,
        "positives": len(dialogues) - hard_negatives,
        "by_language": by_language,
    }


def build_manifest(
    splits: Mapping[str, Sequence[Dialogue]],
    real_heldout: Sequence[Dialogue],
    *,
    seed: int,
    train_fraction: float,
    val_fraction: float,
) -> dict:
    """Build the reproducibility manifest: per-split counts, grand total, seed, fractions,
    and a content hash over every record (splits + real_heldout)."""
    counts = {name: _split_counts(splits.get(name, ())) for name in _SPLIT_NAMES}
    counts["real_heldout"] = _split_counts(real_heldout)
    all_records = [d for name in _SPLIT_NAMES for d in splits.get(name, ())] + list(real_heldout)
    return {
        "schema": _MANIFEST_SCHEMA_NOTE,
        "seed": seed,
        "fractions": {
            "train": train_fraction,
            "val": val_fraction,
            "test": 1.0 - train_fraction - val_fraction,
        },
        "counts": counts,
        "total": len(all_records),
        "content_hash": content_hash(all_records),
    }


def _write_jsonl(dialogues: Sequence[Dialogue], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = (d.model_dump_json() for d in dialogues)
    path.write_text("\n".join(lines) + ("\n" if dialogues else ""), encoding="utf-8")


def _read_jsonl(path: Path) -> list[Dialogue]:
    if not path.exists():
        raise FileNotFoundError(
            f"Synthetic corpus not found: {path}. Run `qorgan.data.generate` first "
            "(or pass dialogues= explicitly)."
        )
    dialogues: list[Dialogue] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            dialogues.append(Dialogue.model_validate_json(line))
    return dialogues


def build_corpus(
    *,
    dialogues: Sequence[Dialogue] | None = None,
    anchor_dialogues: Sequence[Dialogue] | None = None,
    augment_dialogues: Sequence[Dialogue] | None = None,
    synthetic_path: Path | None = None,
    processed_dir: Path | None = None,
    seed: int | None = None,
    train_fraction: float | None = None,
    val_fraction: float | None = None,
) -> dict:
    """Assemble, clean, split, and persist the corpus; return the manifest.

    Inputs default to the configured locations so the CLI is zero-arg, but every input is
    injectable for tests. Synthetic dialogues are scrubbed + deduped + split; the curated
    anchors become `real_heldout`, scrubbed but never mixed into train/val/test.
    `augment_dialogues` (targeted training data, e.g. reassurance hard negatives) are scrubbed
    and added to **train only** -- never val/test/real_heldout, so the eval sets stay a clean
    held-out signal.
    """
    cfg = get_config()
    active_seed = cfg.default_seed if seed is None else seed
    active_train = cfg.split_train_fraction if train_fraction is None else train_fraction
    active_val = cfg.split_val_fraction if val_fraction is None else val_fraction
    active_processed = processed_dir or (cfg.data_dir / "processed")

    if dialogues is None:
        active_synthetic_path = synthetic_path or load_corpus_config().output_path
        dialogues = _read_jsonl(active_synthetic_path)
    anchors = build_anchor_dialogues() if anchor_dialogues is None else anchor_dialogues

    scrubbed = [scrub_dialogue(d) for d in dialogues]
    deduped = deduplicate(scrubbed)
    splits = split_dialogues(
        deduped, seed=active_seed, train_fraction=active_train, val_fraction=active_val
    )

    scrubbed_augment = tuple(scrub_dialogue(d) for d in (augment_dialogues or ()))
    if scrubbed_augment:
        splits = {**splits, "train": deduplicate(splits["train"] + scrubbed_augment)}
    real_heldout = tuple(scrub_dialogue(d) for d in anchors)

    for name in _SPLIT_NAMES:
        _write_jsonl(splits[name], active_processed / f"{name}.jsonl")
    _write_jsonl(real_heldout, active_processed / "real_heldout.jsonl")

    manifest = build_manifest(
        splits, real_heldout, seed=active_seed, train_fraction=active_train, val_fraction=active_val
    )
    manifest["train_augment_count"] = len(scrubbed_augment)
    (active_processed / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


def _read_augment_dir(augment_dir: Path) -> list[Dialogue]:
    """Load all `*.jsonl` in `augment_dir` as train-augmentation dialogues (empty if absent).

    These are committed, targeted training examples (e.g. reassurance hard negatives from
    `scripts/augment_reassurance_negatives.py`) that close a specific model gap.
    """
    if not augment_dir.exists():
        return []
    dialogues: list[Dialogue] = []
    for path in sorted(augment_dir.glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                dialogues.append(Dialogue.model_validate_json(line))
    return dialogues


def main(argv: Sequence[str] | None = None) -> None:
    """CLI: `python -m qorgan.data.build_corpus [--config PATH] [--seed N]`."""
    parser = argparse.ArgumentParser(description="Assemble the Qorgan corpus + manifest.")
    parser.add_argument("--config", type=Path, default=None, help="Path to configs/corpus.yaml")
    parser.add_argument("--seed", type=int, default=None, help="Override the split seed")
    parser.add_argument(
        "--augment-dir", type=Path, default=None, help="Dir of *.jsonl train-augmentation dialogues"
    )
    args = parser.parse_args(argv)

    corpus_cfg = load_corpus_config(args.config)
    augment_dir = args.augment_dir or (get_config().data_dir / "augment")
    manifest = build_corpus(
        synthetic_path=corpus_cfg.output_path,
        augment_dialogues=_read_augment_dir(augment_dir),
        seed=args.seed,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
