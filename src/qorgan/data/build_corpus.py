"""Assemble the shippable corpus (D2-4): read generated synthetic dialogues + the curated
`authored_heldout` anchors, PII-scrub them, deduplicate, deterministically split the synthetic
set into train/val/test, keep `authored_heldout` entirely separate, and write each split plus a
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

from qorgan.config import GEN_VAL_FRACTION, get_config
from qorgan.data.anchors import build_anchor_dialogues
from qorgan.data.asr_style import asr_style_augment
from qorgan.data.clean import clean_dialogue
from qorgan.data.generate import load_corpus_config
from qorgan.data.schema import Dialogue, Label, Utterance, spans_from_phrases
from qorgan.data.scrub import SCRUBBER_VERSION, scrub_text
from qorgan.data.span_grounding import ground_flagged_phrases

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

    A trigger span that carried PII is kept with the PII masked when that form is verbatim in
    the scrubbed text and still says something besides placeholders (ADR D66); a span that *was*
    the PII (e.g. a card number) is dropped rather than left dangling -- we never ship a
    highlighted span that no longer matches the text (the schema's verbatim invariant).
    """
    scrubbed_utterances = tuple(
        Utterance(speaker=u.speaker, text=scrub_text(u.text)) for u in dialogue.utterances
    )
    new_transcript = "\n".join(u.text for u in scrubbed_utterances)
    phrases = ground_flagged_phrases(
        [span.text for span in dialogue.label.trigger_spans], new_transcript, dialogue.transcript()
    )
    regrounded_spans = spans_from_phrases(phrases, new_transcript)
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


def split_counts(dialogues: Sequence[Dialogue]) -> dict:
    """Manifest counts for one split. `scam` / `legit` follow the labeled risk (the cut the
    evaluation uses); `hard_negatives` is the subset of legit calls built to look like scams.
    `positives` duplicates `scam` for readers of manifests written before 2026-09-25, when it
    meant "not a hard negative" and so counted plain legit calls as scams."""
    by_language: dict[str, int] = {}
    for dialogue in dialogues:
        by_language[dialogue.language] = by_language.get(dialogue.language, 0) + 1
    scam = sum(1 for d in dialogues if d.label.is_scam)
    return {
        "total": len(dialogues),
        "scam": scam,
        "legit": len(dialogues) - scam,
        "hard_negatives": sum(1 for d in dialogues if d.label.is_hard_negative),
        "positives": scam,
        "by_language": by_language,
    }


def build_manifest(
    splits: Mapping[str, Sequence[Dialogue]],
    authored_heldout: Sequence[Dialogue],
    *,
    seed: int,
    train_fraction: float,
    val_fraction: float,
) -> dict:
    """Build the reproducibility manifest: per-split counts, grand total, seed, fractions,
    and a content hash over every record (splits + authored_heldout)."""
    counts = {name: split_counts(splits.get(name, ())) for name in _SPLIT_NAMES}
    counts["authored_heldout"] = split_counts(authored_heldout)
    all_records = [d for name in _SPLIT_NAMES for d in splits.get(name, ())] + list(authored_heldout)
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
        # ADR D66: the scrubber that produced this text; an older one means "rebuild pending".
        "scrubber_version": SCRUBBER_VERSION,
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
    asr_style_fraction: float | None = None,
    gen_dialogues: Sequence[Dialogue] = (),
    label_space: str = "v1",
    review_decisions: Mapping[str, dict] | None = None,
    gen_writers: Mapping[str, str] | None = None,
    gen_label_spaces: Mapping[str, str] | None = None,
) -> dict:
    """Assemble, clean, split, and persist the corpus; return the manifest.

    Inputs default to the configured locations so the CLI is zero-arg, but every input is
    injectable for tests. Synthetic dialogues are repaired (`data.clean`, ADR D34; rows with
    unrecoverable artefacts are dropped and listed in the manifest), scrubbed, deduped, split; the curated
    anchors become `authored_heldout`, scrubbed but never mixed into train/val/test.
    `augment_dialogues` (targeted training data, e.g. reassurance hard negatives) are scrubbed
    and added to **train only** -- never val/test/authored_heldout, so the eval sets stay a clean
    held-out signal. `asr_style_fraction` of the final train rows also get an ASR-styled copy
    (`data.asr_style`, PLAN A10) so the heads see the recogniser's register; eval sets stay clean.

    Taxonomy v2 (ADR D63 phase 2): `gen_dialogues` (QC-accepted Claude rows, `data.gen.fold`)
    are repaired, scrubbed and split into train / val only -- never test -- before ASR styling.
    `label_space="v2"` migrates every v1-labelled row (`migrate_v2`, plus the human
    `review_decisions`) and writes `provenance.json`: the generator of every train/val row, which
    the trainer uses to mask v1 rows out of the five new heads and the reports split by.
    """
    cfg = get_config()
    active_seed = cfg.default_seed if seed is None else seed
    active_train = cfg.split_train_fraction if train_fraction is None else train_fraction
    active_val = cfg.split_val_fraction if val_fraction is None else val_fraction
    active_processed = processed_dir or (cfg.data_dir / "processed")
    active_asr_fraction = cfg.asr_style_train_fraction if asr_style_fraction is None else asr_style_fraction

    if dialogues is None:
        active_synthetic_path = synthetic_path or load_corpus_config().output_path
        dialogues = _read_jsonl(active_synthetic_path)
    anchors = build_anchor_dialogues() if anchor_dialogues is None else anchor_dialogues

    cleaned = [(d.id, clean_dialogue(d)) for d in dialogues]
    dropped_corrupted = [dialogue_id for dialogue_id, d in cleaned if d is None]
    scrubbed = [scrub_dialogue(d) for _, d in cleaned if d is not None]
    deduped = deduplicate(scrubbed)
    splits = split_dialogues(
        deduped, seed=active_seed, train_fraction=active_train, val_fraction=active_val
    )

    generated_ids: set[str] = set()
    if gen_dialogues:
        from qorgan.data.gen.fold import split_generated

        cleaned_gen = [(d.id, clean_dialogue(d)) for d in gen_dialogues]
        dropped_corrupted += [dialogue_id for dialogue_id, d in cleaned_gen if d is None]
        scrubbed_gen = [scrub_dialogue(d) for _, d in cleaned_gen if d is not None]
        gen_splits = split_generated(scrubbed_gen, seed=active_seed, val_fraction=GEN_VAL_FRACTION)
        generated_ids = {d.id for d in scrubbed_gen}
        splits = {**splits, "train": deduplicate(splits["train"] + gen_splits["train"]),
                  "val": deduplicate(splits["val"] + gen_splits["val"])}

    cleaned_augment = [(d.id, clean_dialogue(d)) for d in (augment_dialogues or ())]
    dropped_corrupted += [dialogue_id for dialogue_id, d in cleaned_augment if d is None]
    scrubbed_augment = tuple(scrub_dialogue(d) for _, d in cleaned_augment if d is not None)
    if scrubbed_augment:
        splits = {**splits, "train": deduplicate(splits["train"] + scrubbed_augment)}
    # Scrubbed AFTER styling (found while folding taxonomy-v2 rows, ADR D63 phase 2): styling
    # lower-cases a "[NAME]" placeholder into "name", which scrubber v2 (D66) masks again, so an
    # unscrubbed copy would not be the scrub fixed point every published row must be (D45).
    asr_styled = tuple(scrub_dialogue(d) for d in asr_style_augment(splits["train"], fraction=active_asr_fraction, seed=active_seed))
    if asr_styled:
        splits = {**splits, "train": deduplicate(splits["train"] + asr_styled)}
    cleaned_anchors = [(d.id, clean_dialogue(d)) for d in anchors]
    corrupted_anchors = [dialogue_id for dialogue_id, d in cleaned_anchors if d is None]
    if corrupted_anchors:  # hand-written data is fixed at the source, never silently dropped
        raise ValueError(f"corrupted anchor dialogue(s): {corrupted_anchors}")
    authored_heldout = tuple(scrub_dialogue(d) for _, d in cleaned_anchors)

    if label_space != "v1":
        splits, authored_heldout = _migrate_splits(splits, authored_heldout, label_space, review_decisions or {})

    for name in _SPLIT_NAMES:
        _write_jsonl(splits[name], active_processed / f"{name}.jsonl")
    _write_jsonl(authored_heldout, active_processed / "authored_heldout.jsonl")

    manifest = build_manifest(
        splits, authored_heldout, seed=active_seed, train_fraction=active_train, val_fraction=active_val
    )
    manifest["train_augment_count"] = len(scrubbed_augment)
    manifest["train_asr_style_count"] = len(asr_styled)
    manifest["dropped_corrupted"] = dropped_corrupted  # ADR D34: unrecoverable generation artefacts
    manifest["asr_style_fraction"] = active_asr_fraction
    manifest["adversarial_count"] = _copy_adversarial(cfg.data_dir / "adversarial", active_processed)
    if label_space != "v1" or generated_ids:
        provenance = build_provenance(splits, generated_ids, label_space, gen_writers, gen_label_spaces)
        (active_processed / PROVENANCE_FILE).write_text(json.dumps(provenance, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        manifest["label_space"] = label_space
        manifest["generator_balance"] = provenance["balance"]
    (active_processed / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


PROVENANCE_FILE = "provenance.json"
_ASR_SUFFIX = "-asr"


def build_provenance(
    splits: Mapping[str, Sequence[Dialogue]], generated_ids: set[str], label_space: str,
    writers: Mapping[str, str] | None = None, label_spaces: Mapping[str, str] | None = None,
) -> dict:
    """Generator per train/val row (an `-asr` copy inherits its source's) and P(scam | generator)
    per split -- the style-confound check (ADR D63 phase 2). A generated row's generator is its
    run's writer (`writers`, id -> writer; Claude when absent). `row_label_space` is the space each
    row was labelled in (legacy corpus v1; a generated row its run's, `label_spaces`, else v2), which
    the trainer masks by (ADR D73). Ids only, no text."""
    from qorgan.data.gen.fold import DEFAULT_RUN_LABEL_SPACE, GENERATOR_CLAUDE, GENERATOR_GEMINI

    writers = writers or {}
    label_spaces = label_spaces or {}

    generator: dict[str, str] = {}
    row_label_space: dict[str, str] = {}
    balance: dict[str, dict] = {}
    for name in ("train", "val"):
        counts: dict[str, dict[str, int]] = {}
        for d in splits.get(name, ()):
            base = d.id.removesuffix(_ASR_SUFFIX)
            source = writers.get(base, GENERATOR_CLAUDE) if base in generated_ids else GENERATOR_GEMINI
            generator[d.id] = source
            row_label_space[d.id] = label_spaces.get(base, DEFAULT_RUN_LABEL_SPACE) if base in generated_ids else "v1"
            bucket = counts.setdefault(source, {"rows": 0, "scam": 0})
            bucket["rows"] += 1
            bucket["scam"] += int(d.label.is_scam)
        balance[name] = {g: {**c, "p_scam": round(c["scam"] / c["rows"], 4)} for g, c in sorted(counts.items())}
    return {"label_space": label_space, "balance": balance, "generator": generator, "row_label_space": row_label_space}


def _migrate_splits(splits, authored_heldout, label_space: str, decisions: Mapping[str, dict]):
    """v1 tags -> `label_space` tags for every row (generated rows already carry v2 tags and
    pass through unchanged). Review decisions exist only for train/val ids (held-out review rows
    are never opened, ADR D35/D43)."""
    from qorgan.data.migrate_v2 import apply_review, migrate_dialogue

    if label_space not in ("v2", "v3"):  # v3 = v2 + five ids no v1 tag maps to (ADR D73)
        raise ValueError(f"no migration to label space {label_space!r}")

    def migrate(rows):
        return tuple(apply_review(migrate_dialogue(d)[0], dict(decisions)) for d in rows)

    return {name: migrate(rows) for name, rows in splits.items()}, migrate(authored_heldout)


def _copy_adversarial(source_dir: Path, processed_dir: Path) -> int:
    """The lexicon-free paraphrase split (PLAN A9) is eval-only: copied next to the other
    splits when present, never folded into train/val/test, not part of the content hash."""
    total = 0
    for source in sorted(source_dir.glob("adversarial*.jsonl")) if source_dir.is_dir() else ():
        dialogues = _read_jsonl(source)
        _write_jsonl(dialogues, processed_dir / source.name)
        total += len(dialogues)
    return total


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
    parser.add_argument("--processed-dir", type=Path, default=None, help="Output dir (default data/processed)")
    parser.add_argument("--gen-run", action="append", default=[], help="QC'd generation run to fold in (repeatable, ADR D63)")
    parser.add_argument("--label-space", default="v1", choices=("v1", "v2", "v3"), help="Label space of the written corpus")
    args = parser.parse_args(argv)

    corpus_cfg = load_corpus_config(args.config)
    augment_dir = args.augment_dir or (get_config().data_dir / "augment")
    gen_dialogues: list[Dialogue] = []
    gen_writers: dict[str, str] = {}
    gen_label_spaces: dict[str, str] = {}
    if args.gen_run:
        from qorgan.data.gen.__main__ import run_dir
        from qorgan.data.gen.fold import (
            REVIEW_REJECTS, RUN_WRITERS, accepted_dialogues, load_review_rejects, load_run_label_spaces,
            load_run_writers, run_label_space, run_writer,
        )

        rejected = load_review_rejects(get_config().repo_root / REVIEW_REJECTS)
        writers = load_run_writers(get_config().repo_root / RUN_WRITERS)
        spaces = load_run_label_spaces(get_config().repo_root / RUN_WRITERS)
        for run in args.gen_run:
            space = run_label_space(run, spaces)
            if space == "v3" and args.label_space != "v3":
                parser.error(f"run {run!r} is labelled in v3; build it with --label-space v3 (ADR D73)")
            run_rows = accepted_dialogues(run_dir(run), rejected=rejected)
            gen_dialogues += run_rows
            gen_writers.update({d.id: run_writer(run, writers) for d in run_rows})
            gen_label_spaces.update({d.id: space for d in run_rows})
    decisions = None
    if args.label_space in ("v2", "v3"):
        from qorgan.data.migrate_v2 import load_review_decisions

        decisions = load_review_decisions(get_config().repo_root / "data" / "taxonomy" / "migration_v2_review.json")
    manifest = build_corpus(
        synthetic_path=corpus_cfg.output_path,
        augment_dialogues=_read_augment_dir(augment_dir),
        seed=args.seed,
        processed_dir=args.processed_dir,
        gen_dialogues=gen_dialogues,
        label_space=args.label_space,
        review_decisions=decisions,
        gen_writers=gen_writers,
        gen_label_spaces=gen_label_spaces,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
