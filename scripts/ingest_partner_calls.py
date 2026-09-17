"""Ingest one partner batch of real calls (PLAN_2026-09 A8; protocol: docs/DATA_INTAKE.md).

    python scripts/ingest_partner_calls.py <batch_dir> [--dry-run] [--transcribe]
        [--real-dir data/real] [--processed-dir data/processed]
        [--lock-negatives 60] [--lock-positives 40]

1. validates `batch.yaml` + `calls.csv` (labels, tactic ids, files, labeler declaration);
2. scrubs every utterance, hashes caller numbers into a separate linkage file (needs
   `QORGAN_NUMBER_HMAC_KEY`; refuses numbered rows without it);
3. writes the batch under `<real_dir>/batches/`, allocates new calls first-come into the
   locked `real_heldout_v2` (until the targets are met) or `real_train`, and materialises
   both splits under `<processed_dir>/` with a hash-locked `<real_dir>/manifest.json`.

Prints ids and counts only -- never call content. `--dry-run` stops after step 2.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from qorgan.config import get_config
from qorgan.data.real_allocation import (
    ALLOCATION_FILENAME,
    Allocation,
    LockTargets,
    LockViolation,
    allocate,
    load_allocation,
    materialize,
    save_allocation,
)
from qorgan.data.real_intake import BatchValidationError, ingest_batch, write_batch

REPO_ROOT = Path(__file__).resolve().parents[1]


def _whisper_transcriber():
    from qorgan.asr.transcribe import transcribe

    return lambda path: transcribe(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Ingest a partner batch of real calls.")
    parser.add_argument("batch_dir", type=Path)
    parser.add_argument("--real-dir", type=Path, default=REPO_ROOT / "data" / "real")
    parser.add_argument("--processed-dir", type=Path, default=REPO_ROOT / "data" / "processed")
    parser.add_argument("--dry-run", action="store_true", help="validate + convert in memory; write nothing")
    parser.add_argument("--transcribe", action="store_true", help="transcribe audio-only rows with faster-whisper")
    parser.add_argument("--lock-negatives", type=int, default=None, help="locked-set target (first allocation only)")
    parser.add_argument("--lock-positives", type=int, default=None, help="locked-set target (first allocation only)")
    args = parser.parse_args(argv)

    cfg = get_config()
    try:
        ingested = ingest_batch(
            args.batch_dir, hmac_key=cfg.number_hmac_key, transcriber=_whisper_transcriber() if args.transcribe else None
        )
    except BatchValidationError as exc:
        print(f"[ingest] REFUSED: {exc}", file=sys.stderr)
        return 2
    manifest = ingested.manifest
    print(f"[ingest] batch {manifest.batch_id} from {manifest.partner_id} ({manifest.delivered_on}, {manifest.consent_basis})")
    print(f"[ingest] {len(ingested.dialogues)} calls: {ingested.counts}; {len(ingested.linkages)} with a hashed number")
    if args.dry_run:
        print("[ingest] dry run -- nothing written")
        return 0

    write_batch(ingested, args.real_dir)
    allocation_path = args.real_dir / ALLOCATION_FILENAME
    current = load_allocation(allocation_path)
    if not allocation_path.exists() and (args.lock_negatives or args.lock_positives):
        current = Allocation(targets=LockTargets(
            negatives=args.lock_negatives or current.targets.negatives,
            positives=args.lock_positives or current.targets.positives,
        ))
    updated = allocate(current, ingested.dialogues)
    save_allocation(updated, allocation_path)
    try:
        real_manifest = materialize(args.real_dir, args.processed_dir)
    except LockViolation as exc:
        print(f"[ingest] LOCK VIOLATION: {exc}", file=sys.stderr)
        return 3
    state = "LOCKED" if real_manifest.locked else "filling"
    print(f"[ingest] real_heldout_v2 {state}: {real_manifest.counts['real_heldout_v2']} "
          f"(targets {updated.targets.negatives} neg / {updated.targets.positives} pos); "
          f"real_train: {real_manifest.counts['real_train']}")
    if real_manifest.locked:
        print(f"[ingest] locked sha256 {real_manifest.locked_sha256}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
