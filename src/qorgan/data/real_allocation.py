"""Real-call allocation + lock (PLAN_2026-09 A3/A8): the first N negatives / M positives
ever ingested form `real_heldout_v2`, the locked test set; everything after goes to
`real_train`. Once the targets are met the locked file's sha256 is written to
`<real_dir>/manifest.json` and any later change to it is a `LockViolation`.

All functions are pure over the paths they are given; the allocation ledger is the only
state and it is append-only by construction (`allocate` never moves an id).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from qorgan.data.real_intake import BATCHES_SUBDIR
from qorgan.data.schema import Dialogue

LOCKED_SPLIT = "real_heldout_v2"
TRAIN_SPLIT = "real_train"
ALLOCATION_FILENAME = "allocation.json"
MANIFEST_FILENAME = "manifest.json"
# PLAN A3: >= 60 negatives / >= 40 positives before a 0-FP result bounds FPR <= 5 %.
DEFAULT_LOCK_NEGATIVES = 60
DEFAULT_LOCK_POSITIVES = 40


class LockViolation(RuntimeError):
    """The locked held-out set would change (or has changed) after being locked."""


class LockTargets(BaseModel):
    model_config = ConfigDict(frozen=True)

    negatives: int = Field(default=DEFAULT_LOCK_NEGATIVES, gt=0)
    positives: int = Field(default=DEFAULT_LOCK_POSITIVES, gt=0)


class Allocation(BaseModel):
    """Which dialogue ids sit in the locked set and which in `real_train`."""

    model_config = ConfigDict(frozen=True)

    targets: LockTargets = LockTargets()
    locked_negatives: tuple[str, ...] = ()
    locked_positives: tuple[str, ...] = ()
    real_train: tuple[str, ...] = ()

    @property
    def is_locked(self) -> bool:
        return (
            len(self.locked_negatives) >= self.targets.negatives
            and len(self.locked_positives) >= self.targets.positives
        )

    @property
    def allocated(self) -> frozenset[str]:
        return frozenset((*self.locked_negatives, *self.locked_positives, *self.real_train))


class RealManifest(BaseModel):
    """`<real_dir>/manifest.json`: what was materialised and the lock hash."""

    model_config = ConfigDict(frozen=True)

    generated_at: datetime
    batches: tuple[str, ...]
    counts: dict[str, dict[str, int]]
    locked: bool
    locked_sha256: str | None
    targets: LockTargets


# --- allocation ------------------------------------------------------------------------------


def allocate(allocation: Allocation, dialogues: Sequence[Dialogue]) -> Allocation:
    """First-come allocation of not-yet-allocated dialogues; returns a new `Allocation`."""
    negatives, positives, train = list(allocation.locked_negatives), list(allocation.locked_positives), list(allocation.real_train)
    seen = set(allocation.allocated)
    for dialogue in dialogues:
        if dialogue.id in seen:
            continue
        seen.add(dialogue.id)
        if dialogue.label.is_hard_negative and len(negatives) < allocation.targets.negatives:
            negatives.append(dialogue.id)
        elif not dialogue.label.is_hard_negative and len(positives) < allocation.targets.positives:
            positives.append(dialogue.id)
        else:
            train.append(dialogue.id)
    return allocation.model_copy(
        update={"locked_negatives": tuple(negatives), "locked_positives": tuple(positives), "real_train": tuple(train)}
    )


def load_allocation(path: Path) -> Allocation:
    if not path.exists():
        return Allocation()
    return Allocation.model_validate_json(path.read_text(encoding="utf-8"))


def save_allocation(allocation: Allocation, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(allocation.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return path


# --- materialisation + lock ------------------------------------------------------------------


def materialize(real_dir: Path, processed_dir: Path, *, now: datetime | None = None) -> RealManifest:
    """Write `real_heldout_v2.jsonl` + `real_train.jsonl` from the batches and the
    allocation; refuse to change a locked file. Idempotent."""
    allocation = load_allocation(real_dir / ALLOCATION_FILENAME)
    dialogues = _load_batches(real_dir)
    by_id = {d.id: d for d in dialogues}
    locked_ids = (*allocation.locked_negatives, *allocation.locked_positives)
    missing = [i for i in (*locked_ids, *allocation.real_train) if i not in by_id]
    if missing:
        raise LockViolation(f"allocation names dialogues that no batch contains: {missing[:5]}")

    locked_blob = _jsonl([by_id[i] for i in locked_ids])
    locked_hash = _sha256_text(locked_blob)
    previous = _load_manifest(real_dir / MANIFEST_FILENAME)
    if previous is not None and previous.locked and previous.locked_sha256 != locked_hash:
        raise LockViolation(
            f"{LOCKED_SPLIT} is locked (sha256 {previous.locked_sha256[:12]}...) and the allocation/batches would change it"
        )

    processed_dir.mkdir(parents=True, exist_ok=True)
    (processed_dir / f"{LOCKED_SPLIT}.jsonl").write_text(locked_blob, encoding="utf-8")
    (processed_dir / f"{TRAIN_SPLIT}.jsonl").write_text(_jsonl([by_id[i] for i in allocation.real_train]), encoding="utf-8")
    manifest = RealManifest(
        generated_at=previous.generated_at if previous is not None and previous.locked else (now or datetime.now(UTC)),
        batches=tuple(sorted(p.stem for p in (real_dir / BATCHES_SUBDIR).glob("*.jsonl") if not p.name.endswith(".linkage.jsonl"))),
        counts={
            LOCKED_SPLIT: {"positives": len(allocation.locked_positives), "negatives": len(allocation.locked_negatives)},
            TRAIN_SPLIT: _pos_neg([by_id[i] for i in allocation.real_train]),
        },
        locked=allocation.is_locked,
        locked_sha256=locked_hash if allocation.is_locked else None,
        targets=allocation.targets,
    )
    (real_dir / MANIFEST_FILENAME).write_text(manifest.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return manifest


def verify_lock(manifest_path: Path, processed_dir: Path) -> RealManifest:
    """Raise `LockViolation` if the materialised locked file no longer matches the manifest."""
    manifest = _load_manifest(manifest_path)
    if manifest is None:
        raise LockViolation(f"no manifest at {manifest_path}")
    if not manifest.locked:
        return manifest
    actual = _sha256_text((processed_dir / f"{LOCKED_SPLIT}.jsonl").read_text(encoding="utf-8"))
    if actual != manifest.locked_sha256:
        raise LockViolation(f"{LOCKED_SPLIT}.jsonl sha256 {actual[:12]}... != locked {str(manifest.locked_sha256)[:12]}...")
    return manifest


def _load_batches(real_dir: Path) -> list[Dialogue]:
    dialogues: list[Dialogue] = []
    for path in sorted((real_dir / BATCHES_SUBDIR).glob("*.jsonl")):
        if path.name.endswith(".linkage.jsonl"):
            continue
        dialogues.extend(Dialogue.model_validate_json(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    return dialogues


def _load_manifest(path: Path) -> RealManifest | None:
    if not path.exists():
        return None
    return RealManifest.model_validate_json(path.read_text(encoding="utf-8"))


def _jsonl(dialogues: Sequence[Dialogue]) -> str:
    return "".join(d.model_dump_json() + "\n" for d in dialogues)


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _pos_neg(dialogues: Sequence[Dialogue]) -> dict[str, int]:
    positives = sum(1 for d in dialogues if not d.label.is_hard_negative)
    return {"positives": positives, "negatives": len(dialogues) - positives}


def dumps_manifest(manifest: RealManifest) -> str:
    return json.dumps(manifest.model_dump(mode="json"), ensure_ascii=False, indent=2)
