"""Lock test for the real held-out set (PLAN_2026-09 A3/A8): when `data/real/manifest.json`
exists and says locked, `real_heldout_v2.jsonl` must match its sha256 and share nothing
with train / val / augment. Skips until a real set exists -- it never fabricates one."""

from __future__ import annotations

from pathlib import Path

import pytest

from qorgan.data.build_corpus import normalize_for_dedup
from qorgan.data.real_allocation import LOCKED_SPLIT, MANIFEST_FILENAME, verify_lock
from qorgan.data.schema import Dialogue

_REPO = Path(__file__).resolve().parents[2]
_REAL = _REPO / "data" / "real"
_PROCESSED = _REPO / "data" / "processed"
_AUGMENT = _REPO / "data" / "augment"
_SYNTHETIC_SPLITS = ("train.jsonl", "val.jsonl")

pytestmark = pytest.mark.skipif(
    not (_REAL / MANIFEST_FILENAME).exists(), reason="no real held-out set ingested yet (docs/DATA_INTAKE.md)"
)


def _read(path: Path) -> list[Dialogue]:
    if not path.exists():
        return []
    return [Dialogue.model_validate_json(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def test_locked_file_matches_manifest():
    verify_lock(_REAL / MANIFEST_FILENAME, _PROCESSED)


def test_locked_set_shares_nothing_with_training_data():
    locked = _read(_PROCESSED / f"{LOCKED_SPLIT}.jsonl")
    training = [*(d for name in _SYNTHETIC_SPLITS for d in _read(_PROCESSED / name)), *(d for p in _AUGMENT.glob("*.jsonl") for d in _read(p))]
    train_ids = {d.id for d in training}
    train_texts = {normalize_for_dedup(d.transcript()) for d in training}
    assert not [d.id for d in locked if d.id in train_ids]
    assert not [d.id for d in locked if normalize_for_dedup(d.transcript()) in train_texts]
