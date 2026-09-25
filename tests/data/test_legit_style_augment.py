"""Leakage + shape checks for `data/augment/legit_style_scams.jsonl` (PLAN A9b/A9c): train-only
positives in the legit-sounding register, built from TRAIN scams, sharing nothing with any
evaluation split. Skips until the file has been generated."""

from __future__ import annotations

from pathlib import Path

import pytest

from qorgan.classifier.cue_lexicon import load_cue_lexicon
from qorgan.data.adversarial import cue_hits
from qorgan.data.build_corpus import normalize_for_dedup
from qorgan.data.schema import Dialogue

_REPO = Path(__file__).resolve().parents[2]
_AUGMENT = _REPO / "data" / "augment" / "legit_style_scams.jsonl"
_PROCESSED = _REPO / "data" / "processed"
_EVAL_SPLITS = ("test", "ood", "authored_heldout", "adversarial", "adversarial_legit")

pytestmark = pytest.mark.skipif(not _AUGMENT.exists(), reason="legit-style augmentation not generated")


def _read(path: Path) -> list[Dialogue]:
    if not path.exists():
        return []
    return [Dialogue.model_validate_json(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def test_augment_rows_are_train_style_positives_in_the_legit_register():
    rows = _read(_AUGMENT)
    assert rows, "empty augmentation file"
    lexicon = load_cue_lexicon()
    assert all(r.id.startswith("aug-legit-") for r in rows)
    assert all(not r.label.is_hard_negative and r.label.risk >= 0.5 and r.label.tactic_tags for r in rows)
    assert all(r.label.trigger_spans == () for r in rows)  # paraphrased text: nothing re-grounded by hand
    assert all(not cue_hits(r.transcript(), lexicon) for r in rows)  # the adversary's constraint holds


def test_augment_shares_nothing_with_any_evaluation_split():
    rows = _read(_AUGMENT)
    own_texts = {normalize_for_dedup(r.transcript()) for r in rows}
    for split in _EVAL_SPLITS:
        eval_rows = _read(_PROCESSED / f"{split}.jsonl")
        eval_texts = {normalize_for_dedup(d.transcript()) for d in eval_rows}
        assert not (own_texts & eval_texts), f"legit-style augmentation overlaps {split}"
        # and none of them paraphrases an evaluation scam (source ids are train ids)
        eval_ids = {d.id for d in eval_rows} | {d.id.removeprefix("adv-") for d in eval_rows}
        assert not {r.id.removeprefix("aug-legit-") for r in rows} & eval_ids, f"augment built from {split} sources"
