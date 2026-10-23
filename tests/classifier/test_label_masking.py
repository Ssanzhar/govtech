"""Label masking and thin-head dropping for the taxonomy-v2 retrain (ADR D63 phase 2)."""

from __future__ import annotations

import numpy as np

from qorgan.classifier.calibrate import tune_tactic_thresholds
from qorgan.classifier.linear_train import labelled_positives, provenance_mask, thin_heads
from qorgan.classifier.multilabel import MultiLabelHead, out_of_fold_proba
from qorgan.data.schema import Dialogue, Label, TacticTag, Utterance


def _d(i: str, tags=()):
    return Dialogue(id=i, language="ru", utterances=(Utterance(speaker="caller", text=f"t {i}"),),
                    label=Label(risk=0.9 if tags else 0.02, tactic_tags=tuple(TacticTag(id=t) for t in tags)))


def test_masked_rows_never_reach_a_column():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(40, 3))
    y = np.zeros((40, 2))
    y[:10, 0] = 1
    y[:10, 1] = 1
    # rows 20..39 would be "negatives" for column 1 but are unlabelled for it
    mask = np.ones((40, 2), dtype=bool)
    mask[20:, 1] = False
    head = MultiLabelHead(("a", "b")).fit(x, y, label_mask=mask)
    poisoned = MultiLabelHead(("a", "b")).fit(x[:20], y[:20])
    np.testing.assert_allclose(head.models["b"].coef_, poisoned.models["b"].coef_)
    assert not np.allclose(head.models["a"].coef_, poisoned.models["a"].coef_)
    oof = out_of_fold_proba(x, y, ("a", "b"), folds=4, seed=1, label_mask=mask)
    assert oof.shape == (40, 2)


def test_a_column_with_only_masked_negatives_is_skipped_not_crashed():
    x = np.eye(4)
    y = np.array([[1.0], [1.0], [0.0], [0.0]])
    mask = np.array([[True], [True], [False], [False]])
    assert MultiLabelHead(("a",)).fit(x, y, label_mask=mask).models["a"] is None


def test_provenance_mask_only_masks_v1_rows_on_new_tactics():
    rows = [_d("g1"), _d("c1"), _d("c1-asr"), _d("unknown")]
    generator = {"g1": "gemini", "c1": "claude", "c1-asr": "claude"}
    mask = provenance_mask(rows, ("urgency", "refund_request"), generator, new_tactics=("refund_request",), labelled_by="claude")
    assert mask.tolist() == [[True, False], [True, True], [True, True], [True, False]]


def test_thin_heads_and_labelled_positives():
    rows = [_d("a", ("refund_request",)), _d("b", ("refund_request", "urgency")), _d("c", ("urgency",))]
    mask = np.array([[True, False], [True, True], [True, True]])
    pos = labelled_positives(rows, ("urgency", "refund_request"), mask)
    assert pos == {"urgency": 2, "refund_request": 1}
    assert thin_heads(pos, {"urgency": 0, "refund_request": 6}, minimum=8) == ("urgency", "refund_request")
    assert thin_heads(pos, {"urgency": 6, "refund_request": 7}, minimum=8) == ()


def test_tuner_ignores_masked_cells():
    probs, truth = [[0.9], [0.8], [0.6]], [[1], [1], [0]]
    unmasked = tune_tactic_thresholds(probs, truth, ("a",), min_positives=2)
    masked = tune_tactic_thresholds(probs, truth, ("a",), min_positives=2, mask=[[True], [True], [False]])
    assert masked == {"a": 0.5} and unmasked["a"] > 0.6


def test_provenance_mask_accepts_several_v2_writers():
    """Grid rows written by Gemini (D63 v2.2) carry v2 labels like Claude's; only the legacy
    v1 corpus rows are masked out of the new heads."""
    from qorgan.data.gen.fold import V2_LABELLED_GENERATORS

    rows = [_d(i) for i in ("old", "claude_row", "gem_row")]
    generator = {"old": "gemini", "claude_row": "claude", "gem_row": "gemini_grid"}
    mask = provenance_mask(rows, ("urgency", "refund_request"), generator, new_tactics=("refund_request",),
                           labelled_by=V2_LABELLED_GENERATORS)
    assert mask.tolist() == [[True, False], [True, True], [True, True]]
