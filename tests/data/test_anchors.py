"""Tests for the curated `real_heldout` anchors — every anchor must be schema-valid,
its trigger spans verbatim, and the positive/negative balance sane."""

from qorgan.data.anchors import ANCHOR_SPECS, build_anchor_dialogues
from qorgan.data.schema import Dialogue, validate_verbatim_spans
from qorgan.taxonomy import get_taxonomy


def test_all_anchors_build_into_valid_dialogues():
    dialogues = build_anchor_dialogues()
    assert len(dialogues) == len(ANCHOR_SPECS)
    assert all(isinstance(d, Dialogue) for d in dialogues)


def test_anchor_ids_are_unique():
    dialogues = build_anchor_dialogues()
    ids = [d.id for d in dialogues]
    assert len(ids) == len(set(ids))


def test_anchor_trigger_spans_are_verbatim():
    for dialogue in build_anchor_dialogues():
        # Raises SpanValidationError if any span is not a verbatim slice.
        validate_verbatim_spans(dialogue.label.trigger_spans, dialogue.transcript())


def test_anchor_tactic_ids_exist_in_taxonomy():
    valid_ids = set(get_taxonomy().tactic_ids())
    for dialogue in build_anchor_dialogues():
        for tag in dialogue.label.tactic_tags:
            assert tag.id in valid_ids, f"{dialogue.id}: unknown tactic {tag.id!r}"


def test_anchor_set_has_both_scams_and_hard_negatives():
    dialogues = build_anchor_dialogues()
    positives = [d for d in dialogues if not d.label.is_hard_negative and d.label.tactic_tags]
    hard_negatives = [d for d in dialogues if d.label.is_hard_negative]
    assert len(positives) >= 3
    assert len(hard_negatives) >= 3


def test_hard_negatives_have_no_tags_or_spans():
    for dialogue in build_anchor_dialogues():
        if dialogue.label.is_hard_negative:
            assert dialogue.label.tactic_tags == ()
            assert dialogue.label.trigger_spans == ()
            assert dialogue.label.risk < 0.1


def test_positives_have_high_risk_and_spans():
    for dialogue in build_anchor_dialogues():
        if dialogue.label.tactic_tags:
            assert dialogue.label.risk >= 0.7
            assert len(dialogue.label.trigger_spans) >= 1


def test_all_languages_represented():
    languages = {d.language for d in build_anchor_dialogues()}
    assert {"ru", "kk", "mixed"} <= languages
