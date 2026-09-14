"""Tests for the curated `authored_heldout` anchors — every anchor must be schema-valid,
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
    # Widened (Phase 1A) floor: was >=3/>=3 when the anchor set was 27 dialogues.
    assert len(positives) >= 15
    assert len(hard_negatives) >= 10


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


# --- Phase 1A: widened `authored_heldout` anchor set (was 27 dialogues / ~14 negatives) -------
# The hand-written eval anchor set was statistically underpowered for the project's
# FPR-first discipline (CLAUDE.md SS3.5). These tests pin the widened counts and balance.


def test_anchor_set_widened_to_forty_two():
    """27 -> 42 dialogues: +10 negatives (telecom/delivery/other legit calls) and +5
    positives (one anchor each for the previously-starved tail tactics)."""
    assert len(ANCHOR_SPECS) == 42


def test_widened_negative_positive_balance_weighted_toward_negatives():
    dialogues = build_anchor_dialogues()
    negatives = [d for d in dialogues if not d.label.tactic_tags]
    positives = [d for d in dialogues if d.label.tactic_tags]
    assert len(negatives) == 24
    assert len(positives) == 18
    assert len(negatives) > len(positives)


def test_every_negative_spec_has_no_tactic_ids_or_phrases():
    """A negative anchor (empty tactic_ids) must never carry trigger phrases -- covers
    ALL benign anchors, not just `is_hard_negative` ones (extends
    `test_hard_negatives_have_no_tags_or_spans` to plain benign anchors too, e.g. courier/
    chitchat calls that are negatives but not flagged `is_hard_negative`)."""
    for spec in ANCHOR_SPECS:
        if not spec.tactic_ids:
            assert spec.trigger_phrases == ()


def test_all_negative_dialogues_have_low_risk_and_no_spans():
    for dialogue in build_anchor_dialogues():
        if not dialogue.label.tactic_tags:
            assert dialogue.label.risk < 0.1
            assert dialogue.label.trigger_spans == ()


def test_widened_negatives_have_solid_kk_and_mixed_language_coverage():
    """The widening brief calls for legit telecom/delivery/other calls spanning RU+KK+mixed,
    weighted toward negatives -- assert the negative set now has solid kk/mixed coverage."""
    dialogues = build_anchor_dialogues()
    negatives = [d for d in dialogues if not d.label.tactic_tags]
    kk_or_mixed_negatives = [d for d in negatives if d.language in ("kk", "mixed")]
    assert len(kk_or_mixed_negatives) >= 14


def test_starved_tail_tactics_now_have_measurable_recall():
    """mule_recruitment, secrecy, remote_access, investment_scam, prize_lottery were
    under-represented; each must now appear in at least 2 anchors so recall on the tail
    is actually measurable on `authored_heldout`."""
    starved_tactics = (
        "mule_recruitment",
        "secrecy",
        "remote_access",
        "investment_scam",
        "prize_lottery",
    )
    valid_ids = set(get_taxonomy().tactic_ids())
    assert set(starved_tactics) <= valid_ids

    counts = dict.fromkeys(starved_tactics, 0)
    for dialogue in build_anchor_dialogues():
        for tag in dialogue.label.tactic_tags:
            if tag.id in counts:
                counts[tag.id] += 1

    for tactic_id, count in counts.items():
        assert count >= 2, f"{tactic_id} still statistically starved: only {count} anchor(s)"


def test_new_positive_anchors_have_multi_label_tags():
    """Scam positives should carry 2-4 tactic tags, per the widening brief's realism bar."""
    new_positive_ids = {
        "real_scam_mule_recruit_reward_ru",
        "real_scam_tax_refund_secrecy_kk",
        "real_scam_remote_access_antivirus_mixed",
        "real_scam_investment_platform_ru",
        "real_scam_prize_phone_kk",
    }
    dialogues = {d.id: d for d in build_anchor_dialogues()}
    assert new_positive_ids <= dialogues.keys()
    for anchor_id in new_positive_ids:
        tag_count = len(dialogues[anchor_id].label.tactic_tags)
        assert 2 <= tag_count <= 4, f"{anchor_id}: expected 2-4 tags, got {tag_count}"
