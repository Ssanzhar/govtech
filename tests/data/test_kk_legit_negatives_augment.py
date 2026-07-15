"""Tests for `data/augment/kk_legit_negatives.jsonl` -- the hand-authored KK legit hard
negatives (bank / gov / telecom "no code/data needed" calls) that stabilize the KK legit-call
decision boundary in train (see `docs/eval_report.md` addendum + CLAUDE.md SS6/SS7).

Core invariants: every record is schema-valid, labeled as a risk-0 hard negative, and --
critically -- **no utterance is copied or lightly paraphrased from the `real_heldout` anchor
set** (`src/qorgan/data/anchors.py`). That set is the held-out generalization signal; any
verbatim overlap between train-only augmentation and it would be leakage.
"""

from __future__ import annotations

from pathlib import Path

from qorgan.data.anchors import build_anchor_dialogues
from qorgan.data.build_corpus import normalize_for_dedup
from qorgan.data.schema import Dialogue

_AUGMENT_PATH = (
    Path(__file__).resolve().parents[2] / "data" / "augment" / "kk_legit_negatives.jsonl"
)


def _load_augment_dialogues() -> tuple[Dialogue, ...]:
    dialogues = []
    for line in _AUGMENT_PATH.read_text(encoding="utf-8").splitlines():
        if line.strip():
            dialogues.append(Dialogue.model_validate_json(line))
    return tuple(dialogues)


def test_augment_file_exists_and_has_expected_count():
    dialogues = _load_augment_dialogues()
    assert 12 <= len(dialogues) <= 15


def test_all_records_are_valid_dialogues():
    dialogues = _load_augment_dialogues()
    assert all(isinstance(d, Dialogue) for d in dialogues)


def test_all_records_are_zero_risk_hard_negatives_with_no_tags_or_spans():
    for dialogue in _load_augment_dialogues():
        assert dialogue.label.risk <= 0.05
        assert dialogue.label.tactic_tags == ()
        assert dialogue.label.trigger_spans == ()
        assert dialogue.label.is_hard_negative is True


def test_ids_are_unique_and_namespaced():
    dialogues = _load_augment_dialogues()
    ids = [d.id for d in dialogues]
    assert len(ids) == len(set(ids))
    assert all(dialogue_id.startswith("kk_legit_") for dialogue_id in ids)


def test_majority_kazakh_with_some_mixed_language():
    dialogues = _load_augment_dialogues()
    languages = [d.language for d in dialogues]
    assert languages.count("kk") > len(dialogues) / 2
    assert languages.count("mixed") >= 1
    assert set(languages) <= {"kk", "mixed"}


def test_covers_bank_gov_and_telecom_domains():
    dialogues = _load_augment_dialogues()
    ids = {d.id for d in dialogues}
    assert any("bank_service" in i for i in ids)
    assert any("gov_service" in i for i in ids)
    assert any("telecom_notice" in i for i in ids)


def test_several_dialogues_reassure_in_kazakh():
    """Several augment dialogues should contain a Kazakh negation-of-need reassurance
    phrase, mirroring real institutional behaviour scammers don't replicate."""
    kk_reassurance_markers = ("қажеті жоқ", "қажет жоқ", "қажет емес", "талап етпей", "сұрамаймыз")
    hits = 0
    for dialogue in _load_augment_dialogues():
        transcript = dialogue.transcript()
        if any(marker in transcript for marker in kk_reassurance_markers):
            hits += 1
    assert hits >= 5


def test_no_augment_utterance_appears_verbatim_in_any_anchor_dialogue():
    """The held-out `real_heldout` anchor set must stay leakage-free: no augment utterance
    (normalized: whitespace-collapsed, lowercased) may exactly match any anchor utterance."""
    anchor_utterance_texts = {
        normalize_for_dedup(utterance.text)
        for dialogue in build_anchor_dialogues()
        for utterance in dialogue.utterances
    }
    for dialogue in _load_augment_dialogues():
        for utterance in dialogue.utterances:
            normalized = normalize_for_dedup(utterance.text)
            assert normalized not in anchor_utterance_texts, (
                f"Augment utterance in {dialogue.id!r} appears verbatim in an anchor "
                f"dialogue: {utterance.text!r}"
            )


def test_no_augment_transcript_appears_verbatim_in_any_anchor_transcript():
    """Belt-and-braces: also check whole-transcript equality (catches accidental
    full-dialogue duplication even if per-utterance splitting ever changes)."""
    anchor_transcripts = {
        normalize_for_dedup(dialogue.transcript()) for dialogue in build_anchor_dialogues()
    }
    for dialogue in _load_augment_dialogues():
        assert normalize_for_dedup(dialogue.transcript()) not in anchor_transcripts
