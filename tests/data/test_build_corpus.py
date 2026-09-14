"""TDD tests for `qorgan.data.build_corpus` — deterministic assemble/scrub/dedup/split
+ manifest. All pure functions; the orchestrator is exercised against injected inputs
and a tmp output dir (no network, no real corpus)."""

import json

import pytest

from qorgan.data.build_corpus import (
    assign_split,
    build_corpus,
    build_manifest,
    content_hash,
    deduplicate,
    normalize_for_dedup,
    scrub_dialogue,
    split_dialogues,
)
from qorgan.data.schema import Dialogue, Label, TacticTag, Utterance, spans_from_phrases


def _dialogue(did, texts, *, tags=(), phrases=(), risk=0.9, hard_negative=False, language="ru"):
    utterances = tuple(Utterance(speaker="caller", text=t) for t in texts)
    transcript = "\n".join(texts)
    spans = spans_from_phrases(phrases, transcript)
    return Dialogue(
        id=did,
        language=language,
        utterances=utterances,
        label=Label(
            risk=risk,
            tactic_tags=tuple(TacticTag(id=t) for t in tags),
            trigger_spans=spans,
            is_hard_negative=hard_negative,
        ),
    )


# --- normalize_for_dedup -----------------------------------------------------------------


def test_normalize_collapses_whitespace_and_lowercases():
    assert normalize_for_dedup("  Привет   МИР \n\t Тест ") == "привет мир тест"


def test_normalize_idempotent():
    once = normalize_for_dedup("Hello   World")
    assert normalize_for_dedup(once) == once


# --- deduplicate -------------------------------------------------------------------------


def test_deduplicate_removes_exact_duplicates_keeps_first():
    d1 = _dialogue("a", ["Привет мир"])
    d2 = _dialogue("b", ["Привет мир"])  # same transcript
    d3 = _dialogue("c", ["Другой текст"])
    result = deduplicate([d1, d2, d3])
    assert [d.id for d in result] == ["a", "c"]


def test_deduplicate_treats_whitespace_case_variants_as_same():
    d1 = _dialogue("a", ["Привет   мир"])
    d2 = _dialogue("b", ["привет мир"])
    result = deduplicate([d1, d2])
    assert len(result) == 1


def test_deduplicate_empty_returns_empty():
    assert deduplicate([]) == ()


# --- scrub_dialogue ----------------------------------------------------------------------


def test_scrub_dialogue_redacts_phone_number():
    d = _dialogue("a", ["Позвоните на +7 777 123 45 67 прямо сейчас"])
    scrubbed = scrub_dialogue(d)
    assert "+7 777 123 45 67" not in scrubbed.transcript()
    assert "[PHONE]" in scrubbed.transcript()
    assert isinstance(scrubbed, Dialogue)


def test_scrub_dialogue_keeps_clean_trigger_span():
    d = _dialogue(
        "a",
        ["Продиктуйте код из SMS и позвоните на +7 777 123 45 67"],
        phrases=["код из SMS"],
    )
    scrubbed = scrub_dialogue(d)
    # The clean trigger phrase survives and is re-grounded verbatim.
    assert any(s.text == "код из SMS" for s in scrubbed.label.trigger_spans)


def test_scrub_dialogue_drops_span_whose_text_had_pii():
    d = _dialogue(
        "a",
        ["Переведите на счёт 4400430212345678 сейчас"],
        phrases=["4400430212345678"],  # a card number that will be scrubbed away
    )
    scrubbed = scrub_dialogue(d)
    assert scrubbed.label.trigger_spans == ()  # dropped, not left dangling


def test_scrub_dialogue_preserves_id_language_and_flags():
    d = _dialogue("xyz", ["Обычный текст"], hard_negative=True, risk=0.02, language="kk")
    scrubbed = scrub_dialogue(d)
    assert scrubbed.id == "xyz"
    assert scrubbed.language == "kk"
    assert scrubbed.label.is_hard_negative is True
    assert scrubbed.label.risk == 0.02


def test_scrub_dialogue_idempotent():
    d = _dialogue("a", ["Мой номер +7 777 123 45 67"])
    once = scrub_dialogue(d)
    twice = scrub_dialogue(once)
    assert once == twice


# --- assign_split ------------------------------------------------------------------------


def test_assign_split_returns_known_bucket():
    bucket = assign_split("some_id", seed=42, train_fraction=0.7, val_fraction=0.15)
    assert bucket in {"train", "val", "test"}


def test_assign_split_is_deterministic():
    a = assign_split("id_1", seed=42, train_fraction=0.7, val_fraction=0.15)
    b = assign_split("id_1", seed=42, train_fraction=0.7, val_fraction=0.15)
    assert a == b


def test_assign_split_changes_with_seed():
    # Over a batch of ids, a different seed produces a different partition.
    ids = [f"id_{i}" for i in range(200)]
    part_a = [assign_split(i, seed=1, train_fraction=0.7, val_fraction=0.15) for i in ids]
    part_b = [assign_split(i, seed=2, train_fraction=0.7, val_fraction=0.15) for i in ids]
    assert part_a != part_b


def test_assign_split_roughly_honours_fractions():
    ids = [f"id_{i}" for i in range(3000)]
    buckets = [assign_split(i, seed=42, train_fraction=0.7, val_fraction=0.15) for i in ids]
    train_frac = buckets.count("train") / len(ids)
    assert train_frac == pytest.approx(0.7, abs=0.05)


# --- split_dialogues ---------------------------------------------------------------------


def test_split_dialogues_partitions_without_loss_or_duplication():
    dialogues = [_dialogue(f"id_{i}", [f"текст номер {i}"]) for i in range(100)]
    splits = split_dialogues(dialogues, seed=42, train_fraction=0.7, val_fraction=0.15)
    assert set(splits) == {"train", "val", "test"}
    all_ids = [d.id for part in splits.values() for d in part]
    assert sorted(all_ids) == sorted(d.id for d in dialogues)
    assert len(all_ids) == len(set(all_ids))


def test_split_dialogues_reproducible():
    dialogues = [_dialogue(f"id_{i}", [f"текст {i}"]) for i in range(50)]
    s1 = split_dialogues(dialogues, seed=7, train_fraction=0.7, val_fraction=0.15)
    s2 = split_dialogues(dialogues, seed=7, train_fraction=0.7, val_fraction=0.15)
    assert {k: [d.id for d in v] for k, v in s1.items()} == {
        k: [d.id for d in v] for k, v in s2.items()
    }


# --- content_hash ------------------------------------------------------------------------


def test_content_hash_stable_and_order_independent():
    a = _dialogue("a", ["текст а"])
    b = _dialogue("b", ["текст б"])
    assert content_hash([a, b]) == content_hash([b, a])


def test_content_hash_changes_when_data_changes():
    a = _dialogue("a", ["текст а"])
    a2 = _dialogue("a", ["другой текст"])
    assert content_hash([a]) != content_hash([a2])


def test_content_hash_is_hex_sha256():
    h = content_hash([_dialogue("a", ["x y z"])])
    assert len(h) == 64
    int(h, 16)  # parses as hex


# --- build_manifest ----------------------------------------------------------------------


def test_build_manifest_counts_and_metadata():
    splits = {
        "train": [_dialogue("t1", ["a b"], tags=["urgency"]), _dialogue("t2", ["c d"], hard_negative=True, risk=0.02)],
        "val": [_dialogue("v1", ["e f"], language="kk")],
        "test": [_dialogue("te1", ["g h"], language="mixed")],
    }
    authored_heldout = [_dialogue("r1", ["i j"], hard_negative=True, risk=0.02)]
    manifest = build_manifest(
        splits, authored_heldout, seed=42, train_fraction=0.7, val_fraction=0.15
    )
    assert manifest["seed"] == 42
    assert manifest["counts"]["train"]["total"] == 2
    assert manifest["counts"]["train"]["hard_negatives"] == 1
    assert manifest["counts"]["authored_heldout"]["total"] == 1
    assert manifest["total"] == 5
    assert manifest["fractions"] == {"train": 0.7, "val": 0.15, "test": pytest.approx(0.15)}
    assert len(manifest["content_hash"]) == 64
    # per-language breakdown present
    assert "by_language" in manifest["counts"]["val"]


# --- build_corpus (orchestrator) ---------------------------------------------------------


def test_build_corpus_end_to_end_writes_splits_and_manifest(tmp_path):
    synthetic = [_dialogue(f"syn_{i}", [f"синтетический текст {i}"]) for i in range(60)]
    anchors = [
        _dialogue("anc_1", ["реальный звонок один"], hard_negative=True, risk=0.02),
        _dialogue("anc_2", ["реальный звонок два"], tags=["otp_request"], phrases=[]),
    ]
    processed = tmp_path / "processed"

    manifest = build_corpus(
        dialogues=synthetic,
        anchor_dialogues=anchors,
        processed_dir=processed,
        seed=42,
        train_fraction=0.7,
        val_fraction=0.15,
    )

    for name in ("train", "val", "test", "authored_heldout"):
        assert (processed / f"{name}.jsonl").exists()
    assert (processed / "manifest.json").exists()

    # authored_heldout equals the anchors, kept fully separate from the split corpus.
    heldout_lines = (processed / "authored_heldout.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(heldout_lines) == 2
    split_ids = []
    for name in ("train", "val", "test"):
        for line in (processed / f"{name}.jsonl").read_text(encoding="utf-8").strip().splitlines():
            split_ids.append(Dialogue.model_validate_json(line).id)
    assert sorted(split_ids) == sorted(d.id for d in synthetic)
    assert "anc_1" not in split_ids

    # manifest.json round-trips and matches the returned manifest exactly.
    on_disk = json.loads((processed / "manifest.json").read_text(encoding="utf-8"))
    assert on_disk["total"] == 62
    assert manifest == on_disk


def test_build_corpus_scrubs_pii_before_writing(tmp_path):
    synthetic = [_dialogue("syn_1", ["Мой номер +7 777 123 45 67, перезвоните"])]
    anchors = [_dialogue("anc_1", ["чистый текст"], hard_negative=True, risk=0.02)]
    processed = tmp_path / "processed"
    build_corpus(
        dialogues=synthetic,
        anchor_dialogues=anchors,
        processed_dir=processed,
        seed=42,
        train_fraction=0.7,
        val_fraction=0.15,
    )
    all_text = "".join(
        (processed / f"{n}.jsonl").read_text(encoding="utf-8")
        for n in ("train", "val", "test")
    )
    assert "+7 777 123 45 67" not in all_text
    assert "[PHONE]" in all_text


def test_build_corpus_adds_augmentation_to_train_only(tmp_path):
    synthetic = [_dialogue(f"syn_{i}", [f"синтетический текст {i}"]) for i in range(40)]
    augment = [
        _dialogue(f"aug_{i}", [f"аугментация {i}"], hard_negative=True, risk=0.02) for i in range(6)
    ]
    processed = tmp_path / "processed"
    manifest = build_corpus(
        dialogues=synthetic,
        anchor_dialogues=[],
        augment_dialogues=augment,
        processed_dir=processed,
        seed=42,
        train_fraction=0.7,
        val_fraction=0.15,
    )

    def ids(name):
        text = (processed / f"{name}.jsonl").read_text(encoding="utf-8").strip()
        return [Dialogue.model_validate_json(x).id for x in text.splitlines() if x]

    train_ids = ids("train")
    other_ids = ids("val") + ids("test") + ids("authored_heldout")
    assert all(f"aug_{i}" in train_ids for i in range(6))  # augmentation lands in train
    assert not any(x.startswith("aug_") for x in other_ids)  # and nowhere else
    assert manifest["train_augment_count"] == 6


def test_build_corpus_augmentation_defaults_to_none(tmp_path):
    synthetic = [_dialogue(f"syn_{i}", [f"текст {i}"]) for i in range(20)]
    processed = tmp_path / "processed"
    manifest = build_corpus(
        dialogues=synthetic, anchor_dialogues=[], processed_dir=processed,
        seed=42, train_fraction=0.7, val_fraction=0.15,
    )
    assert manifest["train_augment_count"] == 0


def test_build_corpus_deduplicates_synthetic(tmp_path):
    synthetic = [
        _dialogue("syn_1", ["повторяющийся текст"]),
        _dialogue("syn_2", ["повторяющийся текст"]),
        _dialogue("syn_3", ["уникальный текст"]),
    ]
    processed = tmp_path / "processed"
    manifest = build_corpus(
        dialogues=synthetic,
        anchor_dialogues=[],
        processed_dir=processed,
        seed=42,
        train_fraction=0.7,
        val_fraction=0.15,
    )
    # 3 in, 1 duplicate removed -> 2 across splits.
    split_total = sum(manifest["counts"][n]["total"] for n in ("train", "val", "test"))
    assert split_total == 2
