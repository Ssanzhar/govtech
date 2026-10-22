"""`shift_local` (ADR D65): a deterministic plan, labels from briefs, the `shift` build
pipeline, a scrub fixed point, and a committed provenance record without dialogue text."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from qorgan.data import shift_local_plan as plan
from qorgan.data.shift_local_set import CHECKED_SPLITS, SPLIT_NAME, attempt_stats, build, provenance
from qorgan.data.shift_set import parse_raw_row
from qorgan.taxonomy import get_taxonomy

_REPO = Path(__file__).resolve().parents[2]
_BRIEFS = _REPO / "configs" / "shift_local_briefs.yaml"
_PROVENANCE = _REPO / "data" / "provenance" / "shift_local.json"


# --- briefs and plan --------------------------------------------------------------------------


def test_repo_briefs_are_balanced_and_use_known_tactics():
    briefs = plan.load_briefs(_BRIEFS)
    labels = [b.label for b in briefs]
    assert labels.count("scam") == labels.count("legit") >= 20
    known = set(get_taxonomy().tactic_ids())
    for brief in briefs:
        assert set(brief.tactics) <= known, brief.id
        assert (brief.label == "scam") == bool(brief.tactics), brief.id


def test_the_plan_is_deterministic_and_hits_the_language_targets():
    briefs = plan.load_briefs(_BRIEFS)
    first, second = plan.build_plan(briefs), plan.build_plan(briefs)
    assert first == second
    for language, target in plan.LANGUAGE_TARGETS.items():
        slots = [s for s in first if s.language == language]
        assert len(slots) == target
        scams = [s for s in slots if s.brief.label == "scam"]
        assert len(scams) == target // 2
        assert all(s.generator == plan.LANGUAGE_GENERATOR[language] for s in slots)
    assert len({s.slot_id for s in first}) == len(first)
    assert len({s.seed for s in first}) == len(first)


def test_every_brief_is_used_before_any_repeats_within_a_language():
    briefs = plan.load_briefs(_BRIEFS)
    for language in plan.LANGUAGE_TARGETS:
        for label in ("scam", "legit"):
            used = [s.brief.id for s in plan.build_plan(briefs) if s.language == language and s.brief.label == label]
            pool = sum(1 for b in briefs if b.label == label)
            assert len(set(used)) == min(len(used), pool)


def test_every_language_has_a_registered_generator_outside_the_training_families():
    """One family writes the set today (ADR D65 records why); it must not be Gemini or Claude."""
    assert set(plan.LANGUAGE_GENERATOR) == set(plan.LANGUAGE_TARGETS)
    assert set(plan.LANGUAGE_GENERATOR.values()) <= set(plan.GENERATORS)
    for key in set(plan.LANGUAGE_GENERATOR.values()):
        meta = " ".join(plan.GENERATORS[key].values()).lower()
        assert "gemini" not in meta and "claude" not in meta and "gemma" not in meta


def test_the_label_comes_from_the_brief_and_the_raw_row_parses():
    briefs = plan.load_briefs(_BRIEFS)
    slots = plan.build_plan(briefs, targets={"ru": 2})
    utterances = tuple(("caller" if i % 2 == 0 else "callee", f"Реплика номер {i}") for i in range(8))
    for slot in slots:
        dialogue = parse_raw_row(plan.raw_row(slot, utterances))
        assert dialogue.label.is_scam == (slot.brief.label == "scam")
        assert dialogue.label.trigger_spans == ()
        assert {t.id for t in dialogue.label.tactic_tags} == set(slot.brief.tactics)


def test_the_prompt_carries_the_brief_language_and_format():
    slot = plan.build_plan(plan.load_briefs(_BRIEFS), targets={"kk": 2})[0]
    prompt = plan.user_prompt(slot)
    assert slot.brief.brief in prompt
    assert plan.LANGUAGE_INSTRUCTIONS["kk"] in prompt
    assert "CALLER:" in prompt and "CALLEE:" in prompt


def test_legit_briefs_with_tactics_are_refused(tmp_path):
    path = tmp_path / "b.yaml"
    path.write_text("version: 1\nbriefs:\n  - {id: x, label: legit, brief: hi, tactics: [urgency]}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="legit tactics"):
        plan.load_briefs(path)


# --- build ------------------------------------------------------------------------------------


def _row(slot_id: str, language: str, texts: list[str], *, scam: bool) -> dict:
    return {
        "id": slot_id, "language": language, "scenario": "s_test" if scam else "l_test",
        "utterances": [{"speaker": "caller" if i % 2 == 0 else "callee", "text": t} for i, t in enumerate(texts)],
        "risk": 0.95 if scam else 0.02, "is_hard_negative": not scam,
        "tactic_tags": [{"id": "otp_request", "weight": 1.0}] if scam else [], "trigger_phrases": [],
    }


def _setup(tmp_path: Path, *, leak_into: str | None = None) -> tuple[Path, Path]:
    raw = tmp_path / "raw"
    raw.mkdir()
    scam = _row("sl_ru_scam_01", "ru", ["Это курьер, назовите код.", "Какой код?", "Из сообщения, ИИН 770808300300."], scam=True)
    legit = _row("sl_ru_legit_01", "ru", ["Это поликлиника, напоминаю о приёме.", "Спасибо.", "До свидания."], scam=False)
    (raw / "ru.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in (scam, legit)) + "\n", encoding="utf-8")
    processed = tmp_path / "processed"
    processed.mkdir()
    for split in CHECKED_SPLITS:
        rows = [parse_raw_row(legit).model_copy(update={"id": f"{split}_leak"})] if split == leak_into else []
        (processed / f"{split}.jsonl").write_text("".join(d.model_dump_json() + "\n" for d in rows), encoding="utf-8")
    return raw, processed


def test_build_writes_a_scrubbed_split_named_shift_local(tmp_path):
    raw, processed = _setup(tmp_path)
    manifest = build(raw_dir=raw, processed_dir=processed)
    assert manifest["split"] == SPLIT_NAME
    assert manifest["counts"]["total"] == 2
    text = (processed / f"{SPLIT_NAME}.jsonl").read_text(encoding="utf-8")
    assert "770808300300" not in text  # scrubbed (ADR D45)
    assert "shift" in manifest["disjoint_from"]


def test_build_refuses_a_row_that_already_exists_in_shift(tmp_path):
    raw, processed = _setup(tmp_path, leak_into="shift")
    with pytest.raises(ValueError, match="not disjoint"):
        build(raw_dir=raw, processed_dir=processed)


def test_provenance_has_counts_and_hashes_but_no_dialogue_text(tmp_path):
    raw, processed = _setup(tmp_path)
    manifest = build(raw_dir=raw, processed_dir=processed)
    attempts = [
        {"slot": "sl_ru_scam_01", "language": "ru", "label": "scam", "reason": "language", "generator": "yandexgpt5_lite"},
        {"slot": "sl_ru_scam_01", "language": "ru", "label": "scam", "reason": "accepted", "generator": "yandexgpt5_lite"},
        {"slot": "sl_ru_legit_01", "language": "ru", "label": "legit", "reason": "accepted", "generator": "yandexgpt5_lite"},
    ]
    record = provenance(manifest=manifest, attempts=attempts, briefs_path=_BRIEFS, llama_cpp="version: test")
    blob = json.dumps(record, ensure_ascii=False)
    assert "поликлиника" not in blob and "курьер" not in blob
    assert record["attempts"]["ru/scam"] == {
        "slots": 1, "attempts": 2, "accepted": 1, "slots_unfilled": 0, "rejected": {"language": 1}, "rejection_rate": 0.5,
    }
    assert record["acceptance"]["classifier_used_for_selection"] is False
    assert record["published"] is False
    assert record["generators"]["yandexgpt5_lite"]["sha256"] == plan.GENERATORS["yandexgpt5_lite"]["sha256"]


def test_attempt_stats_counts_unfilled_slots():
    stats = attempt_stats([{"slot": "a", "language": "kk", "label": "legit", "reason": "format"}])
    assert stats["kk/legit"]["slots_unfilled"] == 1 and stats["kk/legit"]["rejection_rate"] == 1.0


def test_committed_provenance_never_carries_text():
    if not _PROVENANCE.exists():
        pytest.skip("shift_local not built yet")
    record = json.loads(_PROVENANCE.read_text(encoding="utf-8"))
    assert record["split"] == SPLIT_NAME and record["published"] is False
    assert "utterances" not in json.dumps(record)


@pytest.mark.parametrize("split_name", [SPLIT_NAME, "shift_local_2"])
def test_local_shift_local_shares_nothing_with_training_data(split_name):
    """Track 2 adds generated training rows: none may coincide with the independent held-outs."""
    from qorgan.data.build_corpus import normalize_for_dedup
    from qorgan.data.schema import Dialogue

    processed = _REPO / "data" / "processed"
    held_out = processed / f"{split_name}.jsonl"
    if not held_out.exists():
        pytest.skip(f"{split_name} not built locally")

    def texts(path: Path) -> set[str]:
        if not path.exists():
            return set()
        lines = path.read_text(encoding="utf-8").splitlines()
        return {normalize_for_dedup(Dialogue.model_validate_json(l).transcript()) for l in lines if l.strip()}

    sources = [processed / "train.jsonl", processed / "val.jsonl", *(_REPO / "data" / "augment").glob("*.jsonl")]
    training = set().union(*(texts(p) for p in sources))
    assert not texts(held_out) & training
