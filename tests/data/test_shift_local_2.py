"""`shift_local_2` (PLAN_2026-10 §4 A3; the D65 pipeline, second batch): the same briefs and model
under a new seed, kk/ru only (D70: the English slots go to Kazakh), ids that cannot collide with
`shift_local`, near-duplicate disjointness from every split, and never published.
`shift_local` itself must not move: its plan is pinned by a fingerprint."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from qorgan.data import shift_local_plan as plan
from qorgan.data.shift_local_set import build, checked_splits, prior_accepted_texts, provenance
from qorgan.data.shift_set import parse_raw_row

_REPO = Path(__file__).resolve().parents[2]
_BRIEFS = _REPO / "configs" / "shift_local_briefs.yaml"
_SECOND = "shift_local_2"
# sha256 of the D65 plan (every slot id, language, brief, persona, outcome, length and seed),
# taken before the second batch was added. If this moves, `shift_local` no longer reproduces.
_SHIFT_LOCAL_PLAN_SHA256 = "c5e0a6c287a5f3cce76f8db576f2b799a516201c0205be36bde14470eb06c54a"


def _fingerprint(slots: tuple[plan.Slot, ...]) -> str:
    rows = [[s.slot_id, s.language, s.brief.id, s.persona, s.outcome, s.turns, s.seed, s.generator] for s in slots]
    return hashlib.sha256(json.dumps(rows, ensure_ascii=False).encode("utf-8")).hexdigest()


# --- registry and plan ------------------------------------------------------------------------


def test_both_batches_are_registered_with_distinct_seeds_and_prefixes():
    first, second = plan.LOCAL_SPLITS["shift_local"], plan.LOCAL_SPLITS[_SECOND]
    assert first.plan_seed == plan.PLAN_SEED and first.slot_prefix == "sl"
    assert dict(first.language_targets) == plan.LANGUAGE_TARGETS
    assert second.plan_seed != first.plan_seed and second.slot_prefix == "sl2"
    assert "shift_local" in second.prior  # it must not near-duplicate the first batch
    assert first.prior == ()


def test_shift_local_plan_is_unchanged():
    briefs = plan.load_briefs(_BRIEFS)
    assert _fingerprint(plan.build_plan(briefs)) == _SHIFT_LOCAL_PLAN_SHA256
    assert _fingerprint(plan.plan_for("shift_local", briefs)) == _SHIFT_LOCAL_PLAN_SHA256


def test_shift_local_attempt_seeds_still_come_from_the_plan():
    """The recorded D65 attempts (local-only) must be reproducible from today's plan."""
    attempts_path = _REPO / "data" / "raw" / "shift_local" / "meta" / "attempts.jsonl"
    if not attempts_path.exists():
        pytest.skip("shift_local raw attempts are local-only")
    slots = {s.slot_id: s for s in plan.plan_for("shift_local", plan.load_briefs(_BRIEFS))}
    for line in attempts_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            assert slots[row["slot"]].attempt_seed(row["attempt"]) == row["seed"], row["slot"]


def test_second_batch_is_kk_ru_only_with_the_english_slots_moved_to_kazakh():
    first = plan.LOCAL_SPLITS["shift_local"].language_targets
    targets = plan.LOCAL_SPLITS[_SECOND].language_targets
    assert targets["en"] == 0 and targets["mixed"] == 0  # D70; mixed as D65 (the model never code-switches)
    assert targets["kk"] == first["kk"] + first["en"] and targets["ru"] == first["ru"]
    assert sum(targets.values()) == sum(first.values())  # ~the same size


def test_second_batch_plan_is_the_same_design_with_new_slots():
    briefs = plan.load_briefs(_BRIEFS)
    one = plan.plan_for("shift_local", briefs)
    two = plan.plan_for(_SECOND, briefs)
    assert len(one) == len(two)
    for language, target in plan.LOCAL_SPLITS[_SECOND].language_targets.items():
        assert sum(s.language == language for s in two) == target
        assert sum(s.language == language and s.brief.label == "scam" for s in two) == target // 2
    assert not any(s.language == "en" for s in two)
    assert all(s.generator == plan.LANGUAGE_GENERATOR[s.language] for s in two)
    assert not {s.slot_id for s in one} & {s.slot_id for s in two}
    assert all(s.slot_id.startswith("sl2_") for s in two)
    assert len({s.slot_id for s in two}) == len(two)
    assert not {s.seed for s in one} & {s.seed for s in two}
    assert plan.plan_for(_SECOND, briefs) == two  # deterministic


def test_every_brief_is_used_before_any_repeats_in_the_second_batch():
    briefs = plan.load_briefs(_BRIEFS)
    two = plan.plan_for(_SECOND, briefs)
    for language in ("kk", "ru"):
        for label in ("scam", "legit"):
            used = [s.brief.id for s in two if s.language == language and s.brief.label == label]
            pool = sum(1 for b in briefs if b.label == label)
            assert len(set(used)) == min(len(used), pool)


def test_unknown_split_is_refused():
    with pytest.raises(KeyError):
        plan.plan_for("shift_local_9", plan.load_briefs(_BRIEFS))


def test_second_batch_is_checked_against_the_first():
    assert "shift_local" in checked_splits(_SECOND)
    assert "shift" in checked_splits(_SECOND) and "train" in checked_splits(_SECOND)
    assert _SECOND not in checked_splits(_SECOND)
    assert "shift_local" not in checked_splits("shift_local")


# --- generation: the first batch is part of the duplicate pool ---------------------------------


def _raw(slot_id: str, texts: list[str], language: str = "ru") -> dict:
    return {
        "id": slot_id, "language": language, "scenario": "l_test",
        "utterances": [{"speaker": "caller" if i % 2 == 0 else "callee", "text": t} for i, t in enumerate(texts)],
        "risk": 0.02, "is_hard_negative": True, "tactic_tags": [], "trigger_phrases": [],
    }


_CALL = [
    "Добрый день, это поликлиника номер пять, напоминаю о вашем приёме в четверг утром.",
    "Да, спасибо, я помню, в девять часов, правильно?",
    "Верно, в девять, кабинет двенадцать, возьмите с собой удостоверение личности.",
]


def test_prior_accepted_texts_reads_the_first_batch_of_the_same_language(tmp_path):
    first = tmp_path / "shift_local"
    first.mkdir()
    (first / "ru.jsonl").write_text(json.dumps(_raw("sl_ru_legit_01", _CALL), ensure_ascii=False) + "\n", encoding="utf-8")
    texts = prior_accepted_texts(raw_root=tmp_path, split=_SECOND, language="ru")
    assert texts == ["\n".join(_CALL)]
    assert prior_accepted_texts(raw_root=tmp_path, split=_SECOND, language="kk") == []
    assert prior_accepted_texts(raw_root=tmp_path, split="shift_local", language="ru") == []


# --- build ------------------------------------------------------------------------------------


def _setup(tmp_path: Path, *, near_copy_in: str | None) -> tuple[Path, Path]:
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "ru.jsonl").write_text(json.dumps(_raw("sl2_ru_legit_01", _CALL), ensure_ascii=False) + "\n", encoding="utf-8")
    processed = tmp_path / "processed"
    processed.mkdir()
    near = [*_CALL[:2], _CALL[2].replace("двенадцать", "тринадцать")]  # not identical, a near-duplicate
    for split in checked_splits(_SECOND):
        rows = [parse_raw_row(_raw(f"{split}_x", near))] if split == near_copy_in else []
        (processed / f"{split}.jsonl").write_text("".join(d.model_dump_json() + "\n" for d in rows), encoding="utf-8")
    return raw, processed


def test_build_writes_shift_local_2_when_disjoint(tmp_path):
    raw, processed = _setup(tmp_path, near_copy_in=None)
    manifest = build(raw_dir=raw, processed_dir=processed, split=_SECOND)
    assert manifest["split"] == _SECOND
    assert (processed / f"{_SECOND}.jsonl").exists()
    assert "shift_local" in manifest["disjoint_from"]
    assert manifest["near_duplicate_jaccard"] == pytest.approx(0.7)
    on_disk = json.loads((processed / f"{_SECOND}.manifest.json").read_text(encoding="utf-8"))
    assert on_disk == manifest


@pytest.mark.parametrize("split", ["shift_local", "train", "test"])
def test_build_refuses_a_near_duplicate_of_any_split(tmp_path, split):
    raw, processed = _setup(tmp_path, near_copy_in=split)
    with pytest.raises(ValueError, match=f"near-duplicate.*{split}"):
        build(raw_dir=raw, processed_dir=processed, split=_SECOND)
    assert not (processed / f"{_SECOND}.jsonl").exists()


def test_provenance_records_the_second_batch_seed_and_targets(tmp_path):
    raw, processed = _setup(tmp_path, near_copy_in=None)
    manifest = build(raw_dir=raw, processed_dir=processed, split=_SECOND)
    attempts = [{"slot": "sl2_ru_legit_01", "language": "ru", "label": "legit", "reason": "accepted", "generator": "qwen35_kk"}]
    record = provenance(manifest=manifest, attempts=attempts, briefs_path=_BRIEFS, llama_cpp="v", split=_SECOND)
    assert record["split"] == _SECOND
    assert record["sampling"]["plan_seed"] == plan.LOCAL_SPLITS[_SECOND].plan_seed
    assert record["sampling"]["language_targets"]["en"] == 0
    assert record["sampling"]["temperature"] == plan.TEMPERATURE and record["sampling"]["top_p"] == plan.TOP_P
    assert record["acceptance"]["near_duplicate_jaccard_across_splits"] == pytest.approx(0.7)
    assert record["published"] is False and record["acceptance"]["classifier_used_for_selection"] is False
    assert "поликлиника" not in json.dumps(record, ensure_ascii=False)


def test_committed_shift_local_2_provenance_never_carries_text():
    path = _REPO / "data" / "provenance" / f"{_SECOND}.json"
    if not path.exists():
        pytest.skip("shift_local_2 not built yet")
    record = json.loads(path.read_text(encoding="utf-8"))
    assert record["split"] == _SECOND and record["published"] is False
    assert record["sampling"]["language_targets"]["en"] == 0
    assert "utterances" not in json.dumps(record)
    assert set(record["split_counts"]["by_language"]) <= {"kk", "ru"}


# --- never published, never committed --------------------------------------------------------


def test_shift_local_2_is_never_published():
    from qorgan.data.publish_guard import DATASET_ALLOW_PATTERNS, NEVER_PUBLISHED, PUBLISHED_SPLITS, PublishRefused, assert_publishable

    assert {f"{_SECOND}.jsonl", f"{_SECOND}.manifest.json"} <= set(NEVER_PUBLISHED)
    assert not set(NEVER_PUBLISHED) & set(PUBLISHED_SPLITS)
    assert_publishable(DATASET_ALLOW_PATTERNS)
    for patterns in ([f"{_SECOND}.jsonl"], ["shift_local_*"], ["*.manifest.json"]):
        with pytest.raises(PublishRefused, match=_SECOND):
            assert_publishable(patterns)


def test_shift_local_2_raw_and_processed_files_are_gitignored():
    import subprocess

    for path in (f"data/raw/{_SECOND}/kk.jsonl", f"data/raw/{_SECOND}/meta/attempts.jsonl",
                 f"data/processed/{_SECOND}.jsonl", f"data/processed/{_SECOND}.manifest.json"):
        result = subprocess.run(["git", "check-ignore", "-q", path], cwd=_REPO, check=False)
        assert result.returncode == 0, f"{path} must be gitignored"
