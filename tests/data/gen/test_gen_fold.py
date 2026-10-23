"""Folding QC-accepted generated rows into the corpus (ADR D63 phase 2)."""

from __future__ import annotations

import json

import pytest

from qorgan.data.gen.fold import FoldError, accepted_ids, generated_split


def test_generated_split_is_seeded_stable_and_never_test():
    ids = [f"p2_{i:04d}" for i in range(2000)]
    first = [generated_split(i, seed=42, val_fraction=0.15) for i in ids]
    assert first == [generated_split(i, seed=42, val_fraction=0.15) for i in ids]
    assert set(first) == {"train", "val"}
    assert 0.12 < first.count("val") / len(ids) < 0.18


def test_accepted_ids_prefer_the_run_report_then_chunk_reports(tmp_path):
    run = tmp_path / "r"
    (run / "qc").mkdir(parents=True)
    with pytest.raises(FoldError):
        accepted_ids(tmp_path / "missing")
    (run / "qc" / "chunk_00.json").write_text(json.dumps({"acceptance": {"accepted_ids": ["a", "b"]}}))
    (run / "qc" / "chunk_01.json").write_text(json.dumps({"acceptance": {"accepted_ids": ["c"]}}))
    assert accepted_ids(run) == {"a", "b", "c"}
    (run / "qc_report.json").write_text(json.dumps({"acceptance": {"accepted_ids": ["a"]}}))
    assert accepted_ids(run) == {"a"}


def test_review_rejects_load_and_validate(tmp_path):
    from qorgan.data.gen.fold import load_review_rejects

    assert load_review_rejects(tmp_path / "missing.yaml") == frozenset()
    path = tmp_path / "rejects.yaml"
    path.write_text("rejects:\n  - id: r_0001\n    reason: borderline\n    decided: 2026-10-23\n", encoding="utf-8")
    assert load_review_rejects(path) == {"r_0001"}
    path.write_text("rejects:\n  - id: r_0001\n", encoding="utf-8")  # a reject must say why
    with pytest.raises(ValueError):
        load_review_rejects(path)


def test_accepted_dialogues_drop_review_rejects(monkeypatch, tmp_path):
    from qorgan.data.gen import fold

    monkeypatch.setattr(fold, "accepted_ids", lambda run_dir: frozenset({"a", "b"}))
    monkeypatch.setattr(fold, "load_run_rows", lambda run_dir: ([type("C", (), {"cell_id": i}) for i in "ab"], None))
    dialogues = {i: type("D", (), {"id": i})() for i in "ab"}
    checks = [type("K", (), {"dialogue": d})() for d in dialogues.values()]
    monkeypatch.setattr(fold, "_schema_gate", lambda cells, rows: (checks, None))
    assert [d.id for d in fold.accepted_dialogues(tmp_path)] == ["a", "b"]
    assert [d.id for d in fold.accepted_dialogues(tmp_path, rejected=frozenset({"b"}))] == ["a"]


def test_committed_review_rejects_drop_the_borderline_family_row():
    """User decision 2026-10-23 (D63 v2.1 A4): v21_legit_0100 opens like the new-number scam."""
    from qorgan.config import get_config
    from qorgan.data.gen.fold import REVIEW_REJECTS, load_review_rejects

    assert "v21_legit_0100" in load_review_rejects(get_config().repo_root / REVIEW_REJECTS)


def test_run_writers_default_to_claude_and_validate(tmp_path):
    from qorgan.data.gen.fold import GENERATOR_CLAUDE, GENERATOR_GEMINI_GRID, load_run_writers, run_writer

    assert load_run_writers(tmp_path / "missing.yaml") == {}
    path = tmp_path / "writers.yaml"
    path.write_text("writers:\n  v22_gemini: gemini_grid\n", encoding="utf-8")
    writers = load_run_writers(path)
    assert run_writer("v22_gemini", writers) == GENERATOR_GEMINI_GRID
    assert run_writer("p2", writers) == GENERATOR_CLAUDE
    path.write_text("writers:\n  v22_x: gemini\n", encoding="utf-8")  # legacy v1 corpus label: not a grid writer
    with pytest.raises(ValueError):
        load_run_writers(path)


def test_provenance_records_each_generated_rows_writer():
    from qorgan.data.build_corpus import build_provenance
    from qorgan.data.schema import Dialogue, Label, Utterance

    def row(i, scam):
        return Dialogue(id=i, language="kk", utterances=(Utterance(speaker="caller", text="сәлем"),),
                        label=Label(risk=0.9 if scam else 0.02))

    splits = {"train": [row("old_1", True), row("g_1", True), row("g_1-asr", True), row("c_1", False)], "val": []}
    prov = build_provenance(splits, {"g_1", "c_1"}, "v2", writers={"g_1": "gemini_grid"})
    assert prov["generator"] == {"old_1": "gemini", "g_1": "gemini_grid", "g_1-asr": "gemini_grid", "c_1": "claude"}
    assert prov["balance"]["train"]["gemini_grid"] == {"rows": 2, "scam": 2, "p_scam": 1.0}


def test_committed_run_writers_name_only_grid_writers():
    from qorgan.config import get_config
    from qorgan.data.gen.fold import RUN_WRITERS, V2_LABELLED_GENERATORS, load_run_writers

    assert set(load_run_writers(get_config().repo_root / RUN_WRITERS).values()) <= set(V2_LABELLED_GENERATORS)
