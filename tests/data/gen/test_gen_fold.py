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
