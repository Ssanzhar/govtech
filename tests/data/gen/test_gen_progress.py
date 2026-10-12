"""Phase-2 generation bookkeeping (ADR D63): progress derived from disk, row/chunk verdicts."""

from __future__ import annotations

import json
import os

from qorgan.data.gen.progress import run_status
from qorgan.data.gen.qc import chunk_verdict, row_verdicts


def _spec(run, chunk, n):
    (run / "chunks").mkdir(parents=True, exist_ok=True)
    (run / "chunks" / f"{chunk}.spec.json").write_text(json.dumps({"cells": [{"cell_id": f"{chunk}-{i}"} for i in range(n)]}))


def test_status_walks_every_state(tmp_path):
    run = tmp_path / "r"
    for chunk in ("chunk_00", "chunk_01", "chunk_02", "chunk_03", "chunk_04"):
        _spec(run, chunk, 2)
    (run / "out").mkdir()
    (run / "qc").mkdir()
    (run / "out" / "chunk_01.jsonl").write_text('{"a": 1}\nnot json\n')
    for chunk in ("chunk_02", "chunk_03", "chunk_04"):
        (run / "out" / f"{chunk}.jsonl").write_text('{"a": 1}\n{"a": 2}\n')
    (run / "qc" / "chunk_03.json").write_text(json.dumps({"acceptance": {"verdict": "pass", "accepted": 2}}))
    (run / "qc" / "chunk_04.json").write_text(json.dumps({"acceptance": {"verdict": "fail", "accepted": 1}}))
    for chunk in ("chunk_03", "chunk_04"):  # QC newer than the output
        out = run / "out" / f"{chunk}.jsonl"
        os.utime(out, (1, 1))
    status = run_status(run)
    assert status["by_state"] == {"missing": ["chunk_00"], "incomplete": ["chunk_01"], "written": ["chunk_02"],
                                  "qc_pass": ["chunk_03"], "qc_fail": ["chunk_04"]}
    assert status["summary"]["accepted_rows"] == 2


def test_stale_qc_means_written_again(tmp_path):
    run = tmp_path / "r"
    _spec(run, "chunk_00", 1)
    (run / "out").mkdir()
    (run / "qc").mkdir()
    (run / "qc" / "chunk_00.json").write_text(json.dumps({"acceptance": {"verdict": "pass", "accepted": 1}}))
    os.utime(run / "qc" / "chunk_00.json", (1, 1))
    (run / "out" / "chunk_00.jsonl").write_text('{"a": 1}\n')
    assert run_status(run)["by_state"]["written"] == ["chunk_00"]


def test_row_verdicts_collect_every_reason_and_chunk_verdict_applies_the_floor():
    report = {
        "schema": {"errors": {"x": ["bad"]}},
        "scrub": {"rows_not_fixed_point": ["a"]},
        "near_dup": {"flagged_vs_train": ["b"], "flagged_within_run": []},
        "novelty_rouge_l": {"flagged": ["b"]},
        "disjoint": {"overlaps": {"test": ["c"]}},
    }
    rejected = row_verdicts(report, ["a", "b", "c", "d", "e"])
    assert set(rejected) == {"x", "b", "c"} and len(rejected["b"]) == 2  # "a" is scrubbed at fold, not dropped
    verdict = chunk_verdict(6, rejected, ["a", "b", "c", "d", "e"], 0.3)
    assert verdict["accepted_ids"] == ["a", "d", "e"] and verdict["verdict"] == "pass"
    assert chunk_verdict(6, rejected, ["a", "b", "c", "d", "e"], 0.6)["verdict"] == "fail"
