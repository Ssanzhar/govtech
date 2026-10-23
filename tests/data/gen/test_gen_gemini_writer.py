"""The Gemini grid writer (ADR D63 v2.2): one call per cell from the same chunk spec the Claude
agents receive, every row validated against its cell, errors fed back on retry, resumable."""

from __future__ import annotations

import json

from qorgan.data.gen.gemini_writer import cell_prompt, write_chunk
from qorgan.data.gen.grid import GridCell, Persona

V2 = frozenset({"imp_bank", "otp_request", "urgency"})
HARD = frozenset({"legit_bank_call"})
PERSONA = Persona(age_band="26-40", region="Almaty", occupation="nurse")


def _cell(cell_id: str) -> GridCell:
    return GridCell(cell_id=cell_id, run="r", kind="legit", language="ru", persona=PERSONA, outcome="refuses",
                    channel="cold_call", stage="none", length_target=3, negative_class="legit_bank_call",
                    role="bank_security", hard_negative=True, confusable=False)


def _row(cell_id: str, **kw) -> dict:
    row = {"cell_id": cell_id, "language": "ru", "kind": "legit", "scheme": None, "negative_class": "legit_bank_call",
           "role": "bank_security", "utterances": [
               {"speaker": "caller", "text": "Здравствуйте, это банк, напоминаем о платеже.", "stage": "none", "tactics": []},
               {"speaker": "callee", "text": "Спасибо, оплачу в приложении.", "stage": "none", "tactics": []}]}
    row.update(kw)
    return row


def _run(tmp_path, cell_ids=("r_0000", "r_0001")):
    run = tmp_path / "r"
    (run / "chunks").mkdir(parents=True)
    cells = [_cell(i) for i in cell_ids]
    (run / "manifest.jsonl").write_text("".join(c.model_dump_json() + "\n" for c in cells), encoding="utf-8")
    spec = {"run": "r", "chunk": "chunk_00", "instructions": ["Write one dialogue per cell."], "glossary": {"x": "y"},
            "output_schema": {"type": "object"}, "cells": [json.loads(c.model_dump_json()) for c in cells]}
    (run / "chunks" / "chunk_00.spec.json").write_text(json.dumps(spec), encoding="utf-8")
    return run


def _rows(run) -> list[dict]:
    path = run / "out" / "chunk_00.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_prompt_carries_the_spec_contract_and_only_its_cell():
    spec = {"instructions": ["A", "B"], "glossary": {"g": 1}, "output_schema": {"s": 2},
            "cells": [{"cell_id": "c1"}, {"cell_id": "c2"}]}
    prompt = cell_prompt(spec, spec["cells"][1], errors=())
    assert "A" in prompt and '"g": 1' in prompt and '"s": 2' in prompt and '"c2"' in prompt
    assert '"c1"' not in prompt
    assert "previous attempt" not in prompt.lower()
    assert "unknown tactic" in cell_prompt(spec, spec["cells"][1], errors=("unknown tactic 'x'",))


def test_writes_valid_rows_in_cell_order_with_id_equal_to_cell_id(tmp_path):
    run = _run(tmp_path)
    summary = write_chunk(run, "chunk_00", complete=lambda prompt: _row("r_0001" if '"r_0001"' in prompt else "r_0000"),
                          v2=V2, hard_negative_ids=HARD)
    assert [r["id"] for r in _rows(run)] == ["r_0000", "r_0001"]
    assert all(r["id"] == r["cell_id"] for r in _rows(run))
    assert summary == {"chunk": "chunk_00", "cells": 2, "written": 2, "already_written": 0, "failed": {}}


def test_invalid_output_is_retried_with_its_errors(tmp_path):
    run = _run(tmp_path, ("r_0000",))
    prompts: list[str] = []

    def complete(prompt):
        prompts.append(prompt)
        return _row("r_0000", kind="scam") if len(prompts) == 1 else _row("r_0000")

    summary = write_chunk(run, "chunk_00", complete=complete, v2=V2, hard_negative_ids=HARD)
    assert summary["written"] == 1 and len(prompts) == 2
    assert "previous attempt" in prompts[1].lower()


def test_a_cell_that_never_validates_is_reported_not_written(tmp_path):
    run = _run(tmp_path, ("r_0000",))
    summary = write_chunk(run, "chunk_00", complete=lambda p: {"cell_id": "r_0000"}, v2=V2, hard_negative_ids=HARD, attempts=2)
    assert summary["written"] == 0 and list(summary["failed"]) == ["r_0000"]
    assert not (run / "out" / "chunk_00.jsonl").exists() or _rows(run) == []


def test_a_call_that_raises_counts_as_a_failed_attempt(tmp_path):
    run = _run(tmp_path, ("r_0000",))
    calls = iter([RuntimeError("timeout"), _row("r_0000")])

    def complete(prompt):
        result = next(calls)
        if isinstance(result, Exception):
            raise result
        return result

    assert write_chunk(run, "chunk_00", complete=complete, v2=V2, hard_negative_ids=HARD)["written"] == 1


def test_resumes_without_rewriting_finished_cells(tmp_path):
    run = _run(tmp_path)
    write_chunk(run, "chunk_00", complete=lambda p: _row("r_0000"), v2=V2, hard_negative_ids=HARD, attempts=1)
    seen: list[str] = []

    def complete(prompt):
        seen.append(prompt)
        return _row("r_0001")

    summary = write_chunk(run, "chunk_00", complete=complete, v2=V2, hard_negative_ids=HARD)
    assert summary["already_written"] == 1 and summary["written"] == 1 and len(seen) == 1
    assert [r["id"] for r in _rows(run)] == ["r_0000", "r_0001"]
