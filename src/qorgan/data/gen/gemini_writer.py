"""Gemini as a grid writer (ADR D63 v2.2): the second writer family for generated training rows.

It reads the SAME chunk spec the Claude writer agents receive (instructions, glossary, output
schema, cells) and nothing else: no corpus, lexicon or evaluation split. One call per cell keeps
each JSON answer small. Every row is validated against its cell by `rows.check_row` (the check the
`gen check` CLI runs). An invalid answer is retried with its errors fed back, and a cell that
never validates is reported, not written. Output is `<run>/out/<chunk>.jsonl` in cell order, the
shape `gen qc` and the fold read. Resumable: cells already in the file are skipped.

    python -m qorgan.data.gen.gemini_writer --run v22_gemini [--chunk chunk_00 ...]

Gemini sends the spec (synthetic briefs, no user data) to Google. The run's writer is declared in
`configs/gen_run_writers.yaml` (`gemini_grid`), so provenance and label masking know it.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

from qorgan.data.gen.grid import read_manifest
from qorgan.data.gen.rows import check_row

DEFAULT_ATTEMPTS = 3
_MAX_OUTPUT_TOKENS = 8192
_CALL_DEADLINE_S = 240.0
_TASK = (
    "Write the dialogue for the ONE cell below, following every instruction. Return a single JSON "
    "object: one row conforming to the output schema (copy cell_id, language, kind, scheme, "
    "negative_class and role from the cell)."
)


def cell_prompt(spec: Mapping, cell: Mapping, *, errors: Sequence[str]) -> str:
    """The spec's contract plus this one cell; with `errors`, the previous attempt's errors."""
    parts = [
        "INSTRUCTIONS:\n" + "\n".join(f"- {line}" for line in spec["instructions"]),
        "GLOSSARY:\n" + json.dumps(spec["glossary"], ensure_ascii=False),
        "OUTPUT SCHEMA (one row):\n" + json.dumps(spec["output_schema"], ensure_ascii=False),
        "CELL:\n" + json.dumps(cell, ensure_ascii=False),
        _TASK,
    ]
    if errors:
        parts.append("Your previous attempt for this cell was rejected. Fix these errors:\n" + "\n".join(f"- {e}" for e in errors))
    return "\n\n".join(parts)


def _existing(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    rows = (json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    return {row["cell_id"]: row for row in rows}


def _attempt(complete: Callable[[str], dict], prompt: str, cell_id: str, cells, v2, hard) -> tuple[dict | None, tuple[str, ...]]:
    try:
        raw = complete(prompt)
    except Exception as exc:  # noqa: BLE001 - a failed call is a failed attempt, recorded with its type
        return None, (f"call failed: {type(exc).__name__}: {exc}",)
    if not isinstance(raw, dict):
        return None, ("the answer is not a JSON object",)
    if raw.get("cell_id") != cell_id:
        return None, (f"cell_id must be {cell_id!r}, got {raw.get('cell_id')!r}",)
    row = {**raw, "id": cell_id}  # the spec's cells carry no `id`; rows use the cell id (as the agents do)
    check = check_row(row, cells, v2=v2, hard_negative_ids=hard)
    return (row, ()) if check.dialogue is not None else (None, check.errors)


def _write(path: Path, order: Sequence[str], rows: Mapping[str, dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(rows[c], ensure_ascii=False) + "\n" for c in order if c in rows), encoding="utf-8")


def write_chunk(
    run_dir: Path, chunk: str, *, complete: Callable[[str], dict], v2: frozenset[str], hard_negative_ids: frozenset[str],
    attempts: int = DEFAULT_ATTEMPTS,
) -> dict:
    """Fill `<run>/out/<chunk>.jsonl`; returns counts and the errors of cells that failed."""
    spec = json.loads((run_dir / "chunks" / f"{chunk}.spec.json").read_text(encoding="utf-8"))
    wanted = [c["cell_id"] for c in spec["cells"]]
    cells = {c.cell_id: c for c in read_manifest(run_dir / "manifest.jsonl") if c.cell_id in set(wanted)}
    out_path = run_dir / "out" / f"{chunk}.jsonl"
    done = _existing(out_path)
    already = sum(1 for cell_id in wanted if cell_id in done)
    failed: dict[str, list[str]] = {}
    for spec_cell in spec["cells"]:
        cell_id = spec_cell["cell_id"]
        if cell_id in done:
            continue
        errors: tuple[str, ...] = ()
        for _ in range(attempts):
            row, errors = _attempt(complete, cell_prompt(spec, spec_cell, errors=errors), cell_id, cells, v2, hard_negative_ids)
            if row is not None:
                done = {**done, cell_id: row}
                _write(out_path, wanted, done)  # after every row: an interrupted run loses at most one cell
                break
        else:
            failed[cell_id] = list(errors)
    written = sum(1 for c in wanted if c in done) - already
    return {"chunk": chunk, "cells": len(wanted), "written": written, "already_written": already, "failed": failed}


def _gemini_complete(model: str) -> Callable[[str], dict]:  # pragma: no cover - live network
    from qorgan.config import get_config
    from qorgan.llm_tools import build_client, generate_json, thinking_budget_for

    client = build_client(get_config().gemini_api_key)
    budget = thinking_budget_for(model)

    def complete(prompt: str) -> dict:
        return generate_json(client, model=model, prompt=prompt, max_output_tokens=_MAX_OUTPUT_TOKENS,
                             thinking_budget=budget, deadline_s=_CALL_DEADLINE_S)

    return complete


def main(argv: Sequence[str] | None = None) -> None:  # pragma: no cover - CLI (live network)
    from qorgan.config import get_config
    from qorgan.data.gen.__main__ import run_dir
    from qorgan.taxonomy import get_taxonomy

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", required=True)
    parser.add_argument("--chunk", action="append", default=[], help="default: every chunk of the run")
    parser.add_argument("--attempts", type=int, default=DEFAULT_ATTEMPTS)
    args = parser.parse_args(argv)
    directory = run_dir(args.run)
    chunks = args.chunk or sorted(p.name.removesuffix(".spec.json") for p in (directory / "chunks").glob("*.spec.json"))
    taxonomy = get_taxonomy()
    v2, hard = frozenset(taxonomy.label_space("v2")), frozenset(taxonomy.hard_negative_ids())
    complete = _gemini_complete(get_config().llm_model_quality)
    for chunk in chunks:
        summary = write_chunk(directory, chunk, complete=complete, v2=v2, hard_negative_ids=hard, attempts=args.attempts)
        print(json.dumps({k: (list(v) if k == "failed" else v) for k, v in summary.items()}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
