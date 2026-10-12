"""Run progress from disk alone (ADR D63 phase 2): a generation run must survive a usage window
running out, so its state is never held in memory or in an agent's head. For every chunk spec
the state is one of:

  missing     no output file yet
  incomplete  an output file with fewer parseable rows than the spec has cells
  written     complete, not yet QC'd (or QC'd before the output last changed)
  qc_pass     the chunk QC report (<run>/qc/<chunk>.json) accepts it
  qc_fail     the chunk QC report rejects it (rewrite it)
"""

from __future__ import annotations

import json
from pathlib import Path

STATES = ("missing", "incomplete", "written", "qc_pass", "qc_fail")


def _row_count(path: Path) -> int:
    count = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                json.loads(line)
                count += 1
            except json.JSONDecodeError:
                continue
    return count


def chunk_state(run_dir: Path, chunk: str) -> dict:
    spec = json.loads((run_dir / "chunks" / f"{chunk}.spec.json").read_text(encoding="utf-8"))
    n_cells = len(spec["cells"])
    out = run_dir / "out" / f"{chunk}.jsonl"
    if not out.exists():
        return {"chunk": chunk, "state": "missing", "cells": n_cells, "rows": 0}
    rows = _row_count(out)
    if rows < n_cells:
        return {"chunk": chunk, "state": "incomplete", "cells": n_cells, "rows": rows}
    qc = run_dir / "qc" / f"{chunk}.json"
    if not qc.exists() or qc.stat().st_mtime < out.stat().st_mtime:
        return {"chunk": chunk, "state": "written", "cells": n_cells, "rows": rows}
    acceptance = json.loads(qc.read_text(encoding="utf-8")).get("acceptance", {})
    state = "qc_pass" if acceptance.get("verdict") == "pass" else "qc_fail"
    return {"chunk": chunk, "state": state, "cells": n_cells, "rows": rows, "accepted": acceptance.get("accepted", 0)}


def run_status(run_dir: Path) -> dict:
    chunks = sorted(p.name.removesuffix(".spec.json") for p in (run_dir / "chunks").glob("*.spec.json"))
    states = [chunk_state(run_dir, c) for c in chunks]
    by_state = {s: [c["chunk"] for c in states if c["state"] == s] for s in STATES}
    return {
        "run": run_dir.name,
        "chunks": states,
        "by_state": by_state,
        "summary": {
            "chunks": len(states),
            **{s: len(v) for s, v in by_state.items()},
            "cells": sum(c["cells"] for c in states),
            "accepted_rows": sum(c.get("accepted", 0) for c in states if c["state"] == "qc_pass"),
        },
    }
