"""Fold QC-accepted generated rows into the corpus (ADR D63 phase 2).

Only rows a QC report accepted are used: the run-level `qc_report.json` when it exists (it
also catches near-duplicates ACROSS chunks), otherwise the union of the per-chunk reports.
Generated rows go to `train` or `val` by a seeded id hash and NEVER to `test`: the held-out for
a model trained on Claude rows is built separately, from other generators (ADR D63 §3.2).
Rows a person rejected on review (`REVIEW_REJECTS`, committed, with a reason) are dropped too.
Each run has a writer (`RUN_WRITERS`, committed; default Claude). Every grid writer labels the
v2 space, so only the legacy v1 corpus (`gemini`) is masked out of the new heads.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path

import yaml

from qorgan.data.gen.qc import _schema_gate, load_run_rows
from qorgan.data.schema import Dialogue

GENERATOR_CLAUDE = "claude"
GENERATOR_GEMINI = "gemini"  # the legacy v1 corpus (generate.py), labelled for the v1 space only
GENERATOR_GEMINI_GRID = "gemini_grid"  # Gemini writing grid chunk specs (D63 v2.2): v2 labels
V2_LABELLED_GENERATORS: tuple[str, ...] = (GENERATOR_CLAUDE, GENERATOR_GEMINI_GRID)
_HASH_BUCKETS = 1_000_000
REVIEW_REJECTS = Path("configs") / "gen_review_rejects.yaml"
RUN_WRITERS = Path("configs") / "gen_run_writers.yaml"


class FoldError(RuntimeError):
    """A run cannot be folded in (no QC report yet)."""


def accepted_ids(run_dir: Path) -> frozenset[str]:
    run_report = run_dir / "qc_report.json"
    if run_report.exists():
        report = json.loads(run_report.read_text(encoding="utf-8"))
        if "acceptance" in report:
            return frozenset(report["acceptance"]["accepted_ids"])
    chunk_reports = sorted((run_dir / "qc").glob("*.json")) if (run_dir / "qc").is_dir() else []
    if not chunk_reports:
        raise FoldError(f"{run_dir} has no QC report; run `python -m qorgan.data.gen qc --run {run_dir.name}` first")
    ids: set[str] = set()
    for path in chunk_reports:
        ids |= set(json.loads(path.read_text(encoding="utf-8"))["acceptance"]["accepted_ids"])
    return frozenset(ids)


def load_review_rejects(path: Path) -> frozenset[str]:
    """Ids rejected on human review; every entry needs an `id` and a `reason` (no file = none)."""
    if not path.exists():
        return frozenset()
    entries = (yaml.safe_load(path.read_text(encoding="utf-8")) or {}).get("rejects") or []
    for entry in entries:
        if not isinstance(entry, dict) or not entry.get("id") or not entry.get("reason"):
            raise ValueError(f"{path}: every reject needs an id and a reason, got {entry!r}")
    return frozenset(str(entry["id"]) for entry in entries)


def load_run_writers(path: Path) -> dict[str, str]:
    """Run -> writer for runs NOT written by Claude (no file = every run is Claude's). Only grid
    writers are allowed: the legacy `gemini` label means "v1 labels" and would mask the run."""
    if not path.exists():
        return {}
    writers = (yaml.safe_load(path.read_text(encoding="utf-8")) or {}).get("writers") or {}
    for run, writer in writers.items():
        if writer not in V2_LABELLED_GENERATORS:
            raise ValueError(f"{path}: run {run!r} has writer {writer!r}; expected one of {V2_LABELLED_GENERATORS}")
    return {str(run): str(writer) for run, writer in writers.items()}


def run_writer(run: str, writers: dict[str, str]) -> str:
    return writers.get(run, GENERATOR_CLAUDE)


def accepted_dialogues(run_dir: Path, *, rejected: frozenset[str] = frozenset()) -> list[Dialogue]:
    """The run's QC-accepted rows as corpus dialogues, in manifest order, minus review rejects."""
    keep = accepted_ids(run_dir) - rejected
    cells, rows = load_run_rows(run_dir)
    checks, _ = _schema_gate(cells, rows)
    by_id = {c.dialogue.id: c.dialogue for c in checks if c.dialogue is not None and c.dialogue.id in keep}
    return [by_id[c.cell_id] for c in cells if c.cell_id in by_id]


def generated_split(dialogue_id: str, *, seed: int, val_fraction: float) -> str:
    """`train` or `val` for a generated row: seeded hash of its id (stable under reruns)."""
    digest = hashlib.sha256(f"gen:{seed}:{dialogue_id}".encode("utf-8")).hexdigest()
    return "val" if (int(digest, 16) % _HASH_BUCKETS) / _HASH_BUCKETS < val_fraction else "train"


def split_generated(dialogues: Sequence[Dialogue], *, seed: int, val_fraction: float) -> dict[str, tuple[Dialogue, ...]]:
    out: dict[str, list[Dialogue]] = {"train": [], "val": []}
    for dialogue in dialogues:
        out[generated_split(dialogue.id, seed=seed, val_fraction=val_fraction)].append(dialogue)
    return {k: tuple(v) for k, v in out.items()}
