"""CLI for the taxonomy-v2 generation pipeline (see `qorgan.data.gen`)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from qorgan.config import get_config

GEN_SUBDIR = Path("synthetic") / "gen"
DEFAULT_GRID = Path("configs") / "gen_grid.yaml"


def run_dir(run: str) -> Path:
    return get_config().data_dir / GEN_SUBDIR / run


def _context():
    from qorgan.explain.roles import get_role_lexicon
    from qorgan.explain.schemes import get_schemes
    from qorgan.taxonomy import get_taxonomy

    return get_taxonomy(), get_schemes(), get_role_lexicon()


def cmd_manifest(args) -> None:
    from qorgan.data.gen.grid import build_manifest, load_grid_config, manifest_stats, write_manifest

    taxonomy, schemes, roles = _context()
    cells = build_manifest(load_grid_config(args.grid), run=args.run, size=args.size, schemes=schemes, taxonomy=taxonomy, roles=roles)
    path = write_manifest(cells, run_dir(args.run) / "manifest.jsonl")
    print(json.dumps(manifest_stats(cells), ensure_ascii=False, indent=1))
    print(f"manifest -> {path}")


def cmd_chunks(args) -> None:
    from qorgan.data.gen.grid import read_manifest
    from qorgan.data.gen.spec import build_chunk_spec, chunk_cells, write_spec

    taxonomy, schemes, roles = _context()
    base = run_dir(args.run)
    cells = read_manifest(base / "manifest.jsonl")
    for chunk, members in chunk_cells(cells, args.chunk_size):
        spec = build_chunk_spec(members, run=args.run, chunk=chunk, taxonomy=taxonomy, schemes=schemes, roles=roles)
        digest = write_spec(spec, base / "chunks" / f"{chunk}.spec.json")
        print(json.dumps({"chunk": chunk, "n": len(members), "sha256": digest}))


def cmd_qc(args) -> None:
    from qorgan.data.gen.qc import run_qc

    base = run_dir(args.run)
    report = run_qc(base, embed=not args.no_embed, probe_seed=get_config().default_seed, chunk=args.chunk)
    out = base / "qc" / f"{args.chunk}.json" if args.chunk else base / "qc_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=1))
    if report["rejected"]:
        print(json.dumps(report["rejected"], ensure_ascii=False, indent=1))
    print(f"qc report -> {out}")


def cmd_check(args) -> None:
    """Schema + cell-agreement check of ONE chunk, for the writer to self-correct. Prints only
    its own rows' errors and warnings: no lexicon, corpus or evaluation content."""
    from qorgan.data.gen.qc import _schema_gate, load_run_rows

    cells, rows = load_run_rows(run_dir(args.run), args.chunk)
    _, schema = _schema_gate(cells, rows)
    out = {"chunk": args.chunk, "cells": len(cells), "rows": schema["rows"], "valid": schema["valid"],
           "missing_cells": schema["missing_cells"], "errors": schema["errors"], "warnings": schema["warnings"]}
    print(json.dumps(out, ensure_ascii=False, indent=1))
    raise SystemExit(0 if schema["valid"] == len(cells) and not schema["missing_cells"] else 1)


def cmd_status(args) -> None:
    """Progress of a run, derived from disk only (resumable after any interruption): per chunk,
    whether its output exists and is complete, and its QC verdict. Writes `progress.json`."""
    from qorgan.data.gen.progress import run_status

    status = run_status(run_dir(args.run))
    (run_dir(args.run) / "progress.json").write_text(json.dumps(status, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(status["summary"], ensure_ascii=False, indent=1))
    for state in ("missing", "incomplete", "written", "qc_fail"):
        if status["by_state"].get(state):
            print(f"{state}: {' '.join(status['by_state'][state])}")


def main(argv: list[str] | None = None) -> None:  # pragma: no cover - CLI
    parser = argparse.ArgumentParser(prog="python -m qorgan.data.gen")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("manifest")
    p.add_argument("--run", required=True)
    p.add_argument("--size", type=int, required=True)
    p.add_argument("--grid", type=Path, default=DEFAULT_GRID)
    p.set_defaults(func=cmd_manifest)
    p = sub.add_parser("chunks")
    p.add_argument("--run", required=True)
    p.add_argument("--chunk-size", type=int, default=20)
    p.set_defaults(func=cmd_chunks)
    p = sub.add_parser("qc")
    p.add_argument("--run", required=True)
    p.add_argument("--chunk", default=None, help="QC one chunk (report -> <run>/qc/<chunk>.json)")
    p.add_argument("--no-embed", action="store_true", help="skip the embedding gates (offline smoke run)")
    p.set_defaults(func=cmd_qc)
    p = sub.add_parser("check")
    p.add_argument("--run", required=True)
    p.add_argument("--chunk", required=True)
    p.set_defaults(func=cmd_check)
    p = sub.add_parser("status")
    p.add_argument("--run", required=True)
    p.set_defaults(func=cmd_status)
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":  # pragma: no cover
    main()
