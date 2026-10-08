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

    taxonomy, schemes, _ = _context()
    cells = build_manifest(load_grid_config(args.grid), run=args.run, size=args.size, schemes=schemes, taxonomy=taxonomy)
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

    report = run_qc(run_dir(args.run), embed=not args.no_embed, probe_seed=get_config().default_seed)
    out = run_dir(args.run) / "qc_report.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=1))
    print(f"qc report -> {out}")


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
    p.add_argument("--no-embed", action="store_true", help="skip the embedding gates (offline smoke run)")
    p.set_defaults(func=cmd_qc)
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":  # pragma: no cover
    main()
