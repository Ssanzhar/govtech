"""Seed ~500 Level-2 incidents into data/processed/incidents.jsonl (D5-1 / gap G5).

Sources scam transcripts from the built corpus, groups them into script families with reused
phone numbers, injects a novel scheme, and writes a deterministic incident stream over the
last ~30 days. Run: `python scripts/demo_seed.py`.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from datetime import datetime, timedelta
from pathlib import Path

from qorgan.config import get_config
from qorgan.data.generate import load_corpus_config
from qorgan.data.incident_seed import seed_incidents, write_incidents_jsonl
from qorgan.data.schema import Dialogue


def main(argv: Sequence[str] | None = None) -> None:
    cfg = get_config()
    parser = argparse.ArgumentParser(description="Seed Level-2 incidents from the corpus.")
    parser.add_argument("--count", type=int, default=500)
    parser.add_argument("--seed", type=int, default=cfg.default_seed)
    parser.add_argument("--span-days", type=float, default=30.0)
    parser.add_argument("--out", type=Path, default=cfg.data_dir / "processed" / "incidents.jsonl")
    args = parser.parse_args(argv)

    dialogues_path = load_corpus_config().output_path
    dialogues = [
        Dialogue.model_validate_json(line)
        for line in dialogues_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    start = datetime.now() - timedelta(days=args.span_days)
    incidents = seed_incidents(
        dialogues, count=args.count, seed=args.seed, start_time=start, span_days=args.span_days
    )
    write_incidents_jsonl(incidents, args.out)
    print(f"seeded {len(incidents)} incidents -> {args.out}")


if __name__ == "__main__":
    main()
