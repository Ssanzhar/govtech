"""Generate REASSURANCE hard negatives to close a real train/reality gap.

The shipped `linear` model false-positives on real legitimate RU bank/telecom calls because
the synthetic training negatives never do what real institutions do: **proactively reassure**
the customer that they will never ask for a code/card/password. This batch adds that pattern
(a documented anti-fraud practice -- not copied from the held-out test set) so the model
learns "institutional reassurance -> legitimate".

Reuses `generate_hard_negative`'s existing `style` hook, so the whole generation/validation
path is unchanged; only the injected instruction differs. Output is scrubbed and written to
`data/synthetic/reassurance_negatives.jsonl` (provenance for data/README.md).

Run: `python scripts/augment_reassurance_negatives.py --per-cell 7`
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from qorgan.config import get_config
from qorgan.data.build_corpus import scrub_dialogue
from qorgan.data.generate import generate_hard_negative, load_corpus_config
from qorgan.data.schema import Dialogue
from qorgan.llm_tools import build_client

_MAX_ATTEMPTS = 4

# Institutional reassurance is a real anti-fraud behavior banks/telcos/gov services perform.
# Motivated generally (NOT from the real_heldout transcripts) so any gain there is genuine
# generalization, not test-set leakage.
_REASSURANCE_STYLE = (
    "CRITICAL REALISM -- fraud-safety reassurance: the legitimate caller must PROACTIVELY "
    "REASSURE the customer at least once, exactly as a security-conscious real institution "
    "does. Naturally and in varied wording, have the caller make clear that the "
    "bank/operator/service will NEVER ask for an SMS/OTP code, card number, CVV, PIN, or "
    "password over the phone; that the customer should never share such data with anyone, "
    "even someone claiming to be staff; that no code or card data is needed to resolve this "
    "matter; and that anything sensitive is handled in person at a branch/office with an ID "
    "document. The caller still NEVER asks for any of those things and never redirects money."
)

# The institutional categories where proactive reassurance is realistic.
_REASSURANCE_CATEGORIES = ("legit_bank_call", "legit_gov_service", "delivery_notification")
_LANGUAGES = ("ru", "kk", "mixed")


def _generate_one(client, category_id: str, language: str, dialogue_id: str, cfg) -> Dialogue | None:
    """Generate + scrub one reassurance negative, retrying transient errors (network,
    generation) with exponential backoff. Returns None if it never succeeds."""
    for attempt in range(1, _MAX_ATTEMPTS + 1):
        try:
            dialogue = generate_hard_negative(
                category_id, language, client=client, cfg=cfg,
                dialogue_id=dialogue_id, style=_REASSURANCE_STYLE,
            )
            return scrub_dialogue(dialogue)
        except Exception as exc:  # noqa: BLE001 - resilient batch: skip on persistent failure
            if attempt == _MAX_ATTEMPTS:
                print(f"  FAIL {dialogue_id} after {attempt} attempts: {exc}")
                return None
            time.sleep(2.0 * attempt)
    return None


def generate_reassurance_negatives(*, client, per_cell: int, out_path: Path, cfg=None) -> int:
    """Generate reassurance negatives, appending each to `out_path` as it succeeds (so a
    mid-batch network failure never loses progress). Returns the count written."""
    active_cfg = cfg or load_corpus_config()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with out_path.open("w", encoding="utf-8") as sink:
        for category_id in _REASSURANCE_CATEGORIES:
            for language in _LANGUAGES:
                for i in range(per_cell):
                    dialogue_id = f"reassure_{category_id}_{language}_{i}"
                    dialogue = _generate_one(client, category_id, language, dialogue_id, active_cfg)
                    if dialogue is None:
                        continue
                    sink.write(dialogue.model_dump_json() + "\n")
                    sink.flush()
                    written += 1
                    print(f"  ok   {dialogue_id}  ({len(dialogue.utterances)} utt)")
    return written


def main(argv=None) -> None:  # pragma: no cover - live network CLI
    parser = argparse.ArgumentParser(description="Generate reassurance hard negatives.")
    parser.add_argument("--per-cell", type=int, default=7, help="Dialogues per (category, language)")
    parser.add_argument(
        "--out", type=Path, default=get_config().data_dir / "synthetic" / "reassurance_negatives.jsonl"
    )
    args = parser.parse_args(argv)

    client = build_client(get_config().gemini_api_key)
    written = generate_reassurance_negatives(client=client, per_cell=args.per_cell, out_path=args.out)
    print(f"\nwrote {written} reassurance negatives -> {args.out}")


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
