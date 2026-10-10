"""`shift_local`: the independent held-out split written by local open-weights models (ADR D65).

`shift` (now documented as `shift_v1`) was written by Claude, which is becoming a training
generator, so it stops being independent once Claude rows enter train. `shift_local` is
written offline on the Mac by models from other families (see
`shift_local_plan.GENERATORS`) that never saw the corpus, the prompts or the lexicons.

    PYTHONPATH=src python scripts/gen_shift_local.py --generator <key>   # data/raw/shift_local/*.jsonl
    python -m qorgan.data.shift_local_set   # -> data/processed/shift_local.jsonl + manifest,
                                            #    data/provenance/shift_local.json (committed, no text)

Rules: evaluation only. Nothing is tuned on it, the lexicon is never extended from it, and it
never leaves the machine (`publish_guard.NEVER_PUBLISHED`; data/raw and data/processed are
gitignored). Any row read by a person goes into the inspection ledger.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

from qorgan.data import shift_local_checks as checks
from qorgan.data import shift_local_plan as plan
from qorgan.data.publish_guard import find_unscrubbed
from qorgan.data.shift_set import EVAL_SPLITS_CHECKED, build_shift_split

SPLIT_NAME = "shift_local"
RAW_SUBDIR = Path("raw") / SPLIT_NAME
PROVENANCE_PATH = Path("provenance") / f"{SPLIT_NAME}.json"
CHECKED_SPLITS: tuple[str, ...] = (*EVAL_SPLITS_CHECKED, "shift")
GENERATOR_NOTE = (
    "generated 2026-10-09 offline with llama.cpp by local open-weights models from non-Gemini, non-Claude "
    "families (see data/provenance/shift_local.json); briefs from public scam descriptions; no access to the "
    "corpus, prompts or lexicons; accepted by automated checks only"
)


class NotAFixedPoint(ValueError):
    """The built split is not a scrub fixed point (ADR D45)."""


def build(*, raw_dir: Path, processed_dir: Path) -> dict:
    """Build the split with the `shift` pipeline (clean, scrub, dedup, disjointness), then
    prove the written file is a scrub fixed point. Returns the manifest."""
    manifest = build_shift_split(
        raw_dir=raw_dir, processed_dir=processed_dir, split_name=SPLIT_NAME,
        generator=GENERATOR_NOTE, checked=CHECKED_SPLITS,
    )
    findings = find_unscrubbed([processed_dir / f"{SPLIT_NAME}.jsonl"])
    if findings:
        raise NotAFixedPoint(f"{len(findings)} utterance(s) of {SPLIT_NAME} are not a scrub fixed point")
    return manifest


def attempt_stats(attempts: list[dict]) -> dict:
    """Per language x label: slots, attempts, accepted, and rejections by reason."""
    stats: dict[str, dict] = {}
    for row in attempts:
        cell = stats.setdefault(f"{row['language']}/{row['label']}", {"attempts": 0, "accepted": 0, "rejected": Counter(), "slots": set()})
        cell["attempts"] += 1
        cell["slots"].add(row["slot"])
        if row["reason"] == "accepted":
            cell["accepted"] += 1
        else:
            cell["rejected"][row["reason"]] += 1
    return {
        key: {
            "slots": len(cell["slots"]), "attempts": cell["attempts"], "accepted": cell["accepted"],
            "slots_unfilled": len(cell["slots"]) - cell["accepted"],
            "rejected": dict(sorted(cell["rejected"].items())),
            "rejection_rate": round(1 - cell["accepted"] / cell["attempts"], 3) if cell["attempts"] else None,
        }
        for key, cell in sorted(stats.items())
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def provenance(*, manifest: dict, attempts: list[dict], briefs_path: Path, llama_cpp: str) -> dict:
    """The committed record of how the split was made: models, hashes, seeds, thresholds and
    rejection statistics. Ids and counts only -- never dialogue text."""
    briefs = plan.load_briefs(briefs_path)
    used = sorted({row["generator"] for row in attempts})
    return {
        "split": SPLIT_NAME,
        "adr": "D65",
        "generator_note": GENERATOR_NOTE,
        "llama_cpp": llama_cpp,
        "generators": {key: {**plan.GENERATORS[key], "languages": sorted(l for l, g in plan.LANGUAGE_GENERATOR.items() if g == key)} for key in used},
        "briefs": {
            "file": str(briefs_path.name), "sha256": _sha256(briefs_path),
            "count": dict(Counter(b.label for b in briefs)),
        },
        "sampling": {
            "plan_seed": plan.PLAN_SEED, "temperature": plan.TEMPERATURE, "top_p": plan.TOP_P,
            "max_tokens": plan.MAX_TOKENS, "max_attempts": plan.MAX_ATTEMPTS,
            "attempt_seed_stride": plan.ATTEMPT_SEED_STRIDE, "turn_range": list(plan.TURN_RANGE),
            "language_targets": plan.LANGUAGE_TARGETS,
        },
        "acceptance": {
            "reasons": list(checks.REJECT_REASONS),
            "thresholds": {
                name: getattr(checks, name) for name in (
                    "MIN_TURNS", "MAX_TURNS", "MIN_TURNS_PER_SPEAKER", "MAX_TURN_CHARS", "MAX_STRAY_LINES",
                    "ECHO_NGRAM", "MAX_REPEATS_OF_ONE_TURN", "NEAR_DUP_JACCARD", "MIN_SCRIPT_SHARE",
                    "KK_MIN_SPECIFIC_SHARE", "KK_MAX_RU_MARKER_TURNS", "RU_MAX_SPECIFIC_SHARE",
                    "MIXED_MIN_SPECIFIC_SHARE", "MIXED_MIN_KK_TURNS", "MIXED_MIN_RU_TURNS",
                )
            },
            "classifier_used_for_selection": False,
        },
        "attempts": attempt_stats(attempts),
        "split_counts": manifest["counts"],
        "content_hash": manifest["content_hash"],
        "disjoint_from": manifest["disjoint_from"],
        "scrub_fixed_point": True,
        "published": False,
    }


def main() -> None:  # pragma: no cover - CLI
    import subprocess

    from qorgan.config import get_config

    cfg = get_config()
    raw_dir = cfg.data_dir / RAW_SUBDIR
    manifest = build(raw_dir=raw_dir, processed_dir=cfg.data_dir / "processed")
    attempts_path = raw_dir / "meta" / "attempts.jsonl"
    attempts = [json.loads(l) for l in attempts_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    version = subprocess.run(["llama-server", "--version"], capture_output=True, text=True, check=False)
    llama = " ".join(line for line in (version.stdout + version.stderr).splitlines() if line.startswith("version"))
    record = provenance(
        manifest=manifest, attempts=attempts,
        briefs_path=cfg.data_dir.parent / "configs" / "shift_local_briefs.yaml", llama_cpp=llama or "unknown",
    )
    out = cfg.data_dir / PROVENANCE_PATH
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(manifest["counts"], ensure_ascii=False))
    print(f"wrote {cfg.data_dir / 'processed' / (SPLIT_NAME + '.jsonl')} and {out}")


if __name__ == "__main__":  # pragma: no cover
    main()
