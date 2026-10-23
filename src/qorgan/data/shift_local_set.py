"""`shift_local` / `shift_local_2`: independent held-outs written by local open-weights models (ADR D65).

`shift` (now documented as `shift_v1`) was written by Claude, which is becoming a training
generator, so it stops being independent once Claude rows enter train. `shift_local` is
written offline on the Mac by models from other families (see
`shift_local_plan.GENERATORS`) that never saw the corpus, the prompts or the lexicons.

    PYTHONPATH=src python scripts/gen_shift_local.py --generator <key>   # data/raw/shift_local/*.jsonl
    python -m qorgan.data.shift_local_set   # -> data/processed/shift_local.jsonl + manifest,
                                            #    data/provenance/shift_local.json (committed, no text)
    PYTHONPATH=src python scripts/gen_shift_local.py --generator qwen35_kk --split shift_local_2
    python -m qorgan.data.shift_local_set --split shift_local_2   # the second batch (PLAN A3)

A later batch (`shift_local_plan.LOCAL_SPLITS`) is also proven free of near-duplicates
(char 5-gram Jaccard, the generation check's threshold) of every split incl. the earlier batches.

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
from qorgan.data.build_corpus import normalize_for_dedup, scrub_dialogue
from qorgan.data.publish_guard import find_unscrubbed
from qorgan.data.schema import Dialogue
from qorgan.data.shift_set import EVAL_SPLITS_CHECKED, build_shift_split, load_raw_shift_rows

SPLIT_NAME = "shift_local"
RAW_SUBDIR = Path("raw") / SPLIT_NAME
PROVENANCE_PATH = Path("provenance") / f"{SPLIT_NAME}.json"
GENERATOR_NOTE = plan.LOCAL_SPLITS[SPLIT_NAME].generator_note


def checked_splits(split: str) -> tuple[str, ...]:
    """Every split a local batch must be disjoint from: the evaluation splits, `shift`, and the
    earlier local batches (`KeyError` for an unknown split)."""
    return (*EVAL_SPLITS_CHECKED, "shift", *plan.LOCAL_SPLITS[split].prior)


CHECKED_SPLITS: tuple[str, ...] = checked_splits(SPLIT_NAME)


def raw_subdir(split: str) -> Path:
    return Path("raw") / plan.LOCAL_SPLITS[split].name


def prior_accepted_texts(*, raw_root: Path, split: str, language: str) -> list[str]:
    """Transcripts the earlier batches accepted in `language` (their raw files under
    `raw_root/<prior>/`), so a later batch's duplicate check rejects a near-copy at generation
    time instead of failing the build. Read by code only, never shown."""
    texts: list[str] = []
    for prior in plan.LOCAL_SPLITS[split].prior:
        path = raw_root / prior / f"{language}.jsonl"
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                texts.append("\n".join(u["text"] for u in json.loads(line)["utterances"]))
    return texts


class NotAFixedPoint(ValueError):
    """The built split is not a scrub fixed point (ADR D45)."""


def _split_grams(processed_dir: Path, split: str) -> list[tuple[str, set[str]]]:
    path = processed_dir / f"{split}.jsonl"
    if not path.exists():
        return []
    return [
        (d.id, _grams(normalize_for_dedup(d.transcript())))
        for d in (Dialogue.model_validate_json(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip())
    ]


def _grams(text: str, n: int = checks.NEAR_DUP_N) -> set[str]:
    return {text[i : i + n] for i in range(len(text) - n + 1)}


def _jaccard(a: set[str], b: set[str]) -> float:
    return len(a & b) / len(a | b) if a and b else 0.0


def near_duplicates(raw_dir: Path, processed_dir: Path, checked: tuple[str, ...], threshold: float) -> dict[str, list[str]]:
    """`{split: [new ids]}` for raw rows (scrubbed as the build will) whose char 5-gram Jaccard
    with any row of `split` is >= `threshold` -- the generation check's measure, across splits."""
    own = [(d.id, _grams(normalize_for_dedup(scrub_dialogue(d).transcript()))) for d in load_raw_shift_rows(raw_dir)]
    found: dict[str, list[str]] = {}
    for split in checked:
        others = _split_grams(processed_dir, split)
        hits = [new_id for new_id, grams in own if any(_jaccard(grams, other) >= threshold for _, other in others)]
        if hits:
            found[split] = hits
    return found


def build(*, raw_dir: Path, processed_dir: Path, split: str = SPLIT_NAME) -> dict:
    """Build the split with the `shift` pipeline (clean, scrub, dedup, disjointness), then
    prove the written file is a scrub fixed point. A batch with a near-duplicate threshold is
    first proven free of near-duplicates of every checked split. Returns the manifest."""
    spec = plan.LOCAL_SPLITS[split]
    checked = checked_splits(split)
    if spec.near_duplicate_jaccard is not None:
        hits = near_duplicates(raw_dir, processed_dir, checked, spec.near_duplicate_jaccard)
        if hits:
            detail = "; ".join(f"near-duplicate of {other}: {ids}" for other, ids in hits.items())
            raise ValueError(f"{split} set is not disjoint -- {detail}")
    manifest = build_shift_split(
        raw_dir=raw_dir, processed_dir=processed_dir, split_name=split,
        generator=spec.generator_note, checked=checked,
    )
    findings = find_unscrubbed([processed_dir / f"{split}.jsonl"])
    if findings:
        raise NotAFixedPoint(f"{len(findings)} utterance(s) of {split} are not a scrub fixed point")
    if spec.near_duplicate_jaccard is None:
        return manifest
    manifest = {**manifest, "near_duplicate_jaccard": spec.near_duplicate_jaccard}
    (processed_dir / f"{split}.manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
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


def provenance(*, manifest: dict, attempts: list[dict], briefs_path: Path, llama_cpp: str, split: str = SPLIT_NAME) -> dict:
    """The committed record of how the split was made: models, hashes, seeds, thresholds and
    rejection statistics. Ids and counts only -- never dialogue text."""
    spec = plan.LOCAL_SPLITS[split]
    briefs = plan.load_briefs(briefs_path)
    used = sorted({row["generator"] for row in attempts})
    record = {
        "split": split,
        "adr": spec.adr,
        "generator_note": spec.generator_note,
        "llama_cpp": llama_cpp,
        "generators": {key: {**plan.GENERATORS[key], "languages": sorted(l for l, g in plan.LANGUAGE_GENERATOR.items() if g == key)} for key in used},
        "briefs": {
            "file": str(briefs_path.name), "sha256": _sha256(briefs_path),
            "count": dict(Counter(b.label for b in briefs)),
        },
        "sampling": {
            "plan_seed": spec.plan_seed, "temperature": plan.TEMPERATURE, "top_p": plan.TOP_P,
            "max_tokens": plan.MAX_TOKENS, "max_attempts": plan.MAX_ATTEMPTS,
            "attempt_seed_stride": plan.ATTEMPT_SEED_STRIDE, "turn_range": list(plan.TURN_RANGE),
            "language_targets": dict(spec.language_targets),
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
    if spec.prior or spec.near_duplicate_jaccard is not None or spec.plan_task:
        # later batches only, so the D65 record keeps its exact shape
        record["acceptance"]["near_duplicate_jaccard_across_splits"] = spec.near_duplicate_jaccard
        record["acceptance"]["duplicate_pool_includes"] = list(spec.prior)
        record["slot_prefix"] = spec.slot_prefix
        record["plan_task"] = spec.plan_task
    return record


def main() -> None:  # pragma: no cover - CLI
    import argparse
    import subprocess

    from qorgan.config import get_config

    parser = argparse.ArgumentParser(description="build a local held-out batch (ADR D65)")
    parser.add_argument("--split", default=SPLIT_NAME, choices=sorted(plan.LOCAL_SPLITS))
    split = parser.parse_args().split
    cfg = get_config()
    raw_dir = cfg.data_dir / raw_subdir(split)
    manifest = build(raw_dir=raw_dir, processed_dir=cfg.data_dir / "processed", split=split)
    attempts_path = raw_dir / "meta" / "attempts.jsonl"
    attempts = [json.loads(l) for l in attempts_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    version = subprocess.run(["llama-server", "--version"], capture_output=True, text=True, check=False)
    llama = " ".join(line for line in (version.stdout + version.stderr).splitlines() if line.startswith("version"))
    record = provenance(
        manifest=manifest, attempts=attempts,
        briefs_path=cfg.data_dir.parent / "configs" / "shift_local_briefs.yaml", llama_cpp=llama or "unknown",
        split=split,
    )
    out = cfg.data_dir / "provenance" / f"{split}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(manifest["counts"], ensure_ascii=False))
    print(f"wrote {cfg.data_dir / 'processed' / (split + '.jsonl')} and {out}")


if __name__ == "__main__":  # pragma: no cover
    main()
