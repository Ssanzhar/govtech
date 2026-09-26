"""ASR-realism evaluation (PLAN_2026-09 A10): the same dialogues scored clean and ASR-styled
(`data.asr_style`: lowercase, no punctuation, numerals spelled out, optionally Latin tokens
dropped), paired by id. FPR first, then recall, both with Clopper–Pearson intervals; decision
flips in both directions; and whether the hard-signal cues and reassurance patterns the risk
head consumes survive the styling. The styling models the recogniser's *format*, not its
misrecognitions, so these are floors on ASR damage.

Run: `python -m qorgan.eval.asr_realism --split test --split authored_heldout --split ood [--drop-latin]`
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from pathlib import Path

from qorgan.config import get_config
from qorgan.data.asr_style import asr_style_dialogue
from qorgan.data.schema import SCAM_RISK_THRESHOLD, Dialogue, ScoreResult
from qorgan.eval.intervals import binomial_interval
from qorgan.eval.run import load_split

_TRUTH_THRESHOLD = SCAM_RISK_THRESHOLD
DEFAULT_SPLITS = ("test", "authored_heldout", "ood")
ScoreFn = Callable[[str], ScoreResult]
CueHitsFn = Callable[[str], set[str]]
ReassuresFn = Callable[[str], bool]


def paired_split(
    dialogues: Sequence[Dialogue], *, score_fn: ScoreFn, alert_threshold: float,
    cue_hits: CueHitsFn, reassures: ReassuresFn, drop_latin: bool = False,
) -> dict:
    """Score every dialogue clean and styled; FPR / recall on both sides, flips, cue and
    reassurance survival (styled hits that were also hits on the clean text)."""
    rows, skipped = [], []
    for clean in dialogues:
        try:
            styled = asr_style_dialogue(clean, drop_latin=drop_latin)
        except ValueError:  # nothing left to recognise (e.g. Latin-only text under drop_latin)
            skipped.append(clean.id)
            continue
        clean_text, styled_text = clean.transcript(), styled.transcript()
        rows.append({
            "positive": clean.label.risk >= _TRUTH_THRESHOLD,
            "clean_alert": score_fn(clean_text).risk >= alert_threshold,
            "styled_alert": score_fn(styled_text).risk >= alert_threshold,
            "clean_cues": cue_hits(clean_text),
            "styled_cues": cue_hits(styled_text),
            "clean_reassures": reassures(clean_text),
            "styled_reassures": reassures(styled_text),
            "styled_text": styled_text,
        })
    positives = [r for r in rows if r["positive"]]
    negatives = [r for r in rows if not r["positive"]]
    clean_cue_hits = sum(len(r["clean_cues"]) for r in rows)
    reassurance_clean = sum(1 for r in rows if r["clean_reassures"])
    return {
        "n_positives": len(positives),
        "n_negatives": len(negatives),
        "clean": _side(positives, negatives, "clean_alert"),
        "styled": _side(positives, negatives, "styled_alert"),
        "flips_to_clear": sum(1 for r in positives if r["clean_alert"] and not r["styled_alert"]),
        "flips_to_alert": sum(1 for r in negatives if r["styled_alert"] and not r["clean_alert"]),
        "cue_hits_clean": clean_cue_hits,
        "cue_hits_preserved": sum(len(r["clean_cues"] & r["styled_cues"]) for r in rows),
        "reassurance_clean": reassurance_clean,
        "reassurance_preserved": sum(1 for r in rows if r["clean_reassures"] and r["styled_reassures"]),
        "styled_texts": {r["styled_text"] for r in rows},
        "skipped_ids": tuple(skipped),
    }


def _side(positives: list[dict], negatives: list[dict], key: str) -> dict:
    hits = sum(1 for r in positives if r[key])
    false_alerts = sum(1 for r in negatives if r[key])
    return {
        "fpr": false_alerts / len(negatives) if negatives else 0.0,
        "fpr_interval": binomial_interval(false_alerts, len(negatives)),
        "recall": hits / len(positives) if positives else 0.0,
        "recall_interval": binomial_interval(hits, len(positives)),
    }


def render_table(results: dict[str, dict]) -> str:
    lines = [
        "| Split | Text | FPR [95% CI] | Recall [95% CI] | Flips → alert / → clear | Cues kept | Reassurance kept | N+ | N- |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for split, r in results.items():
        for side in ("clean", "styled"):
            s = r[side]
            flips = f"{r['flips_to_alert']} / {r['flips_to_clear']}" if side == "styled" else "-"
            cues = f"{r['cue_hits_preserved']}/{r['cue_hits_clean']}" if side == "styled" else f"{r['cue_hits_clean']}"
            reassurance = f"{r['reassurance_preserved']}/{r['reassurance_clean']}" if side == "styled" else f"{r['reassurance_clean']}"
            lines.append(
                f"| {split} | {side} | {s['fpr']:.3f} {_ci(s['fpr_interval'])} | {s['recall']:.3f} {_ci(s['recall_interval'])} "
                f"| {flips} | {cues} | {reassurance} | {r['n_positives']} | {r['n_negatives']} |"
            )
    skipped = [f"{split}: {len(r['skipped_ids'])} skipped (nothing left after styling): {', '.join(r['skipped_ids'])}" for split, r in results.items() if r["skipped_ids"]]
    return "\n".join(lines + ([""] + skipped if skipped else []))


def _ci(interval) -> str:
    return f"[{interval.low:.3f}, {interval.high:.3f}]" if interval is not None else "[-]"


def main(argv: Sequence[str] | None = None) -> None:  # pragma: no cover - CLI (real model)
    from qorgan.classifier import predict
    from qorgan.classifier.cue_lexicon import load_cue_lexicon
    from qorgan.classifier.features import match_cues
    from qorgan.classifier.reassurance import load_reassurance_patterns, reassures

    parser = argparse.ArgumentParser(description="Clean vs ASR-styled transcripts, paired, FPR first.")
    parser.add_argument("--split", action="append", dest="splits", default=None)
    parser.add_argument("--processed-dir", type=Path, default=None)
    parser.add_argument("--backend", default=None)
    parser.add_argument("--drop-latin", action="store_true", help="worst case: Latin tokens (SMS, CVV, Kaspi…) vanish")
    args = parser.parse_args(argv)
    cfg = get_config()
    processed = args.processed_dir or cfg.data_dir / "processed"
    lexicon, patterns = load_cue_lexicon(), load_reassurance_patterns()
    results = {
        split: paired_split(
            load_split(processed, split),
            score_fn=lambda text: predict.score(text, backend=args.backend),
            alert_threshold=cfg.risk_threshold,
            cue_hits=lambda text: {m.tactic_id for m in match_cues(text, lexicon)},
            reassures=lambda text: reassures(text, patterns),
            drop_latin=args.drop_latin,
        )
        for split in (args.splits or list(DEFAULT_SPLITS))
    }
    print(f"backend={args.backend or cfg.classifier_backend} threshold={cfg.risk_threshold} drop_latin={args.drop_latin}\n")
    print(render_table(results))


if __name__ == "__main__":
    main()
