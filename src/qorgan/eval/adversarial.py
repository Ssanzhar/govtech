"""Paired adversarial recall (PLAN_2026-09 A9): the same scams before and after the
lexicon-free paraphrase, scored with the configured backend, recall with Clopper–Pearson
intervals, per language, and the plan's gate -- a drop of more than 15 points promotes the
ASR-realism / training work (A10) to a must.

Run: `QORGAN_CLASSIFIER_BACKEND=linear python -m qorgan.eval.adversarial`
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from pathlib import Path

from qorgan.config import get_config
from qorgan.data.adversarial import ADVERSARIAL_ID_PREFIX, ADVERSARIAL_SPLIT, source_positives
from qorgan.data.schema import Dialogue, ScoreResult
from qorgan.eval.intervals import binomial_interval
from qorgan.eval.run import load_split

RECALL_DROP_GATE_POINTS = 15.0
SOURCE_SPLITS = ("test", "ood")
ScoreFn = Callable[[str], ScoreResult]


def paired_recall(
    sources: Sequence[Dialogue], adversarial: Sequence[Dialogue], *, score_fn: ScoreFn, alert_threshold: float
) -> dict:
    """Score each (source, paraphrase) pair once; recall on both sides, flips, per language."""
    by_source_id = {d.id[len(ADVERSARIAL_ID_PREFIX):]: d for d in adversarial if d.id.startswith(ADVERSARIAL_ID_PREFIX)}
    pairs = [(s, by_source_id[s.id]) for s in sources if s.id in by_source_id]
    flagged = [
        (score_fn(s.transcript()).risk >= alert_threshold, score_fn(a.transcript()).risk >= alert_threshold, s.language)
        for s, a in pairs
    ]
    result = _summary(flagged)
    result["by_language"] = {
        language: _summary([f for f in flagged if f[2] == language])
        for language in sorted({f[2] for f in flagged})
    }
    return result


def _summary(flagged: list[tuple[bool, bool, str]]) -> dict:
    n = len(flagged)
    src_hits = sum(1 for s, _, _ in flagged if s)
    adv_hits = sum(1 for _, a, _ in flagged if a)
    src_recall = src_hits / n if n else 0.0
    adv_recall = adv_hits / n if n else 0.0
    drop = round(100.0 * (src_recall - adv_recall), 1)
    return {
        "n_pairs": n,
        "source": {"recall": src_recall, "interval": binomial_interval(src_hits, n)},
        "adversarial": {"recall": adv_recall, "interval": binomial_interval(adv_hits, n)},
        "recall_drop_points": drop,
        "flips_to_clear": sum(1 for s, a, _ in flagged if s and not a),
        "flips_to_scam": sum(1 for s, a, _ in flagged if a and not s),
        "gate_failed": drop > RECALL_DROP_GATE_POINTS,
    }


_SPLIT_LABELS = {
    ADVERSARIAL_SPLIT: "adversarial (lexicon-free paraphrases)",
    "adversarial_legit": "adversarial_legit (lexicon-free + legit-sounding register)",
}


def render_table(result: dict, *, split_name: str = ADVERSARIAL_SPLIT) -> str:
    label = _SPLIT_LABELS.get(split_name, split_name)
    lines = ["| Set | N | Recall [95% CI] |", "|---|---|---|"]
    lines.append(_row("source scams (test + ood)", result["n_pairs"], result["source"]))
    lines.append(_row(label, result["n_pairs"], result["adversarial"]))
    for language, row in result["by_language"].items():
        lines.append(_row(f"&nbsp;&nbsp;{language} · source", row["n_pairs"], row["source"]))
        lines.append(_row(f"&nbsp;&nbsp;{language} · {split_name}", row["n_pairs"], row["adversarial"]))
    verdict = (
        f"FAILS the {RECALL_DROP_GATE_POINTS:.0f}-point gate -- PLAN A10 becomes a must"
        if result["gate_failed"]
        else f"within the {RECALL_DROP_GATE_POINTS:.0f}-point gate"
    )
    lines.append(
        f"\nRecall drop: **{result['recall_drop_points']:.1f} points** "
        f"({result['flips_to_clear']} scams flip to clear, {result['flips_to_scam']} flip to scam) -- {verdict}."
    )
    return "\n".join(lines)


def _row(name: str, n: int, side: dict) -> str:
    interval = side["interval"]
    ci = f"[{interval.low:.3f}, {interval.high:.3f}]" if interval is not None else "-"
    return f"| {name} | {n} | {side['recall']:.3f} {ci} |"


def main(argv: Sequence[str] | None = None) -> None:  # pragma: no cover - CLI (real model)
    from qorgan.classifier import predict

    parser = argparse.ArgumentParser(description="Paired recall on lexicon-free paraphrases.")
    parser.add_argument("--processed-dir", type=Path, default=get_config().data_dir / "processed")
    parser.add_argument("--backend", default=None)
    parser.add_argument("--split", default=ADVERSARIAL_SPLIT, help="adversarial (A9) or adversarial_legit (A9b)")
    args = parser.parse_args(argv)
    cfg = get_config()
    sources = [d for name in SOURCE_SPLITS for d in source_positives(load_split(args.processed_dir, name))]
    adversarial = load_split(args.processed_dir, args.split)
    result = paired_recall(
        sources, adversarial,
        score_fn=lambda text: predict.score(text, backend=args.backend),
        alert_threshold=cfg.risk_threshold,
    )
    print(f"backend={args.backend or cfg.classifier_backend} threshold={cfg.risk_threshold} split={args.split}")
    print(render_table(result, split_name=args.split))


if __name__ == "__main__":
    main()
