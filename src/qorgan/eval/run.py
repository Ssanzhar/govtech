"""FPR-first evaluation harness (D2-6): score a corpus split with the configured classifier
and report metrics on `test` **and** `real_heldout` **separately** (CLAUDE.md SS6).

Two thresholds are in play and kept distinct:
- the *label* threshold (`_TRUTH_THRESHOLD`) turns a corpus record's labeled risk into the
  binary ground truth (is this actually a scam?);
- the *alert* threshold (`config.risk_threshold`) turns the classifier's predicted risk into
  a fired/not-fired decision -- this is what FPR/precision/recall are measured against.

The classifier is injected as `score_fn` for testability; the CLI default wires
`qorgan.classifier.predict.score` (backend selected via config), so evaluation swaps LLM
vs XLM-R vs mock with no code change.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

from qorgan.config import get_config
from qorgan.data.schema import Dialogue, ScoreResult
from qorgan.eval import metrics
from qorgan.taxonomy import get_taxonomy

# Labeled risk at/above this is treated as a scam in the ground truth. Corpus labels sit
# near 0.9 (scam) or 0.02 (legit), so the exact midpoint is unambiguous.
_TRUTH_THRESHOLD = 0.5

ScoreFn = Callable[[str], ScoreResult]


def load_split(processed_dir: Path, split_name: str) -> tuple[Dialogue, ...]:
    """Load `processed_dir/<split_name>.jsonl` into `Dialogue`s (raises if absent)."""
    path = processed_dir / f"{split_name}.jsonl"
    if not path.exists():
        raise FileNotFoundError(
            f"Split not found: {path}. Run `python -m qorgan.data.build_corpus` first."
        )
    dialogues: list[Dialogue] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            dialogues.append(Dialogue.model_validate_json(line))
    return tuple(dialogues)


def evaluate_split(
    dialogues: Sequence[Dialogue], *, score_fn: ScoreFn, alert_threshold: float
) -> dict:
    """Score every dialogue and return an FPR-first metrics dict (+ per-tactic F1)."""
    y_true: list[int] = []
    y_scores: list[float] = []
    true_tag_sets: list[set[str]] = []
    pred_tag_sets: list[set[str]] = []

    for dialogue in dialogues:
        result = score_fn(dialogue.transcript())
        y_true.append(1 if dialogue.label.risk >= _TRUTH_THRESHOLD else 0)
        y_scores.append(result.risk)
        true_tag_sets.append({tag.id for tag in dialogue.label.tactic_tags})
        pred_tag_sets.append({tag.id for tag in result.tags})

    report = metrics.binary_report(y_true, y_scores, threshold=alert_threshold)
    report["per_tactic_f1"] = metrics.per_label_f1(
        true_tag_sets, pred_tag_sets, get_taxonomy().tactic_ids()
    )
    report["n"] = len(dialogues)
    return report


def run(
    processed_dir: Path,
    split_names: Sequence[str],
    *,
    score_fn: ScoreFn | None = None,
    backend: str | None = None,
    alert_threshold: float | None = None,
) -> dict[str, dict]:
    """Evaluate each named split separately and return `{split_name: metrics}`."""
    cfg = get_config()
    active_threshold = cfg.risk_threshold if alert_threshold is None else alert_threshold
    active_score_fn = score_fn or _default_score_fn(backend)
    return {
        name: evaluate_split(
            load_split(processed_dir, name),
            score_fn=active_score_fn,
            alert_threshold=active_threshold,
        )
        for name in split_names
    }


def _default_score_fn(backend: str | None) -> ScoreFn:
    from qorgan.classifier.predict import score

    return lambda transcript: score(transcript, backend=backend)


_REPORT_COLUMNS = ("fpr", "precision", "recall", "f1", "pr_auc", "support")


def format_report(results: Mapping[str, dict]) -> str:
    """Render an FPR-first markdown table: one row per split, FPR as the first metric."""
    header = "| Split | FPR | Precision | Recall | F1 | PR-AUC | N |"
    divider = "|---|---|---|---|---|---|---|"
    rows = [header, divider]
    for split_name, metric in results.items():
        cells = [split_name] + [_fmt(metric.get(col)) for col in _REPORT_COLUMNS]
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join(rows)


def _fmt(value) -> str:
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def main(argv: Sequence[str] | None = None) -> None:
    """CLI: `python -m qorgan.eval.run --split test --split real_heldout [--backend mock]`."""
    cfg = get_config()
    parser = argparse.ArgumentParser(description="FPR-first evaluation over corpus splits.")
    parser.add_argument("--split", action="append", dest="splits", default=None, help="Split name (repeatable)")
    parser.add_argument("--backend", default=None, help="Override classifier backend (llm|xlmr|mock)")
    parser.add_argument("--processed-dir", type=Path, default=None, help="Dir holding <split>.jsonl")
    args = parser.parse_args(argv)

    split_names = args.splits or ["test", "real_heldout"]
    processed_dir = args.processed_dir or (cfg.data_dir / "processed")
    results = run(processed_dir, split_names, backend=args.backend)
    print(format_report(results))


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
