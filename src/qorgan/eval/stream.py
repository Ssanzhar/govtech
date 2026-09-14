"""Streaming (per-turn) eval harness: replay a dialogue's utterances through the live
pipeline (`qorgan.live.session`) turn by turn and report the live analogs of the
full-transcript metrics in `qorgan.eval.run`.

Two harnesses, two questions:
- `eval/run.py` -- "if I score the whole call transcript once, how good is the verdict?"
- `eval/stream.py` (this module) -- "as the call unfolds turn by turn, when (if ever) does
  the live suspicion meter latch, and does it latch on calls it shouldn't?"

`false_latch_rate` is the primary number here -- the live analog of FPR: the fraction of
negative (non-scam) dialogues whose meter ever crosses the warning latch during the call.
The ground-truth positive/negative split reuses `eval.run._TRUTH_THRESHOLD` verbatim (same
corpus labels, same threshold) rather than re-declaring it, so streaming and full-transcript
metrics are read against one single definition of "scam" (CLAUDE.md SS6: no duplicated
sources of truth). Unsupported-locale/backend inputs are not re-validated here either --
`live.session.initial_session` already rejects an unsupported locale eagerly, and
`classifier.predict.score` (invoked inside `live.session.advance`) already rejects an
unsupported backend with a clear `UnknownBackendError` on the first turn -- duplicating
that list here would just be another way for the two checks to drift apart.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from qorgan.asr.stream import CommittedUtterance
from qorgan.config import get_config
from qorgan.data.schema import Dialogue
from qorgan.eval import intervals
from qorgan.eval.run import _TRUTH_THRESHOLD, load_split
from qorgan.live.meter import Band
from qorgan.live.session import advance, initial_session

# Utterances are replayed as if perfectly transcribed -- streaming ASR confidence loss is a
# separate concern (`qorgan.asr.stream.stream_transcribe`), not what this harness measures.
_REPLAY_CONFIDENCE = 1.0
_DEFAULT_SPLITS: tuple[str, ...] = ("test", "authored_heldout")

# (column label, StreamReport field, interval field or None) -- false-latch rate first, per
# the CLI's contract; rates carry their exact binomial 95 % CI inline (PLAN_2026-09 A1).
_REPORT_COLUMNS: tuple[tuple[str, str, str | None], ...] = (
    ("False-Latch Rate [95% CI]", "false_latch_rate", "false_latch_ci"),
    ("Alert-Hit Rate [95% CI]", "alert_hit_rate", "alert_hit_ci"),
    ("Median Turns", "median_turns_to_alert", None),
    ("P90 Turns", "p90_turns_to_alert", None),
    ("N+", "n_positive", None),
    ("N-", "n_negative", None),
)


class StreamResult(BaseModel):
    """One dialogue replayed turn by turn through the live pipeline."""

    model_config = ConfigDict(frozen=True)

    dialogue_id: str
    is_scam: bool
    latched: bool
    # 1-indexed turn where the meter FIRST latched during the call; `None` if it never did.
    turns_to_alert: int | None = Field(default=None, ge=1)
    max_score: float = Field(ge=0.0, le=100.0)
    final_band: Band


class StreamReport(BaseModel):
    """Aggregate streaming metrics over a corpus split."""

    model_config = ConfigDict(frozen=True)

    false_latch_rate: float = Field(ge=0.0, le=1.0)
    # Exact binomial 95 % interval as (low, high); None when the class is absent.
    false_latch_ci: tuple[float, float] | None = None
    alert_hit_rate: float = Field(ge=0.0, le=1.0)
    alert_hit_ci: tuple[float, float] | None = None
    median_turns_to_alert: float | None = Field(default=None, ge=1.0)
    p90_turns_to_alert: float | None = Field(default=None, ge=1.0)
    n_positive: int = Field(ge=0)
    n_negative: int = Field(ge=0)


def replay_dialogue(
    dialogue: Dialogue, *, locale: str, backend: str | None = None
) -> StreamResult:
    """Feed `dialogue`'s utterances one at a time through `live.session.advance` (each as a
    `CommittedUtterance` at full ASR confidence) and summarize how the meter behaved."""
    state = initial_session(locale, backend=backend)
    max_score = 0.0
    turns_to_alert: int | None = None
    ever_latched = False
    final_band: Band = "low"

    for turn_index, utterance in enumerate(dialogue.utterances, start=1):
        committed = CommittedUtterance(text=utterance.text, confidence=_REPLAY_CONFIDENCE)
        state, update = advance(state, committed)
        max_score = max(max_score, update.meter.score)
        final_band = update.band
        if update.meter.latched:
            ever_latched = True
            if turns_to_alert is None:
                turns_to_alert = turn_index

    return StreamResult(
        dialogue_id=dialogue.id,
        is_scam=dialogue.label.risk >= _TRUTH_THRESHOLD,
        latched=ever_latched,
        turns_to_alert=turns_to_alert,
        max_score=max_score,
        final_band=final_band,
    )


def evaluate_stream(
    dialogues: Sequence[Dialogue], *, locale: str, backend: str | None = None
) -> StreamReport:
    """Replay every dialogue and aggregate into a `StreamReport`.

    `false_latch_rate` is the primary number: the fraction of negative dialogues whose
    meter ever latches during the call (the live analog of FPR). `alert_hit_rate` is its
    positive-side counterpart. `median_turns_to_alert`/`p90_turns_to_alert` summarize how
    fast the meter reacts, computed only over positives that did latch (`None` when none
    did -- a percentile over zero samples is undefined, not zero). Raises `ValueError` for
    an empty `dialogues` sequence.
    """
    if not dialogues:
        raise ValueError("dialogues must not be empty")

    results = tuple(replay_dialogue(d, locale=locale, backend=backend) for d in dialogues)
    positives = [r for r in results if r.is_scam]
    negatives = [r for r in results if not r.is_scam]
    latch_turns = [
        r.turns_to_alert for r in positives if r.latched and r.turns_to_alert is not None
    ]

    false_latches = sum(1 for r in negatives if r.latched)
    hits = sum(1 for r in positives if r.latched)
    return StreamReport(
        false_latch_rate=_rate(false_latches, len(negatives)),
        false_latch_ci=_ci_tuple(false_latches, len(negatives)),
        alert_hit_rate=_rate(hits, len(positives)),
        alert_hit_ci=_ci_tuple(hits, len(positives)),
        median_turns_to_alert=_percentile(latch_turns, 0.5) if latch_turns else None,
        p90_turns_to_alert=_percentile(latch_turns, 0.9) if latch_turns else None,
        n_positive=len(positives),
        n_negative=len(negatives),
    )


def _ci_tuple(numerator: int, denominator: int) -> tuple[float, float] | None:
    interval = intervals.binomial_interval(numerator, denominator)
    return interval.as_tuple() if interval else None


def _rate(numerator: int, denominator: int) -> float:
    """`numerator / denominator`, defined as 0.0 for a zero denominator: no negatives means
    vacuously no false latches, and no positives means vacuously no hits."""
    return numerator / denominator if denominator else 0.0


def _percentile(values: Sequence[int], q: float) -> float:
    """Linear-interpolation percentile (the same convention as `numpy.percentile`'s default
    `'linear'` method) over `values`, for `q` in `[0, 1]`. A single-element sequence returns
    that element at any `q`. Raises `ValueError` for an empty sequence -- callers must guard
    the "no samples" case explicitly rather than get a silent/misleading number back."""
    if not values:
        raise ValueError("values must not be empty")
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])

    rank = q * (len(ordered) - 1)
    lower_index = int(rank)
    upper_index = min(lower_index + 1, len(ordered) - 1)
    weight = rank - lower_index
    return ordered[lower_index] * (1.0 - weight) + ordered[upper_index] * weight


def format_stream_report(results: Mapping[str, StreamReport]) -> str:
    """Render a compact markdown table, one row per split, false-latch rate first among the
    metric columns (per the CLI's contract). `None` percentiles render as `-`."""
    header = "| Split | " + " | ".join(label for label, _, _ in _REPORT_COLUMNS) + " |"
    divider = "|---|" + "|".join("---" for _ in _REPORT_COLUMNS) + "|"
    rows = [header, divider]
    for split_name, report in results.items():
        cells = [split_name] + [
            _fmt_with_interval(getattr(report, field), getattr(report, ci_field))
            if ci_field else _fmt(getattr(report, field))
            for _, field, ci_field in _REPORT_COLUMNS
        ]
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join(rows)


def _fmt(value: object) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _fmt_with_interval(value: object, interval: tuple[float, float] | None) -> str:
    if interval is None:
        return f"{_fmt(value)} [-]"
    return f"{_fmt(value)} [{_fmt(interval[0])}, {_fmt(interval[1])}]"


def main(argv: Sequence[str] | None = None) -> None:
    """CLI: `python -m qorgan.eval.stream --split authored_heldout --split test [--backend mock]`."""
    cfg = get_config()
    parser = argparse.ArgumentParser(
        description="Streaming (per-turn) live-meter evaluation over corpus splits."
    )
    parser.add_argument("--split", action="append", dest="splits", default=None, help="Split name (repeatable)")
    parser.add_argument("--backend", default=None, help="Override classifier backend (llm|xlmr|linear|mock)")
    parser.add_argument("--processed-dir", type=Path, default=None, help="Dir holding <split>.jsonl")
    parser.add_argument("--locale", default=None, help="Locale for the replayed live session")
    args = parser.parse_args(argv)

    split_names = args.splits or list(_DEFAULT_SPLITS)
    processed_dir = args.processed_dir or (cfg.data_dir / "processed")
    locale = args.locale or cfg.default_locale

    results = {
        name: evaluate_stream(load_split(processed_dir, name), locale=locale, backend=args.backend)
        for name in split_names
    }
    print(format_stream_report(results))


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
