"""Offline meter sweep (PLAN_2026-09 A6, ADR D29): tune the live meter without re-scoring.

`eval.stream` re-embeds every rolling window for every parameter set, so a single run takes
minutes. The meter, however, only consumes `(risk, hard signals)` per turn -- and those do
not depend on the meter. This module traces each dialogue once (the same rolling window
`live.session` uses, the same `predict.score`), caches the traces, and then simulates any
number of meter variants in milliseconds through the **production** `meter.update` with an
injected config, so the sweep can never drift from what ships. Tables are false-latch first.

Run:
  python -m qorgan.eval.meter_sweep --split test --split authored_heldout \\
      --cache data/cache/meter_traces.json --min-turns 1,2,3 --damping 1,2,3
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from qorgan.config import Config, get_config
from qorgan.data.schema import Dialogue, ScoreResult
from qorgan.eval.run import load_split
from qorgan.live import meter as meter_mod
from qorgan.live.session import _rolling_window
from qorgan.taxonomy import get_taxonomy

_TRACE_FORMAT_VERSION = 1
_TRUTH_THRESHOLD = 0.5
_REPLAY_CONFIDENCE = 1.0
ScoreFn = Callable[[str], ScoreResult]


class TurnTrace(BaseModel):
    """What the meter sees for one dialogue: per turn, the window's risk and hard signals."""

    model_config = ConfigDict(frozen=True)

    dialogue_id: str
    split: str
    positive: bool
    turns: tuple[dict[str, Any], ...] = Field(min_length=1)  # {"risk": float, "hard": {id: weight}}


class MeterVariant(BaseModel):
    model_config = ConfigDict(frozen=True)

    min_turns_to_arm: int = Field(ge=1)
    short_window_turns: int = Field(default=1, ge=1)

    @property
    def label(self) -> str:
        damping = f"+damping={self.short_window_turns}" if self.short_window_turns > 1 else ""
        return f"min_turns={self.min_turns_to_arm}{damping}"


class SplitOutcome(BaseModel):
    model_config = ConfigDict(frozen=True)

    false_latches: int
    negatives: int
    alert_hits: int
    positives: int
    median_turns_to_alert: float | None
    p90_turns_to_alert: float | None


class SweepRow(BaseModel):
    model_config = ConfigDict(frozen=True)

    variant: MeterVariant
    by_split: dict[str, SplitOutcome]


# --- tracing (the expensive step, done once) --------------------------------------------------


def trace_dialogue(
    dialogue_id: str, split: str, positive: bool, utterances: Sequence[str], *, score_fn: ScoreFn, hard_signal_ids: set[str]
) -> TurnTrace:
    turns = []
    for i in range(1, len(utterances) + 1):
        result = score_fn(_rolling_window(tuple(utterances[:i])))
        turns.append({"risk": result.risk, "hard": {t.id: t.weight for t in result.tags if t.id in hard_signal_ids}})
    return TurnTrace(dialogue_id=dialogue_id, split=split, positive=positive, turns=tuple(turns))


def trace_split(dialogues: Sequence[Dialogue], split: str, *, score_fn: ScoreFn) -> list[TurnTrace]:
    hard = set(get_taxonomy().hard_signal_ids())
    return [
        trace_dialogue(d.id, split, d.label.risk >= _TRUTH_THRESHOLD, [u.text for u in d.utterances], score_fn=score_fn, hard_signal_ids=hard)
        for d in dialogues
    ]


def save_traces(traces: Sequence[TurnTrace], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"format_version": _TRACE_FORMAT_VERSION, "traces": [t.model_dump() for t in traces]}
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


def load_traces(path: Path) -> list[TurnTrace]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("format_version") != _TRACE_FORMAT_VERSION:
        raise ValueError(f"unsupported trace format {payload.get('format_version')!r}")
    return [TurnTrace.model_validate(t) for t in payload["traces"]]


# --- simulation (the cheap step) --------------------------------------------------------------


def simulate(trace: TurnTrace, variant: MeterVariant, config: Config) -> int | None:
    """Replay a trace through the production meter under `variant`; the 1-indexed turn at
    which the latch first engaged, or None."""
    cfg = config.model_copy(update={"meter_min_turns_to_arm": variant.min_turns_to_arm, "meter_short_window_turns": variant.short_window_turns})
    state = meter_mod.initial_state()
    for i, turn in enumerate(trace.turns, 1):
        state = meter_mod.update(state, risk=turn["risk"], asr_confidence=_REPLAY_CONFIDENCE, hard_signals=turn["hard"], config=cfg)
        if state.latched:
            return i
    return None


def sweep(traces: Sequence[TurnTrace], variants: Sequence[MeterVariant], config: Config) -> list[SweepRow]:
    rows = []
    for variant in variants:
        by_split: dict[str, SplitOutcome] = {}
        for split in sorted({t.split for t in traces}):
            outcomes = [(t.positive, simulate(t, variant, config)) for t in traces if t.split == split]
            hits = sorted(turn for positive, turn in outcomes if positive and turn is not None)
            by_split[split] = SplitOutcome(
                false_latches=sum(1 for positive, turn in outcomes if not positive and turn is not None),
                negatives=sum(1 for positive, _ in outcomes if not positive),
                alert_hits=len(hits),
                positives=sum(1 for positive, _ in outcomes if positive),
                median_turns_to_alert=_percentile(hits, 0.5),
                p90_turns_to_alert=_percentile(hits, 0.9),
            )
        rows.append(SweepRow(variant=variant, by_split=by_split))
    return rows


def _percentile(values: Sequence[int], q: float) -> float | None:
    if not values:
        return None
    return float(values[min(len(values) - 1, int(round(q * (len(values) - 1))))])


def render_sweep(rows: Sequence[SweepRow]) -> str:
    splits = sorted({s for r in rows for s in r.by_split})
    header = "| variant | " + " | ".join(f"{s} false-latch | {s} alert-hit | {s} median/p90 turns" for s in splits) + " |"
    lines = [header, "|---|" + "|".join("---|---|---" for _ in splits) + "|"]
    for row in rows:
        cells = []
        for s in splits:
            o = row.by_split[s]
            cells.append(f"{o.false_latches}/{o.negatives} | {o.alert_hits}/{o.positives} | {_fmt(o.median_turns_to_alert)}/{_fmt(o.p90_turns_to_alert)}")
        lines.append(f"| {row.variant.label} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _fmt(value: float | None) -> str:
    return "-" if value is None else f"{value:.0f}"


# --- CLI ---------------------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> None:  # pragma: no cover - CLI (real model)
    from qorgan.classifier import predict

    parser = argparse.ArgumentParser(description="Sweep live-meter parameters on cached per-turn traces.")
    parser.add_argument("--split", action="append", dest="splits", default=None)
    parser.add_argument("--processed-dir", type=Path, default=None)
    parser.add_argument("--cache", type=Path, default=None, help="trace cache; built when absent, reused when present")
    parser.add_argument("--min-turns", default="1,2,3")
    parser.add_argument("--damping", default="1,2,3")
    parser.add_argument("--backend", default=None)
    args = parser.parse_args(argv)
    cfg = get_config()
    processed = args.processed_dir or cfg.data_dir / "processed"
    splits = args.splits or ["test", "authored_heldout"]
    cache = args.cache or cfg.data_dir / "cache" / "meter_traces.json"
    if cache.exists():
        traces = [t for t in load_traces(cache) if t.split in splits]
        print(f"traces: {len(traces)} from {cache}")
    else:
        traces = []
        for split in splits:
            traces.extend(trace_split(load_split(processed, split), split, score_fn=lambda text: predict.score(text, backend=args.backend)))
        save_traces(traces, cache)
        print(f"traces: {len(traces)} -> {cache}")
    variants = [
        MeterVariant(min_turns_to_arm=int(m), short_window_turns=int(d))
        for m in args.min_turns.split(",") for d in args.damping.split(",")
    ]
    print(f"shipped: min_turns_to_arm={cfg.meter_min_turns_to_arm}, short_window_turns={cfg.meter_short_window_turns}\n")
    print(render_sweep(sweep(traces, variants, cfg)))


if __name__ == "__main__":
    main()
