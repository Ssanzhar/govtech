"""What the shipped model does on real scammer phrasing from court verdicts (ADR D68).

    QORGAN_CLASSIFIER_BACKEND=linear QORGAN_LINEAR_WEIGHTS=web python -m qorgan.eval.court_quotes

`court_quotes` is positives-only and utterance-level, so there is no FPR here -- only how
often a real scam line, on its own, (a) reaches the alert threshold as a per-utterance risk,
(b) fires a hard-signal cue, and (c) leaves the live meter latched or in the High band after
one turn. Each quote is replayed as a fresh one-turn session through `live.session.advance`,
exactly what the citizen page does with its first utterance. A secondary, clearly composed
number streams all quotes of one verdict, in verdict order, as a pseudo-call.

Every rate carries its Clopper-Pearson interval. Held-out rules: never tune on this, never
extend the lexicon from it (ADRs D35/D43). Prints ids and numbers, never quote text.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from qorgan.asr.stream import CommittedUtterance
from qorgan.config import get_config
from qorgan.data.court_quotes import QUOTES_FILENAME, RAW_SUBDIR, QuoteRow, load_quotes
from qorgan.eval import intervals

_REPLAY_CONFIDENCE = 1.0
_LOCALE_BY_LANGUAGE = {"ru": "ru", "kk": "kk", "mixed": "ru", "en": "en"}


@dataclass(frozen=True)
class QuoteScore:
    id: str
    verdict_id: str
    language: str
    speech: str
    risk: float
    alert: bool  # per-utterance risk >= the shipped alert threshold
    cue_fired: bool  # any hard-signal cue matched in the quote
    cue_tactics: tuple[str, ...]
    meter_score: float
    meter_latched: bool
    meter_band: str


@dataclass(frozen=True)
class Rate:
    hits: int
    n: int
    low: float
    high: float

    @property
    def value(self) -> float:
        return self.hits / self.n if self.n else 0.0

    def render(self) -> str:
        return f"{self.value:.3f} [{self.low:.3f}, {self.high:.3f}] ({self.hits}/{self.n})"


def rate(hits: int, n: int) -> Rate:
    interval = intervals.binomial_interval(hits, n)
    low, high = (interval.low, interval.high) if interval else (0.0, 0.0)
    return Rate(hits=hits, n=n, low=low, high=high)


def _default_cue_fn() -> Callable[[str], tuple[str, ...]]:
    from qorgan.classifier.features import match_cues
    from qorgan.classifier.predict import _get_linear_bundle

    lexicon = _get_linear_bundle().lexicon
    return lambda text: tuple(sorted({m.tactic_id for m in match_cues(text, lexicon)}))


def score_quotes(
    rows: Sequence[QuoteRow],
    *,
    cue_fn: Callable[[str], tuple[str, ...]] | None = None,
    backend: str | None = None,
) -> tuple[QuoteScore, ...]:
    """Replay each quote as a one-turn live session; pure apart from the model call."""
    from qorgan.live.session import advance, initial_session

    cfg = get_config()
    cues = cue_fn or _default_cue_fn()
    scores = []
    for row in rows:
        state = initial_session(_LOCALE_BY_LANGUAGE.get(row.language, "ru"), backend=backend)
        _, update = advance(state, CommittedUtterance(text=row.text, confidence=_REPLAY_CONFIDENCE))
        fired = cues(row.text)
        scores.append(
            QuoteScore(
                id=row.id, verdict_id=row.verdict_id, language=row.language, speech=row.speech,
                risk=update.result.risk, alert=update.result.risk >= cfg.risk_threshold,
                cue_fired=bool(fired), cue_tactics=fired,
                meter_score=update.meter.score, meter_latched=update.meter.latched, meter_band=update.band,
            )
        )
    return tuple(scores)


def stream_verdicts(rows: Sequence[QuoteRow], *, backend: str | None = None) -> dict[str, bool]:
    """Composed pseudo-calls: every quote of one verdict, in order -> did the meter ever latch?"""
    from qorgan.live.session import advance, initial_session

    by_verdict: dict[str, list[QuoteRow]] = defaultdict(list)
    for row in rows:
        by_verdict[row.verdict_id].append(row)
    latched: dict[str, bool] = {}
    for verdict_id, quotes in by_verdict.items():
        state = initial_session(_LOCALE_BY_LANGUAGE.get(quotes[0].language, "ru"), backend=backend)
        ever = False
        for quote in quotes:
            state, update = advance(state, CommittedUtterance(text=quote.text, confidence=_REPLAY_CONFIDENCE))
            ever = ever or update.meter.latched
        latched[verdict_id] = ever
    return latched


def summarize(scores: Sequence[QuoteScore]) -> dict[str, Rate | float]:
    n = len(scores)
    return {
        "risk>=threshold": rate(sum(s.alert for s in scores), n),
        "hard-signal cue fired": rate(sum(s.cue_fired for s in scores), n),
        "meter latched after 1 turn": rate(sum(s.meter_latched for s in scores), n),
        "meter band high/critical after 1 turn": rate(sum(s.meter_band in ("high", "critical") for s in scores), n),
        "median risk": statistics.median(s.risk for s in scores) if scores else 0.0,
    }


def render(scores: Sequence[QuoteScore], latched_calls: dict[str, bool], model_source: str) -> str:
    cfg = get_config()
    lines = [
        f"court_quotes -- model source: {model_source}; alert threshold {cfg.risk_threshold}",
        "",
        "| Group | N | risk>=thr [95% CI] | cue fired [95% CI] | latched@1 [95% CI] | high band@1 [95% CI] | median risk |",
        "|---|---|---|---|---|---|---|",
    ]
    groups: dict[str, list[QuoteScore]] = {"all": list(scores)}
    for s in scores:
        groups.setdefault(f"lang={s.language}", []).append(s)
        groups.setdefault(f"speech={s.speech}", []).append(s)
    for name, group in groups.items():
        m = summarize(group)
        lines.append(
            f"| {name} | {len(group)} | {m['risk>=threshold'].render()} | {m['hard-signal cue fired'].render()} "
            f"| {m['meter latched after 1 turn'].render()} | {m['meter band high/critical after 1 turn'].render()} "
            f"| {m['median risk']:.3f} |"
        )
    calls = rate(sum(latched_calls.values()), len(latched_calls))
    lines += ["", f"pseudo-calls (all quotes of one verdict streamed in order; composed): latched {calls.render()}"]
    tactics: dict[str, int] = defaultdict(int)
    for s in scores:
        for tactic in s.cue_tactics:
            tactics[tactic] += 1
    lines.append("cue tactics fired: " + (", ".join(f"{t}={c}" for t, c in sorted(tactics.items())) or "none"))
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    from qorgan.classifier.predict import linear_model_source

    parser = argparse.ArgumentParser(description="Shipped model on the court_quotes held-out split (ADR D68).")
    parser.add_argument("--path", type=Path, default=None, help="defaults to data/raw/court_quotes/quotes.jsonl")
    parser.add_argument("--out", type=Path, default=None, help="per-quote scores JSON (ids only, no text)")
    args = parser.parse_args(argv)

    cfg = get_config()
    if cfg.classifier_backend != "linear":
        raise SystemExit("court_quotes measures the shipped model: set QORGAN_CLASSIFIER_BACKEND=linear")
    path = args.path or cfg.data_dir / RAW_SUBDIR / QUOTES_FILENAME
    try:
        rows = load_quotes(path)
    except FileNotFoundError:
        raise SystemExit(f"{path} not found: court_quotes is local-only (docs/DECISIONS.md D68)") from None
    if not rows:
        raise SystemExit(f"{path} is empty")
    scores = score_quotes(rows)
    print(render(scores, stream_verdicts(rows), linear_model_source()))
    if args.out:
        args.out.write_text(
            json.dumps([s.__dict__ for s in scores], ensure_ascii=False, indent=1, default=list), encoding="utf-8"
        )


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
