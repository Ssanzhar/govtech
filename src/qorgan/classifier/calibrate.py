"""Temperature scaling for the fine-tuned XLM-R risk head (CLAUDE.md SS4 -- calibrated
confidence is required for grounded explainability).

`apply_temperature` maps raw logits to calibrated probabilities given a fitted
`temperature`; `fit_temperature` learns that scalar on a held-out (validation) set by
minimising negative log-likelihood; `save_temperature`/`load_temperature` persist it next
to the exported model. All of this is pure numeric code -- no torch -- so it is fast and
trivially testable.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from pathlib import Path

# Exponent magnitude beyond which `math.exp` would over/underflow a float64; clamping
# here keeps `apply_temperature` total (never raises `OverflowError`) while still
# saturating to the correct 0.0/1.0 limit.
_MAX_EXPONENT_MAGNITUDE = 700.0
# Small floor so log() never sees exactly 0.0/1.0.
_EPS = 1e-7
# Coarse-to-fine search grid for the temperature (positive scalar).
_SEARCH_LOW = 0.05
_SEARCH_HIGH = 10.0
_SEARCH_STEPS = 100
_REFINE_ROUNDS = 3
_BINARY_VALUES = (0, 1)


def apply_temperature(logits: Sequence[float], temperature: float) -> list[float]:
    """Temperature-scaled sigmoid: `1 / (1 + exp(-logit / temperature))` for each logit.

    Raises `ValueError` if `temperature <= 0`. Clamps the exponent so extreme-magnitude
    logits saturate to `~0.0`/`~1.0` instead of overflowing.
    """
    if temperature <= 0:
        raise ValueError(f"temperature must be > 0, got {temperature}")

    return [_scaled_sigmoid(logit, temperature) for logit in logits]


def _scaled_sigmoid(logit: float, temperature: float) -> float:
    exponent = -logit / temperature
    if exponent > _MAX_EXPONENT_MAGNITUDE:
        return 0.0
    if exponent < -_MAX_EXPONENT_MAGNITUDE:
        return 1.0
    return 1.0 / (1.0 + math.exp(exponent))


def negative_log_likelihood(logits: Sequence[float], targets: Sequence[int], temperature: float) -> float:
    """Mean binary NLL of `targets` under temperature-scaled `logits`. Lower = better calibrated."""
    if len(logits) != len(targets):
        raise ValueError(f"logits and targets must have the same length, got {len(logits)} and {len(targets)}")
    probs = apply_temperature(logits, temperature)
    total = 0.0
    for prob, target in zip(probs, targets):
        clipped = min(max(prob, _EPS), 1.0 - _EPS)
        total += -(target * math.log(clipped) + (1 - target) * math.log(1.0 - clipped))
    return total / len(targets)


def fit_temperature(logits: Sequence[float], targets: Sequence[int]) -> float:
    """Learn the temperature that minimises NLL of `targets` under `logits`.

    Coarse-to-fine grid search over a positive range (no scipy/torch dependency). Raises
    `ValueError` if inputs are empty, mismatched, or `targets` is non-binary.
    """
    if not logits:
        raise ValueError("logits must not be empty")
    if len(logits) != len(targets):
        raise ValueError(f"logits and targets must have the same length, got {len(logits)} and {len(targets)}")
    for value in targets:
        if value not in _BINARY_VALUES:
            raise ValueError(f"targets must contain only 0/1 values, found {value!r}")

    low, high = _SEARCH_LOW, _SEARCH_HIGH
    best_t = 1.0
    for _ in range(_REFINE_ROUNDS):
        best_t = _grid_argmin(logits, targets, low, high)
        window = (high - low) / _SEARCH_STEPS
        low = max(_SEARCH_LOW, best_t - window)
        high = min(_SEARCH_HIGH, best_t + window)
    return best_t


def _grid_argmin(logits: Sequence[float], targets: Sequence[int], low: float, high: float) -> float:
    step = (high - low) / _SEARCH_STEPS
    best_t, best_nll = low, float("inf")
    for i in range(_SEARCH_STEPS + 1):
        temperature = low + i * step
        if temperature <= 0:
            continue
        nll = negative_log_likelihood(logits, targets, temperature)
        if nll < best_nll:
            best_nll, best_t = nll, temperature
    return best_t


def save_temperature(temperature: float, path: Path) -> None:
    """Persist a fitted temperature as JSON (`{"temperature": T}`)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"temperature": temperature}), encoding="utf-8")


def load_temperature(path: Path) -> float:
    """Load a temperature saved by `save_temperature`; defaults to `1.0` if absent."""
    if not path.exists():
        return 1.0
    return float(json.loads(path.read_text(encoding="utf-8"))["temperature"])


# --- per-tactic decision thresholds (2026-09-20, ADR D30) ---------------------------------------
# The class-weighted per-tactic LRs over-predict at a flat 0.5 (recall ~0.9, precision 0.3
# on the rare tactics). Each tactic gets the threshold that maximises its F1 on the tuning
# split (`val`). Guard rails, measured on 2026-09-20: a threshold tuned on fewer than 8 val
# positives overfits (gov_police tuned on 4 collapsed to F1 0 on test), so those tactics keep
# the default; and the grid stops at 0.70 so a confident hard signal (weight >= 0.8, the live
# meter's floor) can never be filtered out -- the streaming behaviour is provably unchanged.
TACTIC_THRESHOLD_GRID: tuple[float, ...] = tuple(round(0.5 + 0.05 * i, 2) for i in range(5))  # 0.50 .. 0.70
TACTIC_THRESHOLD_MIN_POSITIVES = 8


def tune_tactic_thresholds(
    probs: Sequence[Sequence[float]],
    truth: Sequence[Sequence[int]],
    label_space: Sequence[str],
    *,
    default: float = 0.5,
    grid: Sequence[float] = TACTIC_THRESHOLD_GRID,
    min_positives: int = TACTIC_THRESHOLD_MIN_POSITIVES,
) -> dict[str, float]:
    """`{tactic_id: threshold}` maximising per-tactic F1 over `grid` on a tuning set
    (`probs`/`truth` are `(n, L)`, columns in `label_space` order). Ties go to the LOWEST
    threshold (keep recall); a tactic with fewer than `min_positives` positives, or one
    where no threshold beats the default's F1, keeps `default`. Raises `ValueError` on
    shape mismatch."""
    rows = [list(r) for r in probs]
    labels = [list(r) for r in truth]
    if len(rows) != len(labels) or any(len(r) != len(label_space) for r in rows) or any(len(r) != len(label_space) for r in labels):
        raise ValueError("probs and truth must be (n, len(label_space))")
    chosen: dict[str, float] = {}
    for j, tactic_id in enumerate(label_space):
        column = [(r[j], l[j]) for r, l in zip(rows, labels)]
        if sum(t for _, t in column) < min_positives:
            chosen[tactic_id] = default
            continue
        best_threshold, best_f1 = default, _f1_at(column, default)
        for threshold in grid:
            f1 = _f1_at(column, threshold)
            if f1 > best_f1 + 1e-12:
                best_threshold, best_f1 = threshold, f1
        chosen[tactic_id] = float(best_threshold)
    return chosen


def _f1_at(column: Sequence[tuple[float, int]], threshold: float) -> float:
    tp = sum(1 for p, t in column if p >= threshold and t)
    fp = sum(1 for p, t in column if p >= threshold and not t)
    fn = sum(1 for p, t in column if p < threshold and t)
    return 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
