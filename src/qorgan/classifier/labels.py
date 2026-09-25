"""Multi-label tactic label space and target/weight encoding for the fine-tuned XLM-R
head (CLAUDE.md SS4 -- classifier has a binary risk head plus a multi-label tactic head).

Invariant: positive class = scam = 1, negative = legitimate = 0 (same convention as
`qorgan.eval.metrics`). Every function here is pure: no I/O, no mutation of inputs.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

_BINARY_VALUES = (0, 1)
_POSITIVE_THRESHOLD = 0.5


def build_label_space(tactic_ids: Sequence[str]) -> tuple[str, ...]:
    """Deduplicate `tactic_ids` preserving first-seen order.

    Raises `ValueError` if `tactic_ids` is empty or every entry is blank.
    """
    space: list[str] = []
    seen: set[str] = set()
    for tactic_id in tactic_ids:
        if not tactic_id or not tactic_id.strip():
            continue
        if tactic_id in seen:
            continue
        seen.add(tactic_id)
        space.append(tactic_id)

    if not space:
        raise ValueError("tactic_ids must contain at least one non-blank id")
    return tuple(space)


def default_label_space() -> tuple[str, ...]:
    """The project's tactic label space, sourced from the configured taxonomy."""
    from qorgan.taxonomy import get_taxonomy

    return build_label_space(get_taxonomy().tactic_ids())


def encode_tactics(ids: Sequence[str], label_space: Sequence[str]) -> list[float]:
    """Multi-hot encode `ids` against `label_space`: `1.0` if present, else `0.0`.

    Ids not found in `label_space` are ignored (not an error) -- callers may pass
    raw LLM/annotator tags that fall outside the trained label space.
    """
    id_set = set(ids)
    return [1.0 if label in id_set else 0.0 for label in label_space]


def decode_tactics(
    probs: Sequence[float],
    label_space: Sequence[str],
    threshold: float,
    per_tactic: Mapping[str, float] | None = None,
) -> list[tuple[str, float]]:
    """Return `(id, prob)` pairs for every position at or above its threshold, sorted by
    prob DESC then id ASC. `per_tactic` overrides `threshold` for the tactics it names
    (tuned on out-of-fold train + `val`, ADR D30); the rest use `threshold`.

    Raises `ValueError` if `probs` and `label_space` differ in length, or a threshold
    is not in `[0, 1]`.
    """
    if len(probs) != len(label_space):
        raise ValueError(
            f"probs and label_space must have the same length, got {len(probs)} and {len(label_space)}"
        )
    overrides = dict(per_tactic or {})
    for value in (threshold, *overrides.values()):
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"threshold must be in [0, 1], got {value}")

    selected = [
        (label, prob) for label, prob in zip(label_space, probs) if prob >= overrides.get(label, threshold)
    ]
    return sorted(selected, key=lambda pair: (-pair[1], pair[0]))


def pos_weight(targets: Sequence[int]) -> float:
    """`BCEWithLogitsLoss`-style `pos_weight`: `num_zeros / num_ones`.

    Returns `1.0` if there are no positives (undefined ratio, neutral default).
    Raises `ValueError` if `targets` is empty or contains a value outside `{0, 1}`.
    """
    if not targets:
        raise ValueError("targets must not be empty")
    for value in targets:
        if value not in _BINARY_VALUES:
            raise ValueError(f"targets must contain only 0/1 values, found {value!r}")

    num_ones = sum(1 for value in targets if value == 1)
    num_zeros = len(targets) - num_ones
    if num_ones == 0:
        return 1.0
    return num_zeros / num_ones


def tactic_pos_weights(label_matrix: Sequence[Sequence[float]]) -> list[float]:
    """Per-column `pos_weight` over a matrix of multi-hot rows (value `>= 0.5` counts
    as positive).

    Raises `ValueError` if `label_matrix` is empty or its rows have differing lengths.
    """
    if not label_matrix:
        raise ValueError("label_matrix must not be empty")

    row_length = len(label_matrix[0])
    for row in label_matrix:
        if len(row) != row_length:
            raise ValueError("label_matrix rows must all have the same length")

    weights: list[float] = []
    for column in range(row_length):
        column_values = [row[column] for row in label_matrix]
        binary_targets = [1 if value >= _POSITIVE_THRESHOLD else 0 for value in column_values]
        weights.append(pos_weight(binary_targets))
    return weights
