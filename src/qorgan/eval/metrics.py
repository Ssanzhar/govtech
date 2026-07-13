"""FPR-first evaluation metric primitives for `qorgan.eval.run`.

Convention: positive class = scam = 1, negative = legitimate = 0. False-positive rate
(FPR) is the project's PRIMARY metric (CLAUDE.md SS3.5) -- a false alarm on a real bank
call destroys user trust, so it is reported first everywhere it matters, not buried
after precision/recall. Every function here is pure: no I/O, no mutation of inputs.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from sklearn.metrics import adjusted_rand_score, average_precision_score

_BINARY_VALUES = (0, 1)


def _validate_same_length(a: Sequence[Any], b: Sequence[Any], a_name: str, b_name: str) -> None:
    if len(a) != len(b):
        raise ValueError(f"{a_name} and {b_name} must have the same length, got {len(a)} and {len(b)}")


def _validate_binary(values: Sequence[int], name: str) -> None:
    for value in values:
        if value not in _BINARY_VALUES:
            raise ValueError(f"{name} must contain only 0/1 values, found {value!r}")


def confusion_counts(y_true: Sequence[int], y_pred: Sequence[int]) -> tuple[int, int, int, int]:
    """Return `(tp, fp, tn, fn)` for binary labels (positive = scam = 1).

    Raises `ValueError` if `y_true`/`y_pred` have different lengths, or contain any
    value outside `{0, 1}`.
    """
    y_true_list = list(y_true)
    y_pred_list = list(y_pred)
    _validate_same_length(y_true_list, y_pred_list, "y_true", "y_pred")
    _validate_binary(y_true_list, "y_true")
    _validate_binary(y_pred_list, "y_pred")

    tp = fp = tn = fn = 0
    for true_value, pred_value in zip(y_true_list, y_pred_list):
        if true_value == 1 and pred_value == 1:
            tp += 1
        elif true_value == 0 and pred_value == 1:
            fp += 1
        elif true_value == 0 and pred_value == 0:
            tn += 1
        else:
            fn += 1
    return tp, fp, tn, fn


def false_positive_rate(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    """PRIMARY METRIC: `fp / (fp + tn)`. Returns `0.0` when there are no true negatives
    (`fp + tn == 0`) rather than raising -- a well-defined "no false alarms observed".
    """
    _, fp, tn, _ = confusion_counts(y_true, y_pred)
    denom = fp + tn
    return fp / denom if denom else 0.0


def precision(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    """`tp / (tp + fp)`, `0.0` if no predicted positives."""
    tp, fp, _, _ = confusion_counts(y_true, y_pred)
    denom = tp + fp
    return tp / denom if denom else 0.0


def recall(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    """`tp / (tp + fn)`, `0.0` if no actual positives."""
    tp, _, _, fn = confusion_counts(y_true, y_pred)
    denom = tp + fn
    return tp / denom if denom else 0.0


def f1(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    """Harmonic mean of precision and recall, `0.0` if both are `0.0`."""
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    denom = p + r
    return 2 * p * r / denom if denom else 0.0


def binarize(scores: Sequence[float], threshold: float) -> list[int]:
    """Threshold continuous `scores` into `{0, 1}`; `score == threshold` maps to `1`.

    Raises `ValueError` if `threshold` is not in `[0, 1]`.
    """
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"threshold must be in [0, 1], got {threshold}")
    return [1 if score >= threshold else 0 for score in scores]


def pr_auc(y_true: Sequence[int], y_scores: Sequence[float]) -> float:
    """Average precision (area under the PR curve) via sklearn.

    Guarded against sklearn's own edge-case exceptions: an empty `y_true` or a
    single-class `y_true` (PR-AUC is undefined without both classes present) returns
    `0.0` instead of raising.
    """
    y_true_list = list(y_true)
    if not y_true_list or len(set(y_true_list)) < 2:
        return 0.0
    return float(average_precision_score(y_true_list, list(y_scores)))


def per_label_f1(
    true_labels: Sequence[set[str]],
    pred_labels: Sequence[set[str]],
    label_ids: Sequence[str],
) -> dict[str, float]:
    """Multi-label per-tactic F1: for each id in `label_ids`, treat "id present in the
    set at this position" as the positive class across the aligned sequences.

    Raises `ValueError` if `true_labels`/`pred_labels` have different lengths.
    """
    true_list = list(true_labels)
    pred_list = list(pred_labels)
    _validate_same_length(true_list, pred_list, "true_labels", "pred_labels")

    scores: dict[str, float] = {}
    for label_id in label_ids:
        y_true_bin = [1 if label_id in labels else 0 for labels in true_list]
        y_pred_bin = [1 if label_id in labels else 0 for labels in pred_list]
        scores[label_id] = f1(y_true_bin, y_pred_bin)
    return scores


def purity(labels_true: Sequence[Any], labels_pred: Sequence[Any]) -> float:
    """Cluster purity: sum over predicted clusters of the max true-class overlap,
    divided by N. `0.0` for empty input.

    Raises `ValueError` if `labels_true`/`labels_pred` have different lengths.
    """
    true_list = list(labels_true)
    pred_list = list(labels_pred)
    _validate_same_length(true_list, pred_list, "labels_true", "labels_pred")
    if not true_list:
        return 0.0

    cluster_class_counts: dict[Any, dict[Any, int]] = {}
    for true_value, pred_value in zip(true_list, pred_list):
        class_counts = cluster_class_counts.setdefault(pred_value, {})
        class_counts[true_value] = class_counts.get(true_value, 0) + 1

    total_correct = sum(max(class_counts.values()) for class_counts in cluster_class_counts.values())
    return total_correct / len(true_list)


def adjusted_rand(labels_true: Sequence[Any], labels_pred: Sequence[Any]) -> float:
    """Adjusted Rand Index via sklearn. `0.0` for empty input.

    Raises `ValueError` if `labels_true`/`labels_pred` have different lengths.
    """
    true_list = list(labels_true)
    pred_list = list(labels_pred)
    _validate_same_length(true_list, pred_list, "labels_true", "labels_pred")
    if not true_list:
        return 0.0
    return float(adjusted_rand_score(true_list, pred_list))


def binary_report(y_true: Sequence[int], y_scores: Sequence[float], *, threshold: float) -> dict[str, Any]:
    """The headline aggregator for the eval harness: binarize `y_scores` at
    `threshold`, then return an insertion-ordered dict with FPR reported first
    (CLAUDE.md SS3.5 -- FPR is the primary metric, never buried).
    """
    y_true_list = list(y_true)
    y_pred = binarize(y_scores, threshold)
    tp, fp, tn, fn = confusion_counts(y_true_list, y_pred)

    return {
        "fpr": false_positive_rate(y_true_list, y_pred),
        "precision": precision(y_true_list, y_pred),
        "recall": recall(y_true_list, y_pred),
        "f1": f1(y_true_list, y_pred),
        "pr_auc": pr_auc(y_true_list, y_scores),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "support": len(y_true_list),
        "threshold": threshold,
    }
