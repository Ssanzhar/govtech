"""Multi-label tactic head for the `linear` classifier.

Deliberately in its own module (not `linear_train`) so its pickled `__module__` is always
`qorgan.classifier.multilabel` -- never `__main__`. Running training via
`python -m qorgan.classifier.linear_train` makes that module `__main__`, which would
otherwise pickle this class as `__main__.MultiLabelHead` and break loading in any other
process (e.g. the eval harness or the app).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression

_LR_MAX_ITER = 2000


class MultiLabelHead:
    """One Logistic Regression per tactic. Any tactic with only one class present in
    training is skipped (predicted absent), so single-class columns never crash `fit`."""

    def __init__(self, label_space: Sequence[str]) -> None:
        self.label_space = tuple(label_space)
        self.models: dict[str, Any] = {}

    def fit(
        self, features: np.ndarray, target_matrix: np.ndarray, sample_weight: np.ndarray | None = None
    ) -> "MultiLabelHead":
        """`sample_weight` (optional, per row) lets an augmented copy share one unit with its
        source instead of counting as new evidence (ADR D31)."""
        for column, tactic_id in enumerate(self.label_space):
            column_targets = target_matrix[:, column].astype(int)
            if len(set(column_targets.tolist())) >= 2:
                self.models[tactic_id] = LogisticRegression(
                    class_weight="balanced", max_iter=_LR_MAX_ITER
                ).fit(features, column_targets, sample_weight=sample_weight)
            else:
                self.models[tactic_id] = None
        return self

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        out = np.zeros((features.shape[0], len(self.label_space)), dtype=np.float64)
        for column, tactic_id in enumerate(self.label_space):
            model = self.models.get(tactic_id)
            if model is not None:
                out[:, column] = model.predict_proba(features)[:, 1]
        return out


def out_of_fold_proba(
    features: np.ndarray, target_matrix: np.ndarray, label_space: Sequence[str], *, folds: int, seed: int,
    sample_weight: np.ndarray | None = None,
) -> np.ndarray:
    """`(n, L)` probabilities where every row comes from a head fitted WITHOUT that row
    (seeded K-fold). These are the fit-independent probabilities per-tactic thresholds are
    tuned on (ADR D30): the training rows become usable tuning data instead of the small
    `val` split alone. Raises `ValueError` for fewer than 2 folds or more folds than rows."""
    rows = features.shape[0]
    if folds < 2 or folds > rows:
        raise ValueError(f"folds must be in [2, {rows}], got {folds}")
    order = np.random.default_rng(seed).permutation(rows)
    out = np.zeros((rows, len(label_space)), dtype=np.float64)
    for held_out in np.array_split(order, folds):
        fitted_on = np.setdiff1d(order, held_out)
        fold_weight = None if sample_weight is None else sample_weight[fitted_on]
        head = MultiLabelHead(label_space).fit(features[fitted_on], target_matrix[fitted_on], sample_weight=fold_weight)
        out[held_out] = head.predict_proba(features[held_out])
    return out
