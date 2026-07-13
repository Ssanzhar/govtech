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

    def fit(self, features: np.ndarray, target_matrix: np.ndarray) -> "MultiLabelHead":
        for column, tactic_id in enumerate(self.label_space):
            column_targets = target_matrix[:, column].astype(int)
            if len(set(column_targets.tolist())) >= 2:
                self.models[tactic_id] = LogisticRegression(
                    class_weight="balanced", max_iter=_LR_MAX_ITER
                ).fit(features, column_targets)
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
