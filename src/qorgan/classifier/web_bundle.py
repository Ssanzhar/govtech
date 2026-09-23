"""Export the `linear` bundle as plain JSON for on-device inference (PLAN_2026-09 B1).

A browser cannot unpickle sklearn, so the heads are flattened to arrays:

- risk head: `CalibratedClassifierCV(LogisticRegression, method="sigmoid")` -- one member
  per CV fold, each `(coef, intercept, calib_a, calib_b)`. Probability = mean over members
  of `sigmoid(-(a * (coef . x + intercept) + b))` (sklearn's `_SigmoidCalibration`).
- tactic head: one `LogisticRegression` per tactic over the embedding only;
  `sigmoid(coef . e + intercept)`; tactics with no model are absent.
- feature order: `["embedding", "cue:<tactic>"..., "reassurance"]` (the hybrid layout), and
  the *content* of both lexicons so the client can never drift from the weights.

`WebScorer` is the Python reference implementation over that JSON. The JS port
(`site/core/`) must match it on the golden fixtures; it must match sklearn here.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

from qorgan.classifier.cue_match import MATCHER_VERSION
from typing import Any

import numpy as np

from qorgan.classifier.features import hard_signal_feature_ids

WEB_BUNDLE_FILENAME = "weights.json"
FORMAT_VERSION = 1
_RISK_HEAD_TYPE = "calibrated_lr_sigmoid_mean"
_FEATURE_EMBEDDING = "embedding"
_FEATURE_REASSURANCE = "reassurance"
_CUE_PREFIX = "cue:"


def feature_order(hard_signal_enabled: bool) -> list[str]:
    if not hard_signal_enabled:
        return [_FEATURE_EMBEDDING]
    return [_FEATURE_EMBEDDING, *(f"{_CUE_PREFIX}{tid}" for tid in hard_signal_feature_ids()), _FEATURE_REASSURANCE]


def export_web_bundle(bundle: Any, *, thresholds: Mapping[str, float]) -> dict[str, Any]:
    """Flatten a trained `LinearBundle` into a JSON-serialisable dict."""
    members = [_risk_member(cc) for cc in bundle.risk_clf.calibrated_classifiers_]
    first_dim = len(members[0]["coef"])
    extra = len(feature_order(bundle.hard_signal_enabled)) - 1
    return {
        "format_version": FORMAT_VERSION,
        "embed_model_name": bundle.embed_model_name,
        "embed_backend": bundle.embed_backend,
        "embedding_dim": first_dim - extra,
        "feature_order": feature_order(bundle.hard_signal_enabled),
        "feature_version": bundle.feature_version,
        # The cue block is a model input, so the browser must refuse weights trained under a
        # different matcher exactly as `load_linear` does server-side (review, 2026-09-23).
        "cue_matcher_version": MATCHER_VERSION,
        "risk_head": {"type": _RISK_HEAD_TYPE, "members": members},
        "tactic_head": {
            "threshold": float(bundle.tactic_threshold),
            "thresholds": {k: float(v) for k, v in getattr(bundle, "tactic_thresholds", {}).items()},
            "models": {
                tactic_id: {"coef": _floats(model.coef_[0]), "intercept": float(model.intercept_[0])}
                for tactic_id, model in bundle.tactic_clf.models.items()
                if model is not None
            },
        },
        "label_space": list(bundle.label_space),
        "thresholds": {k: float(v) for k, v in thresholds.items()},
        "lexicon": _lexicon_block(bundle) if bundle.hard_signal_enabled else None,
    }


def _risk_member(calibrated: Any) -> dict[str, Any]:
    estimator = calibrated.estimator
    (calibrator,) = calibrated.calibrators  # binary: one sigmoid over the decision function
    return {
        "coef": _floats(estimator.coef_[0]),
        "intercept": float(estimator.intercept_[0]),
        "calib_a": float(calibrator.a_),
        "calib_b": float(calibrator.b_),
    }


def _lexicon_block(bundle: Any) -> dict[str, Any]:
    patterns = bundle.reassurance_patterns
    return {
        "cue_lexicon_hash": bundle.cue_lexicon_hash,
        "reassurance_hash": bundle.reassurance_hash,
        "cues": {tid: list(cues) for tid, cues in bundle.lexicon.entries.items()},
        "reassurance": {
            "sensitive_terms": list(patterns.sensitive_terms),
            "reassurance_terms": list(patterns.reassurance_terms),
            "window_chars": int(patterns.window_chars),
        },
    }


def _floats(values: Any) -> list[float]:
    return [float(v) for v in np.asarray(values).ravel()]


def write_web_bundle(bundle: Any, path: Path, *, thresholds: Mapping[str, float]) -> Path:
    """Write the JSON deterministically (sorted keys, fixed separators)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = export_web_bundle(bundle, thresholds=thresholds)
    path.write_text(json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False), encoding="utf-8")
    return path


class WebScorer:
    """Reference scorer over the exported JSON -- numpy only, no sklearn."""

    def __init__(self, data: Mapping[str, Any]) -> None:
        if data.get("format_version") != FORMAT_VERSION:
            raise ValueError(f"unsupported web bundle format: {data.get('format_version')!r}")
        head = data["risk_head"]
        if head["type"] != _RISK_HEAD_TYPE:
            raise ValueError(f"unsupported risk head type: {head['type']!r}")
        self._coefs = np.asarray([m["coef"] for m in head["members"]], dtype=np.float64)
        self._intercepts = np.asarray([m["intercept"] for m in head["members"]], dtype=np.float64)
        self._calib_a = np.asarray([m["calib_a"] for m in head["members"]], dtype=np.float64)
        self._calib_b = np.asarray([m["calib_b"] for m in head["members"]], dtype=np.float64)
        self.label_space: tuple[str, ...] = tuple(data["label_space"])
        self.tactic_threshold = float(data["tactic_head"]["threshold"])
        self.tactic_thresholds = {str(k): float(v) for k, v in data["tactic_head"].get("thresholds", {}).items()}
        self._tactics = {tid: (np.asarray(m["coef"], dtype=np.float64), float(m["intercept"])) for tid, m in data["tactic_head"]["models"].items()}
        self.feature_order: tuple[str, ...] = tuple(data["feature_order"])
        self.embedding_dim = int(data["embedding_dim"])

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "WebScorer":
        return cls(data)

    @property
    def feature_dim(self) -> int:
        return int(self._coefs.shape[1])

    def risk_proba(self, features: np.ndarray) -> np.ndarray:
        """`(n,)` calibrated scam probability for hybrid feature rows."""
        rows = np.asarray(features, dtype=np.float64)
        if rows.ndim != 2 or rows.shape[1] != self.feature_dim:
            raise ValueError(f"expected (n, {self.feature_dim}) features, got {rows.shape}")
        decision = rows @ self._coefs.T + self._intercepts  # (n, members)
        calibrated = _sigmoid(-(self._calib_a * decision + self._calib_b))
        return calibrated.mean(axis=1)

    def tactic_proba(self, embeddings: np.ndarray) -> np.ndarray:
        """`(n, len(label_space))` per-tactic probabilities; absent models score 0."""
        rows = np.asarray(embeddings, dtype=np.float64)
        if rows.ndim != 2 or rows.shape[1] != self.embedding_dim:
            raise ValueError(f"expected (n, {self.embedding_dim}) embeddings, got {rows.shape}")
        out = np.zeros((rows.shape[0], len(self.label_space)), dtype=np.float64)
        for column, tactic_id in enumerate(self.label_space):
            model = self._tactics.get(tactic_id)
            if model is not None:
                coef, intercept = model
                out[:, column] = _sigmoid(rows @ coef + intercept)
        return out


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def main(argv: list[str] | None = None) -> None:  # pragma: no cover - CLI
    """`python -m qorgan.classifier.web_bundle [--model-dir DIR] [--out PATH]` -- export the
    trained bundle (default: the configured one) as `web/weights.json`."""
    import argparse

    from qorgan.classifier.linear_train import load_linear
    from qorgan.config import get_config

    cfg = get_config()
    parser = argparse.ArgumentParser(description="Export the linear bundle for on-device inference.")
    parser.add_argument("--model-dir", type=Path, default=cfg.linear_model_dir)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    bundle = load_linear(args.model_dir)
    out = args.out or args.model_dir / "web" / WEB_BUNDLE_FILENAME
    path = write_web_bundle(
        bundle, out,
        thresholds={"risk": cfg.risk_threshold, "enter": cfg.risk_threshold_enter, "exit": cfg.risk_threshold_exit},
    )
    print(f"exported -> {path} ({path.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":  # pragma: no cover
    main()
