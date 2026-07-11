"""Train + export the `linear` scam classifier: frozen multilingual embeddings ->
class-weighted, calibrated Logistic Regression (risk head) + robust multi-label LR (tactic
head). This is the offline model that ships in place of the collapsed XLM-R fine-tune
(`docs/eval_report.md`): it learns on a few hundred examples, trains in seconds on CPU, and
is transparent + calibrated.

The embedder is injected so smoke tests run offline with a fake; the CLI uses the real
`config.embed_model_name`. Both heads guard against degenerate inputs (too-few samples to
calibrate, single-class tactic columns) so training never crashes on small or skewed data.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression

from qorgan.classifier import embed, labels
from qorgan.classifier.multilabel import MultiLabelHead  # re-exported: keep import path stable
from qorgan.config import get_config
from qorgan.data.schema import Dialogue

__all__ = ["MultiLabelHead", "LinearBundle", "train_linear", "train_and_export", "load_linear", "export_linear"]

# Labeled risk >= this is a scam in the binary target (matches eval/run.py).
_TRUTH_THRESHOLD = 0.5
_DEFAULT_TACTIC_THRESHOLD = 0.5
_MAX_CALIBRATION_FOLDS = 3
_LR_MAX_ITER = 2000
_RISK_C = 4.0

_RISK_CLF_FILE = "risk_clf.joblib"
_TACTIC_CLF_FILE = "tactic_clf.joblib"
_METADATA_FILE = "metadata.json"


@dataclass(frozen=True)
class LinearBundle:
    """A trained `linear` classifier (heads + metadata); the embedder is loaded separately."""

    risk_clf: Any
    tactic_clf: MultiLabelHead
    label_space: tuple[str, ...]
    embed_model_name: str
    tactic_threshold: float


def _targets(dialogues: Sequence[Dialogue], label_space: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
    y_risk = np.array([1 if d.label.risk >= _TRUTH_THRESHOLD else 0 for d in dialogues], dtype=int)
    tactic_matrix = np.array(
        [labels.encode_tactics([t.id for t in d.label.tactic_tags], label_space) for d in dialogues],
        dtype=float,
    )
    return y_risk, tactic_matrix


def _fit_risk_head(features: np.ndarray, y_risk: np.ndarray) -> Any:
    """Class-weighted LR calibrated with as many folds as the smaller class allows; falls
    back to an uncalibrated LR when there are too few samples to calibrate."""
    base = LogisticRegression(class_weight="balanced", max_iter=_LR_MAX_ITER, C=_RISK_C)
    min_class = int(np.bincount(y_risk, minlength=2).min())
    if min_class >= 2:
        folds = min(_MAX_CALIBRATION_FOLDS, min_class)
        return CalibratedClassifierCV(base, method="sigmoid", cv=folds).fit(features, y_risk)
    return base.fit(features, y_risk)


def train_linear(
    train_dialogues: Sequence[Dialogue],
    *,
    label_space: Sequence[str],
    embedder: Any = None,
    model_name: str | None = None,
    tactic_threshold: float = _DEFAULT_TACTIC_THRESHOLD,
) -> LinearBundle:
    """Embed transcripts, fit the calibrated risk head + multi-label tactic head."""
    if not train_dialogues:
        raise ValueError("train_dialogues must not be empty")
    features = embed.embed_texts(
        [d.transcript() for d in train_dialogues], embedder=embedder, model_name=model_name
    )
    y_risk, tactic_matrix = _targets(train_dialogues, label_space)
    risk_clf = _fit_risk_head(features, y_risk)
    tactic_clf = MultiLabelHead(label_space).fit(features, tactic_matrix)
    embed_model = model_name or get_config().embed_model_name
    return LinearBundle(risk_clf, tactic_clf, tuple(label_space), embed_model, tactic_threshold)


def export_linear(bundle: LinearBundle, out_dir: Path) -> dict:
    """Persist both heads (joblib) + metadata.json; return the metadata."""
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle.risk_clf, out_dir / _RISK_CLF_FILE)
    joblib.dump(bundle.tactic_clf, out_dir / _TACTIC_CLF_FILE)
    metadata = {
        "embed_model_name": bundle.embed_model_name,
        "label_space": list(bundle.label_space),
        "tactic_threshold": bundle.tactic_threshold,
    }
    (out_dir / _METADATA_FILE).write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def load_linear(model_dir: Path) -> LinearBundle:
    """Reconstruct a `LinearBundle` from an export dir (raises if absent)."""
    metadata_path = model_dir / _METADATA_FILE
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"No exported linear model in {model_dir}. Train one with "
            "`python -m qorgan.classifier.linear_train`."
        )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return LinearBundle(
        risk_clf=joblib.load(model_dir / _RISK_CLF_FILE),
        tactic_clf=joblib.load(model_dir / _TACTIC_CLF_FILE),
        label_space=tuple(metadata["label_space"]),
        embed_model_name=metadata["embed_model_name"],
        tactic_threshold=metadata["tactic_threshold"],
    )


def train_and_export(
    train_dialogues: Sequence[Dialogue],
    *,
    label_space: Sequence[str],
    out_dir: Path,
    embedder: Any = None,
    model_name: str | None = None,
    tactic_threshold: float = _DEFAULT_TACTIC_THRESHOLD,
) -> dict:
    """Train then export; return the metadata."""
    bundle = train_linear(
        train_dialogues,
        label_space=label_space,
        embedder=embedder,
        model_name=model_name,
        tactic_threshold=tactic_threshold,
    )
    return export_linear(bundle, out_dir)


def main(argv: Sequence[str] | None = None) -> None:  # pragma: no cover - CLI (real embeddings)
    """CLI: `python -m qorgan.classifier.linear_train` -- train on the processed splits."""
    from qorgan.eval.run import load_split

    cfg = get_config()
    parser = argparse.ArgumentParser(description="Train + export the linear scam classifier.")
    parser.add_argument("--processed-dir", type=Path, default=cfg.data_dir / "processed")
    parser.add_argument("--out-dir", type=Path, default=cfg.linear_model_dir)
    parser.add_argument("--model-name", default=None, help="Override the embedding model")
    args = parser.parse_args(argv)

    metadata = train_and_export(
        load_split(args.processed_dir, "train"),
        label_space=labels.default_label_space(),
        out_dir=args.out_dir,
        model_name=args.model_name,
    )
    print(json.dumps(metadata, indent=2))
    print(f"exported -> {args.out_dir}")


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
