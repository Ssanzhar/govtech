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

# Feature-schema version stamped into the bundle; bump when the hybrid layout changes.
_FEATURE_VERSION = "hybrid-v1"


class LinearFeatureMismatchError(RuntimeError):
    """Raised when a hybrid bundle is loaded against a cue lexicon / reassurance patterns whose
    content hash differs from what it was trained with (feature layout would silently drift)."""


@dataclass(frozen=True)
class LinearBundle:
    """A trained `linear` classifier (heads + metadata); the embedder is loaded separately.

    When `hard_signal_enabled`, the risk head consumes the hybrid vector
    `[embedding | hard-signal cues | reassurance]` and the bundle carries the resolved
    `lexicon` / `reassurance_patterns` (+ their content hashes) so inference rebuilds the
    exact same feature layout. The tactic head is always embedding-only.
    """

    risk_clf: Any
    tactic_clf: MultiLabelHead
    label_space: tuple[str, ...]
    embed_model_name: str
    tactic_threshold: float
    hard_signal_enabled: bool = False
    lexicon: Any = None
    reassurance_patterns: Any = None
    feature_version: str = _FEATURE_VERSION
    cue_lexicon_hash: str = ""
    reassurance_hash: str = ""


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
    hard_signal: bool = False,
    lexicon: Any = None,
    reassurance_patterns: Any = None,
) -> LinearBundle:
    """Embed transcripts, fit the calibrated risk head + multi-label tactic head.

    With `hard_signal=True` the risk head is trained on the hybrid vector
    `[embedding | cue features | reassurance]` (the tactic head stays embedding-only), and the
    resolved lexicon/patterns + content hashes are stored on the bundle. `hard_signal=False`
    (default) preserves the original embedding-only path for backward compatibility.
    """
    if not train_dialogues:
        raise ValueError("train_dialogues must not be empty")
    texts = [d.transcript() for d in train_dialogues]
    y_risk, tactic_matrix = _targets(train_dialogues, label_space)
    embed_model = model_name or get_config().embed_model_name

    if not hard_signal:
        features = embed.embed_texts(texts, embedder=embedder, model_name=model_name)
        risk_clf = _fit_risk_head(features, y_risk)
        tactic_clf = MultiLabelHead(label_space).fit(features, tactic_matrix)
        return LinearBundle(risk_clf, tactic_clf, tuple(label_space), embed_model, tactic_threshold)

    from qorgan.classifier import features as feat
    from qorgan.classifier.cue_lexicon import load_cue_lexicon, lexicon_hash
    from qorgan.classifier.reassurance import load_reassurance_patterns
    from qorgan.classifier.reassurance import reassurance_hash as _reassurance_hash

    lex = lexicon or load_cue_lexicon()
    patterns = reassurance_patterns or load_reassurance_patterns()
    blocks = feat.compute_feature_blocks(
        texts, embedder=embedder, model_name=model_name, lexicon=lex, reassurance_patterns=patterns
    )
    risk_clf = _fit_risk_head(feat.hybrid_matrix(blocks), y_risk)
    tactic_clf = MultiLabelHead(label_space).fit(blocks.embedding, tactic_matrix)
    return LinearBundle(
        risk_clf,
        tactic_clf,
        tuple(label_space),
        embed_model,
        tactic_threshold,
        hard_signal_enabled=True,
        lexicon=lex,
        reassurance_patterns=patterns,
        cue_lexicon_hash=lexicon_hash(lex),
        reassurance_hash=_reassurance_hash(patterns),
    )


def export_linear(bundle: LinearBundle, out_dir: Path) -> dict:
    """Persist both heads (joblib) + metadata.json; return the metadata."""
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle.risk_clf, out_dir / _RISK_CLF_FILE)
    joblib.dump(bundle.tactic_clf, out_dir / _TACTIC_CLF_FILE)
    metadata = {
        "embed_model_name": bundle.embed_model_name,
        "label_space": list(bundle.label_space),
        "tactic_threshold": bundle.tactic_threshold,
        "hard_signal_enabled": bundle.hard_signal_enabled,
        "feature_version": bundle.feature_version,
        "cue_lexicon_hash": bundle.cue_lexicon_hash,
        "reassurance_hash": bundle.reassurance_hash,
    }
    (out_dir / _METADATA_FILE).write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def load_linear(model_dir: Path) -> LinearBundle:
    """Reconstruct a `LinearBundle` from an export dir (raises if absent).

    For a hybrid bundle (`hard_signal_enabled`), reloads the cue lexicon + reassurance patterns
    and raises `LinearFeatureMismatchError` if either's content hash differs from training --
    the guard against a silent feature-layout drift. Legacy metadata (no flag) loads as
    embedding-only for backward compatibility.
    """
    metadata_path = model_dir / _METADATA_FILE
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"No exported linear model in {model_dir}. Train one with "
            "`python -m qorgan.classifier.linear_train`."
        )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    hard_signal_enabled = bool(metadata.get("hard_signal_enabled", False))
    cue_hash = metadata.get("cue_lexicon_hash", "")
    reass_hash = metadata.get("reassurance_hash", "")

    lexicon = patterns = None
    if hard_signal_enabled:
        from qorgan.classifier.cue_lexicon import load_cue_lexicon, lexicon_hash
        from qorgan.classifier.reassurance import load_reassurance_patterns
        from qorgan.classifier.reassurance import reassurance_hash as _reassurance_hash

        lexicon = load_cue_lexicon()
        patterns = load_reassurance_patterns()
        if lexicon_hash(lexicon) != cue_hash:
            raise LinearFeatureMismatchError(
                f"Cue lexicon changed since {model_dir} was trained (hash mismatch); retrain with "
                "`python -m qorgan.classifier.linear_train`."
            )
        if _reassurance_hash(patterns) != reass_hash:
            raise LinearFeatureMismatchError(
                f"Reassurance patterns changed since {model_dir} was trained (hash mismatch); retrain."
            )

    return LinearBundle(
        risk_clf=joblib.load(model_dir / _RISK_CLF_FILE),
        tactic_clf=joblib.load(model_dir / _TACTIC_CLF_FILE),
        label_space=tuple(metadata["label_space"]),
        embed_model_name=metadata["embed_model_name"],
        tactic_threshold=metadata["tactic_threshold"],
        hard_signal_enabled=hard_signal_enabled,
        lexicon=lexicon,
        reassurance_patterns=patterns,
        feature_version=metadata.get("feature_version", _FEATURE_VERSION),
        cue_lexicon_hash=cue_hash,
        reassurance_hash=reass_hash,
    )


def train_and_export(
    train_dialogues: Sequence[Dialogue],
    *,
    label_space: Sequence[str],
    out_dir: Path,
    embedder: Any = None,
    model_name: str | None = None,
    tactic_threshold: float = _DEFAULT_TACTIC_THRESHOLD,
    hard_signal: bool = False,
) -> dict:
    """Train then export; return the metadata."""
    bundle = train_linear(
        train_dialogues,
        label_space=label_space,
        embedder=embedder,
        model_name=model_name,
        tactic_threshold=tactic_threshold,
        hard_signal=hard_signal,
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
    parser.add_argument(
        "--no-hard-signal", action="store_true",
        help="Train the legacy embedding-only risk head (default: hybrid cue+reassurance features)",
    )
    args = parser.parse_args(argv)

    metadata = train_and_export(
        load_split(args.processed_dir, "train"),
        label_space=labels.default_label_space(),
        out_dir=args.out_dir,
        model_name=args.model_name,
        hard_signal=not args.no_hard_signal,
    )
    print(json.dumps(metadata, indent=2))
    print(f"exported -> {args.out_dir}")


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
