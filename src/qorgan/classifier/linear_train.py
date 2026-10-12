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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression

from qorgan.classifier import embed, labels
from qorgan.classifier.cue_match import MATCHER_VERSION
from qorgan.classifier.multilabel import MultiLabelHead, out_of_fold_proba  # MultiLabelHead re-exported: keep import path stable
from qorgan.config import V2_MIN_OOF_POSITIVES, V2_NEW_TACTICS, get_config
from qorgan.data.schema import SCAM_RISK_THRESHOLD, Dialogue

__all__ = ["MultiLabelHead", "LinearBundle", "train_linear", "train_and_export", "load_linear", "export_linear"]

# Labeled risk >= this is a scam in the binary target (matches eval/run.py).
_TRUTH_THRESHOLD = SCAM_RISK_THRESHOLD
_DEFAULT_TACTIC_THRESHOLD = 0.5
_MAX_CALIBRATION_FOLDS = 3
_LR_MAX_ITER = 2000
_RISK_C = 4.0
# Per-tactic threshold tuning (ADR D30): out-of-fold train probabilities + val.
_TUNING_FOLDS = 5
_TUNING_SEED = 42

# (trained-on, running-on) pairs allowed to differ: the server's int8 ONNX graph and the
# browser's WASM build of the same graph are cosine-0.98 proxies of each other (ADR D32/D33);
# anything else is a different embedding distribution and is refused.
_PROXY_BACKENDS = frozenset({("device", "onnx"), ("onnx", "device")})
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
    # Per-tactic decision thresholds tuned on out-of-fold train + `val` (ADR D30); empty = flat `tactic_threshold`.
    tactic_thresholds: dict[str, float] = field(default_factory=dict)
    # Which embedder produced the training vectors ("sentence-transformers" fp32 or the int8
    # "onnx" graph the browser ships). Loading under a different backend is refused: the
    # heads are only valid on the embedding distribution they were fitted on (A4).
    embed_backend: str = "sentence-transformers"


def _targets(dialogues: Sequence[Dialogue], label_space: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
    y_risk = np.array([1 if d.label.risk >= _TRUTH_THRESHOLD else 0 for d in dialogues], dtype=int)
    tactic_matrix = np.array(
        [labels.encode_tactics([t.id for t in d.label.tactic_tags], label_space) for d in dialogues],
        dtype=float,
    )
    return y_risk, tactic_matrix


def _fit_risk_head(features: np.ndarray, y_risk: np.ndarray, sample_weight: np.ndarray | None = None) -> Any:
    """Class-weighted LR calibrated with as many folds as the smaller class allows; falls
    back to an uncalibrated LR when there are too few samples to calibrate. `sample_weight`
    reaches both the base estimator and the calibrator."""
    base = LogisticRegression(class_weight="balanced", max_iter=_LR_MAX_ITER, C=_RISK_C)
    min_class = int(np.bincount(y_risk, minlength=2).min())
    if min_class >= 2:
        folds = min(_MAX_CALIBRATION_FOLDS, min_class)
        return CalibratedClassifierCV(base, method="sigmoid", cv=folds).fit(features, y_risk, sample_weight=sample_weight)
    return base.fit(features, y_risk, sample_weight=sample_weight)


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
    tuning_dialogues: Sequence[Dialogue] | None = None,
    sample_weights: Sequence[float] | None = None,
    label_mask: np.ndarray | None = None,
    tuning_mask: np.ndarray | None = None,
) -> LinearBundle:
    """Embed transcripts, fit the calibrated risk head + multi-label tactic head; with
    `tuning_dialogues` (the `val` split) also tune per-tactic decision thresholds on out-of-fold
    train probabilities + `val` (ADR D30). `sample_weights` (optional, one per train dialogue)
    reach both heads, the calibrator and the out-of-fold tuner.

    With `hard_signal=True` the risk head is trained on the hybrid vector
    `[embedding | cue features | reassurance]` (the tactic head stays embedding-only), and the
    resolved lexicon/patterns + content hashes are stored on the bundle. `hard_signal=False`
    (default) preserves the original embedding-only path for backward compatibility.

    `label_mask` / `tuning_mask` (`(n, L)` bool, True = labelled; ADR D63 phase 2) keep rows
    written for an older label space out of the heads of tactics that space did not have.
    """
    if not train_dialogues:
        raise ValueError("train_dialogues must not be empty")
    if sample_weights is not None and len(sample_weights) != len(train_dialogues):
        raise ValueError(f"sample_weights must have one entry per train dialogue ({len(train_dialogues)}), got {len(sample_weights)}")
    weights = None if sample_weights is None else np.asarray(sample_weights, dtype=float)
    texts = [d.transcript() for d in train_dialogues]
    y_risk, tactic_matrix = _targets(train_dialogues, label_space)
    cfg = get_config()
    embed_model = model_name or cfg.embed_model_name

    if not hard_signal:
        features = embed.embed_texts(texts, embedder=embedder, model_name=model_name)
        risk_clf = _fit_risk_head(features, y_risk, sample_weight=weights)
        tactic_clf = MultiLabelHead(label_space).fit(features, tactic_matrix, sample_weight=weights, **_mask_kwargs(label_mask))
        thresholds = _tune_thresholds(
            tactic_clf, label_space, tuning_dialogues, tactic_threshold,
            train_features=features, train_targets=tactic_matrix, train_weights=weights, embedder=embedder, model_name=model_name,
            train_mask=label_mask, tuning_mask=tuning_mask,
        )
        return LinearBundle(
            risk_clf, tactic_clf, tuple(label_space), embed_model, tactic_threshold, embed_backend=cfg.embed_backend,
            tactic_thresholds=thresholds,
        )

    from qorgan.classifier import features as feat
    from qorgan.classifier.cue_lexicon import load_cue_lexicon, lexicon_hash
    from qorgan.classifier.reassurance import load_reassurance_patterns
    from qorgan.classifier.reassurance import reassurance_hash as _reassurance_hash

    lex = lexicon or load_cue_lexicon()
    patterns = reassurance_patterns or load_reassurance_patterns()
    blocks = feat.compute_feature_blocks(
        texts, embedder=embedder, model_name=model_name, lexicon=lex, reassurance_patterns=patterns
    )
    risk_clf = _fit_risk_head(feat.hybrid_matrix(blocks), y_risk, sample_weight=weights)
    tactic_clf = MultiLabelHead(label_space).fit(blocks.embedding, tactic_matrix, sample_weight=weights, **_mask_kwargs(label_mask))
    thresholds = _tune_thresholds(
        tactic_clf, label_space, tuning_dialogues, tactic_threshold,
        train_features=blocks.embedding, train_targets=tactic_matrix, train_weights=weights, embedder=embedder, model_name=model_name,
        train_mask=label_mask, tuning_mask=tuning_mask,
    )
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
        embed_backend=cfg.embed_backend,
        tactic_thresholds=thresholds,
    )


def _tune_thresholds(
    tactic_clf: MultiLabelHead, label_space: Sequence[str], tuning: Sequence[Dialogue] | None, default: float,
    *, train_features: np.ndarray, train_targets: np.ndarray, train_weights: np.ndarray | None, embedder: Any, model_name: str | None,
    train_mask: np.ndarray | None = None, tuning_mask: np.ndarray | None = None,
) -> dict[str, float]:
    """Per-tactic thresholds (ADR D30) from fit-independent probabilities: out-of-fold
    probabilities on the training rows (seeded K-fold of the same head) concatenated with the
    fitted head's probabilities on the tuning split (`val`). `{}` without a tuning split --
    the small `val` split alone starves most tactics of the support the tuner requires."""
    if not tuning:
        return {}
    from qorgan.classifier.calibrate import tune_tactic_thresholds

    tuning_features = embed.embed_texts([d.transcript() for d in tuning], embedder=embedder, model_name=model_name)
    _, tuning_targets = _targets(tuning, label_space)
    probs = np.vstack([
        out_of_fold_proba(train_features, train_targets, label_space, folds=_TUNING_FOLDS, seed=_TUNING_SEED,
                          sample_weight=train_weights, **_mask_kwargs(train_mask)),
        tactic_clf.predict_proba(tuning_features),
    ])
    truth = np.vstack([train_targets, tuning_targets]).astype(int)
    mask = None
    if train_mask is not None or tuning_mask is not None:
        mask = np.vstack([
            train_mask if train_mask is not None else np.ones_like(train_targets, dtype=bool),
            tuning_mask if tuning_mask is not None else np.ones_like(tuning_targets, dtype=bool),
        ]).tolist()
    return tune_tactic_thresholds(probs.tolist(), truth.tolist(), label_space, default=default, mask=mask)


def _mask_kwargs(mask: np.ndarray | None) -> dict:
    """Pass a label mask only when there is one (keeps the v1 call path byte-identical)."""
    return {} if mask is None else {"label_mask": mask}


def provenance_mask(
    dialogues: Sequence[Dialogue], label_space: Sequence[str], generator: dict[str, str], *, new_tactics: Sequence[str], labelled_by: str,
) -> np.ndarray:
    """`(n, L)` bool, True = labelled. A row whose generator is not `labelled_by` (it was written
    for the v1 space) is unlabelled for `new_tactics` (ADR D63 label masking, approved
    2026-10-09). Rows missing from `generator` count as v1 rows."""
    mask = np.ones((len(dialogues), len(label_space)), dtype=bool)
    columns = [j for j, t in enumerate(label_space) if t in set(new_tactics)]
    for i, dialogue in enumerate(dialogues):
        if generator.get(dialogue.id) != labelled_by:
            mask[i, columns] = False
    return mask


def labelled_positives(dialogues: Sequence[Dialogue], label_space: Sequence[str], mask: np.ndarray | None = None) -> dict[str, int]:
    """Positives per tactic over labelled rows only."""
    _, targets = _targets(dialogues, label_space)
    if mask is not None:
        targets = targets * mask
    return {t: int(targets[:, j].sum()) for j, t in enumerate(label_space)}


def thin_heads(train_positives: dict[str, int], tuning_positives: dict[str, int], *, minimum: int) -> tuple[str, ...]:
    """Tactics with fewer than `minimum` out-of-fold positives (train OOF + val): dropped from
    the space rather than shipped at an untuned 0.5 (ADR D63 "thin heads", approved)."""
    return tuple(t for t in train_positives if train_positives[t] + tuning_positives.get(t, 0) < minimum)


def export_linear(bundle: LinearBundle, out_dir: Path) -> dict:
    """Persist both heads (joblib) + metadata.json; return the metadata."""
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle.risk_clf, out_dir / _RISK_CLF_FILE)
    joblib.dump(bundle.tactic_clf, out_dir / _TACTIC_CLF_FILE)
    metadata = {
        "embed_model_name": bundle.embed_model_name,
        "label_space": list(bundle.label_space),
        "tactic_threshold": bundle.tactic_threshold,
        "tactic_thresholds": dict(bundle.tactic_thresholds),
        "hard_signal_enabled": bundle.hard_signal_enabled,
        "feature_version": bundle.feature_version,
        "cue_lexicon_hash": bundle.cue_lexicon_hash,
        "cue_matcher_version": MATCHER_VERSION,
        "reassurance_hash": bundle.reassurance_hash,
        "embed_backend": bundle.embed_backend,
    }
    (out_dir / _METADATA_FILE).write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    # On-device export (PLAN_2026-09 B1): the same heads as plain JSON, next to the joblibs.
    from qorgan.classifier.web_bundle import WEB_BUNDLE_FILENAME, write_web_bundle
    from qorgan.config import get_config

    cfg = get_config()
    write_web_bundle(
        bundle,
        out_dir / "web" / WEB_BUNDLE_FILENAME,
        thresholds={"risk": cfg.risk_threshold, "enter": cfg.risk_threshold_enter, "exit": cfg.risk_threshold_exit},
    )
    return metadata


def check_embed_backend(trained_backend: str, *, source: str) -> None:
    """Heads are only valid on the embedding distribution they were fitted on (A4); the
    device/onnx pair is the documented proxy (ADR D32/D33)."""
    runtime_backend = get_config().embed_backend
    if trained_backend != runtime_backend and (trained_backend, runtime_backend) not in _PROXY_BACKENDS:
        raise LinearFeatureMismatchError(
            f"{source} was trained on {trained_backend!r} embeddings but QORGAN_EMBED_BACKEND is "
            f"{runtime_backend!r}; set the backend to match or retrain."
        )


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
    embed_backend = metadata.get("embed_backend", "sentence-transformers")
    check_embed_backend(embed_backend, source=str(model_dir))
    hard_signal_enabled = bool(metadata.get("hard_signal_enabled", False))
    trained_matcher = metadata.get("cue_matcher_version", 1)
    if hard_signal_enabled and trained_matcher != MATCHER_VERSION:
        raise LinearFeatureMismatchError(
            f"{model_dir} was trained with cue matcher v{trained_matcher} but this build uses "
            f"v{MATCHER_VERSION}; the cue features are computed with it, so retrain "
            "(`python -m qorgan.classifier.linear_train`)."
        )
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
        tactic_thresholds={str(k): float(v) for k, v in metadata.get("tactic_thresholds", {}).items()},
        hard_signal_enabled=hard_signal_enabled,
        lexicon=lexicon,
        reassurance_patterns=patterns,
        feature_version=metadata.get("feature_version", _FEATURE_VERSION),
        cue_lexicon_hash=cue_hash,
        reassurance_hash=reass_hash,
        embed_backend=embed_backend,
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
    tuning_dialogues: Sequence[Dialogue] | None = None,
    sample_weights: Sequence[float] | None = None,
    label_mask: np.ndarray | None = None,
    tuning_mask: np.ndarray | None = None,
) -> dict:
    """Train then export; return the metadata."""
    bundle = train_linear(
        train_dialogues,
        label_space=label_space,
        embedder=embedder,
        model_name=model_name,
        tactic_threshold=tactic_threshold,
        hard_signal=hard_signal,
        tuning_dialogues=tuning_dialogues,
        sample_weights=sample_weights,
        label_mask=label_mask,
        tuning_mask=tuning_mask,
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

    train = load_split(args.processed_dir, "train")
    val = load_split(args.processed_dir, "val")
    label_space = labels.default_label_space()
    train_mask = val_mask = None
    report: dict = {}
    provenance_path = args.processed_dir / "provenance.json"
    if provenance_path.exists() and set(V2_NEW_TACTICS) & set(label_space):
        # Taxonomy v2 (ADR D63 phase 2): mask v1 rows out of the new heads, drop thin heads.
        from qorgan.data.gen.fold import GENERATOR_CLAUDE

        generator = json.loads(provenance_path.read_text(encoding="utf-8"))["generator"]

        def masks(space):
            return (provenance_mask(train, space, generator, new_tactics=V2_NEW_TACTICS, labelled_by=GENERATOR_CLAUDE),
                    provenance_mask(val, space, generator, new_tactics=V2_NEW_TACTICS, labelled_by=GENERATOR_CLAUDE))

        train_mask, val_mask = masks(label_space)
        positives = (labelled_positives(train, label_space, train_mask), labelled_positives(val, label_space, val_mask))
        dropped = thin_heads(*positives, minimum=V2_MIN_OOF_POSITIVES)
        label_space = tuple(t for t in label_space if t not in dropped)
        train_mask, val_mask = masks(label_space)
        report = {"label_masking": {"new_tactics": list(V2_NEW_TACTICS), "masked_train_rows": int((~train_mask).any(axis=1).sum())},
                  "oof_positives": {t: positives[0][t] + positives[1][t] for t in positives[0]},
                  "dropped_thin_heads": list(dropped), "min_oof_positives": V2_MIN_OOF_POSITIVES}
    metadata = train_and_export(
        train,
        label_space=label_space,
        out_dir=args.out_dir,
        model_name=args.model_name,
        hard_signal=not args.no_hard_signal,
        tuning_dialogues=val,  # per-tactic thresholds (ADR D30)
        # No sample weights: the ASR-styled copies count as full rows. Pair-weighting (0.5 + 0.5)
        # was measured and rejected -- it halves the clean register's evidence too (ADR D31).
        label_mask=train_mask,
        tuning_mask=val_mask,
    )
    if report:
        (args.out_dir / "v2_training_report.json").write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=1))
    print(json.dumps(metadata, indent=2))
    print(f"exported -> {args.out_dir}")


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
