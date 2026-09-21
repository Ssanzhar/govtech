"""SMOKE tests for `qorgan.classifier.linear_train` — fake embedder, offline, deterministic."""

import json

import pytest

from qorgan.classifier.embed import embed_texts
from qorgan.classifier.linear_train import (
    LinearBundle,
    MultiLabelHead,
    load_linear,
    train_and_export,
    train_linear,
)
from qorgan.data.schema import Dialogue, Label, TacticTag, Utterance, spans_from_phrases

_LABEL_SPACE = ("otp_request", "urgency", "safe_account")


def _d(did, text, *, risk, tags=(), phrases=()):
    return Dialogue(
        id=did,
        language="ru",
        utterances=(Utterance(speaker="caller", text=text),),
        label=Label(
            risk=risk,
            tactic_tags=tuple(TacticTag(id=t) for t in tags),
            trigger_spans=spans_from_phrases(phrases, text),
        ),
    )


def _corpus():
    scams = [
        _d(f"s{i}", f"Продиктуйте код из SMS номер {i}", risk=0.9, tags=["otp_request", "urgency"], phrases=["код из SMS"])
        for i in range(10)
    ]
    legit = [_d(f"n{i}", f"Обычный разговор про погоду {i}", risk=0.03) for i in range(10)]
    return scams + legit


def test_train_and_export_round_trips(tmp_path, fake_embedder):
    out = tmp_path / "linear"
    meta = train_and_export(_corpus(), label_space=_LABEL_SPACE, out_dir=out, embedder=fake_embedder)

    for f in ("risk_clf.joblib", "tactic_clf.joblib", "metadata.json"):
        assert (out / f).exists()
    assert meta["label_space"] == list(_LABEL_SPACE)
    assert json.loads((out / "metadata.json").read_text())["label_space"] == list(_LABEL_SPACE)

    bundle = load_linear(out)
    assert isinstance(bundle, LinearBundle)
    assert isinstance(bundle.tactic_clf, MultiLabelHead)


def test_risk_head_scores_scam_above_legit(fake_embedder):
    bundle = train_linear(_corpus(), label_space=_LABEL_SPACE, embedder=fake_embedder)
    feats = embed_texts(["Продиктуйте код из SMS сейчас", "Как дела на выходных"], embedder=fake_embedder)
    scam_p = bundle.risk_clf.predict_proba(feats)[0, 1]
    legit_p = bundle.risk_clf.predict_proba(feats)[1, 1]
    assert scam_p > legit_p


def test_tactic_head_detects_present_tactic_and_skips_absent(fake_embedder):
    bundle = train_linear(_corpus(), label_space=_LABEL_SPACE, embedder=fake_embedder)
    feats = embed_texts(["Продиктуйте код из SMS"], embedder=fake_embedder)
    probs = bundle.tactic_clf.predict_proba(feats)[0]
    idx = {t: i for i, t in enumerate(_LABEL_SPACE)}
    assert probs[idx["otp_request"]] > 0.5          # present in scams -> learned
    assert probs[idx["safe_account"]] == 0.0        # all-absent column -> guarded to 0


def test_train_linear_empty_raises(fake_embedder):
    import pytest

    with pytest.raises(ValueError):
        train_linear([], label_space=_LABEL_SPACE, embedder=fake_embedder)


# --- hybrid (hard-signal) path + bundle versioning ---------------------------------------


def test_hybrid_train_export_load_roundtrip(tmp_path, fake_embedder):
    out = tmp_path / "linear_hybrid"
    meta = train_and_export(
        _corpus(), label_space=_LABEL_SPACE, out_dir=out, embedder=fake_embedder, hard_signal=True
    )
    assert meta["hard_signal_enabled"] is True
    assert meta["cue_lexicon_hash"] and meta["reassurance_hash"]

    bundle = load_linear(out)
    assert bundle.hard_signal_enabled is True
    assert bundle.lexicon is not None and bundle.reassurance_patterns is not None


def test_legacy_metadata_loads_as_embed_only(tmp_path, fake_embedder):
    out = tmp_path / "linear_legacy"
    train_and_export(_corpus(), label_space=_LABEL_SPACE, out_dir=out, embedder=fake_embedder)
    meta = json.loads((out / "metadata.json").read_text())
    for key in ("hard_signal_enabled", "feature_version", "cue_lexicon_hash", "reassurance_hash"):
        meta.pop(key, None)  # simulate an old bundle predating the hybrid feature
    (out / "metadata.json").write_text(json.dumps(meta))

    bundle = load_linear(out)
    assert bundle.hard_signal_enabled is False


def test_hybrid_load_raises_on_lexicon_drift(tmp_path, fake_embedder, monkeypatch):
    import pytest

    from qorgan.classifier.linear_train import LinearFeatureMismatchError
    from qorgan.config import get_config

    out = tmp_path / "linear_hybrid_drift"
    train_and_export(
        _corpus(), label_space=_LABEL_SPACE, out_dir=out, embedder=fake_embedder, hard_signal=True
    )
    # Point the cue lexicon at a DIFFERENT (drifted) file for loading -> hash mismatch.
    drifted = tmp_path / "drifted_cues.yaml"
    original = get_config().cue_lexicon_path.read_text(encoding="utf-8")
    drifted.write_text(original + '    - "новая фраза дрейфа"\n', encoding="utf-8")
    monkeypatch.setenv("QORGAN_CUE_LEXICON_PATH", str(drifted))

    with pytest.raises(LinearFeatureMismatchError):
        load_linear(out)


# --- embed backend is part of the bundle contract (PLAN_2026-09 A4) --------------------------


def test_export_records_the_embed_backend_and_load_refuses_a_mismatch(tmp_path, fake_embedder, monkeypatch):
    from qorgan.classifier.linear_train import LinearFeatureMismatchError

    out = tmp_path / "linear"
    monkeypatch.setenv("QORGAN_EMBED_BACKEND", "onnx")
    meta = train_and_export(_corpus(), label_space=_LABEL_SPACE, out_dir=out, embedder=fake_embedder)
    assert meta["embed_backend"] == "onnx"
    assert load_linear(out).embed_backend == "onnx"

    monkeypatch.setenv("QORGAN_EMBED_BACKEND", "sentence-transformers")
    with pytest.raises(LinearFeatureMismatchError):
        load_linear(out)


def test_legacy_bundle_without_embed_backend_loads_as_sentence_transformers(tmp_path, fake_embedder, monkeypatch):
    monkeypatch.setenv("QORGAN_EMBED_BACKEND", "sentence-transformers")
    out = tmp_path / "linear"
    train_and_export(_corpus(), label_space=_LABEL_SPACE, out_dir=out, embedder=fake_embedder)
    meta_path = out / "metadata.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta.pop("embed_backend")
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    assert load_linear(out).embed_backend == "sentence-transformers"


# --- per-tactic thresholds (ADR D30): tuned on out-of-fold train + val ------------------------


def _val():
    return [_d("v0", "Скажите код из SMS быстро", risk=0.9, tags=["otp_request"]), _d("v1", "Поговорим о погоде", risk=0.05)]


def test_out_of_fold_proba_predicts_every_row_from_a_head_that_did_not_see_it(monkeypatch):
    import numpy as np

    from qorgan.classifier import multilabel

    fits, predictions = [], []
    real_fit, real_predict = multilabel.MultiLabelHead.fit, multilabel.MultiLabelHead.predict_proba

    def spy_fit(self, features, targets, sample_weight=None):
        fits.append(features.shape[0])
        return real_fit(self, features, targets, sample_weight=sample_weight)

    def spy_predict(self, features):
        predictions.append(features.shape[0])
        return real_predict(self, features)

    monkeypatch.setattr(multilabel.MultiLabelHead, "fit", spy_fit)
    monkeypatch.setattr(multilabel.MultiLabelHead, "predict_proba", spy_predict)
    rng = np.random.default_rng(0)
    features = rng.normal(size=(20, 8))
    targets = np.array([[1.0, 0.0]] * 10 + [[0.0, 1.0]] * 10)
    first = multilabel.out_of_fold_proba(features, targets, ("a", "b"), folds=5, seed=1)
    assert first.shape == (20, 2)
    assert fits == [16] * 5 and predictions == [4] * 5  # each fold: fit on the rest, predict the held-out rows
    second = multilabel.out_of_fold_proba(features, targets, ("a", "b"), folds=5, seed=1)
    assert np.array_equal(first, second)  # a seeded permutation -- reproducible exports


def test_out_of_fold_proba_rejects_bad_folds():
    import numpy as np

    from qorgan.classifier.multilabel import out_of_fold_proba

    with pytest.raises(ValueError):
        out_of_fold_proba(np.zeros((3, 2)), np.zeros((3, 1)), ("a",), folds=1, seed=0)
    with pytest.raises(ValueError):
        out_of_fold_proba(np.zeros((3, 2)), np.zeros((3, 1)), ("a",), folds=4, seed=0)  # more folds than rows


def test_train_linear_tunes_a_threshold_for_every_tactic_from_train_oof_plus_val(fake_embedder):
    from qorgan.classifier.calibrate import TACTIC_THRESHOLD_GRID

    bundle = train_linear(_corpus(), label_space=_LABEL_SPACE, embedder=fake_embedder, tuning_dialogues=_val())
    assert set(bundle.tactic_thresholds) == set(_LABEL_SPACE)
    allowed = {0.5, *TACTIC_THRESHOLD_GRID}
    assert all(v in allowed for v in bundle.tactic_thresholds.values())
    again = train_linear(_corpus(), label_space=_LABEL_SPACE, embedder=fake_embedder, tuning_dialogues=_val())
    assert again.tactic_thresholds == bundle.tactic_thresholds
    assert train_linear(_corpus(), label_space=_LABEL_SPACE, embedder=fake_embedder).tactic_thresholds == {}


def test_train_positives_count_toward_the_tuning_support(fake_embedder, monkeypatch):
    """`otp_request` has 10 train positives and 1 val positive: with val alone it would sit
    under the support guard and keep the default; out-of-fold train rows make it tunable."""
    from qorgan.classifier import linear_train

    seen = {}

    def spy(probs, truth, label_space, **kwargs):
        seen["positives"] = {tid: sum(int(r[j]) for r in truth) for j, tid in enumerate(label_space)}
        return {tid: 0.5 for tid in label_space}

    monkeypatch.setattr("qorgan.classifier.calibrate.tune_tactic_thresholds", spy)
    linear_train.train_linear(_corpus(), label_space=_LABEL_SPACE, embedder=fake_embedder, tuning_dialogues=_val())
    assert seen["positives"]["otp_request"] == 11 and seen["positives"]["urgency"] == 10

