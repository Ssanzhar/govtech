"""SMOKE tests for `qorgan.classifier.linear_train` — fake embedder, offline, deterministic."""

import json

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
