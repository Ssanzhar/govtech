"""TDD tests for `qorgan.classifier.web_bundle` -- the on-device export (PLAN_2026-09 B1).

The browser cannot unpickle sklearn. Everything the client needs is exported as plain
arrays in one JSON: the calibrated risk head (per-fold LR + Platt sigmoid, averaged), the
per-tactic LR head, feature order, thresholds, and the *content* of both lexicons (so the
client can never drift from the weights it ships with). `WebScorer` is the Python reference
re-implementation from that JSON; the JS port must match it, and it must match sklearn.
"""

import json

import numpy as np
import pytest

from qorgan.classifier.features import compute_feature_blocks, hybrid_matrix
from qorgan.classifier.linear_train import train_and_export, train_linear
from qorgan.classifier.web_bundle import (
    WEB_BUNDLE_FILENAME,
    WebScorer,
    export_web_bundle,
    write_web_bundle,
)
from qorgan.data.schema import Dialogue, Label, TacticTag, Utterance, spans_from_phrases

_LABEL_SPACE = ("otp_request", "urgency", "safe_account")
_THRESHOLDS = {"risk": 0.55, "enter": 0.55, "exit": 0.45}


def _d(did, text, *, risk, tags=(), phrases=()):
    return Dialogue(
        id=did, language="ru", utterances=(Utterance(speaker="caller", text=text),),
        label=Label(risk=risk, tactic_tags=tuple(TacticTag(id=t) for t in tags), trigger_spans=spans_from_phrases(phrases, text)),
    )


def _corpus():
    scams = [_d(f"s{i}", f"Продиктуйте код из SMS номер {i}", risk=0.9, tags=["otp_request", "urgency"], phrases=["код из SMS"]) for i in range(10)]
    legit = [_d(f"n{i}", f"Обычный разговор про погоду {i}", risk=0.03) for i in range(10)]
    return scams + legit


@pytest.fixture()
def hybrid_bundle(fake_embedder):
    return train_linear(_corpus(), label_space=_LABEL_SPACE, embedder=fake_embedder, hard_signal=True)


@pytest.fixture()
def embed_only_bundle(fake_embedder):
    return train_linear(_corpus(), label_space=_LABEL_SPACE, embedder=fake_embedder)


# --- export shape ---------------------------------------------------------------------------


def test_export_is_plain_json_with_the_documented_layout(hybrid_bundle):
    data = export_web_bundle(hybrid_bundle, thresholds=_THRESHOLDS)
    json.dumps(data)  # nothing numpy-ish leaks
    assert data["format_version"] == 1
    assert data["embed_model_name"] == hybrid_bundle.embed_model_name
    assert data["embedding_dim"] == 16
    assert data["feature_order"][0] == "embedding" and data["feature_order"][-1] == "reassurance"
    assert len(data["feature_order"]) == 1 + 5 + 1
    members = data["risk_head"]["members"]
    assert data["risk_head"]["type"] == "calibrated_lr_sigmoid_mean" and len(members) >= 2
    for m in members:
        assert len(m["coef"]) == 16 + 5 + 1
        assert set(m) == {"coef", "intercept", "calib_a", "calib_b"}
    assert data["tactic_head"]["threshold"] == hybrid_bundle.tactic_threshold
    assert set(data["tactic_head"]["models"]) <= set(_LABEL_SPACE)
    assert all(len(m["coef"]) == 16 for m in data["tactic_head"]["models"].values())
    assert data["label_space"] == list(_LABEL_SPACE)
    assert data["thresholds"] == _THRESHOLDS


def test_export_embeds_the_lexicon_content_it_was_trained_with(hybrid_bundle):
    data = export_web_bundle(hybrid_bundle, thresholds=_THRESHOLDS)
    lex = data["lexicon"]
    assert lex["cue_lexicon_hash"] == hybrid_bundle.cue_lexicon_hash
    assert lex["reassurance_hash"] == hybrid_bundle.reassurance_hash
    assert set(lex["cues"]) == set(hybrid_bundle.lexicon.entries)
    assert lex["reassurance"]["window_chars"] == hybrid_bundle.reassurance_patterns.window_chars
    assert lex["reassurance"]["sensitive_terms"] and lex["reassurance"]["reassurance_terms"]


def test_embed_only_bundle_exports_without_lexicon_features(embed_only_bundle):
    data = export_web_bundle(embed_only_bundle, thresholds=_THRESHOLDS)
    assert data["feature_order"] == ["embedding"]
    assert all(len(m["coef"]) == 16 for m in data["risk_head"]["members"])
    assert data["lexicon"] is None


# --- parity: JSON reference == sklearn --------------------------------------------------------


def _features(bundle, texts, embedder):
    blocks = compute_feature_blocks(
        texts, embedder=embedder, lexicon=bundle.lexicon, reassurance_patterns=bundle.reassurance_patterns
    )
    return hybrid_matrix(blocks), blocks.embedding


def test_reference_scorer_matches_sklearn_on_real_feature_rows(hybrid_bundle, fake_embedder):
    texts = [
        "Продиктуйте код из SMS сейчас",
        "Как дела на выходных",
        "Код называть не нужно, это служба банка",
        "Переведите деньги на безопасный счёт, никому не говорите",
    ]
    hybrid, embedding = _features(hybrid_bundle, texts, fake_embedder)
    scorer = WebScorer.from_dict(export_web_bundle(hybrid_bundle, thresholds=_THRESHOLDS))

    expected_risk = hybrid_bundle.risk_clf.predict_proba(hybrid)[:, 1]
    np.testing.assert_allclose(scorer.risk_proba(hybrid), expected_risk, atol=1e-6)
    expected_tactics = hybrid_bundle.tactic_clf.predict_proba(embedding)
    np.testing.assert_allclose(scorer.tactic_proba(embedding), expected_tactics, atol=1e-6)


def test_reference_scorer_matches_sklearn_on_random_rows(hybrid_bundle):
    rng = np.random.default_rng(0)
    rows = rng.normal(size=(200, 16 + 5 + 1)).astype(np.float32)
    scorer = WebScorer.from_dict(export_web_bundle(hybrid_bundle, thresholds=_THRESHOLDS))
    np.testing.assert_allclose(scorer.risk_proba(rows), hybrid_bundle.risk_clf.predict_proba(rows)[:, 1], atol=1e-6)


def test_reference_scorer_round_trips_through_json_text(hybrid_bundle):
    text = json.dumps(export_web_bundle(hybrid_bundle, thresholds=_THRESHOLDS))
    scorer = WebScorer.from_dict(json.loads(text))
    rows = np.zeros((1, 16 + 5 + 1), dtype=np.float32)
    assert 0.0 <= float(scorer.risk_proba(rows)[0]) <= 1.0


def test_reference_scorer_rejects_wrong_width(hybrid_bundle):
    scorer = WebScorer.from_dict(export_web_bundle(hybrid_bundle, thresholds=_THRESHOLDS))
    with pytest.raises(ValueError):
        scorer.risk_proba(np.zeros((1, 16), dtype=np.float32))


# --- integration with the export CLI ----------------------------------------------------------


def test_train_and_export_writes_the_web_bundle_next_to_the_joblibs(tmp_path, fake_embedder):
    out = tmp_path / "linear"
    train_and_export(_corpus(), label_space=_LABEL_SPACE, out_dir=out, embedder=fake_embedder, hard_signal=True)
    path = out / "web" / WEB_BUNDLE_FILENAME
    assert path.exists()
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["risk_head"]["members"] and data["lexicon"]["cues"]


def test_write_web_bundle_is_deterministic(tmp_path, hybrid_bundle):
    a = write_web_bundle(hybrid_bundle, tmp_path / "a.json", thresholds=_THRESHOLDS).read_bytes()
    b = write_web_bundle(hybrid_bundle, tmp_path / "b.json", thresholds=_THRESHOLDS).read_bytes()
    assert a == b
