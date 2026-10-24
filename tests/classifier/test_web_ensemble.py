"""A two-model risk ensemble as one browser weights file (ADR D75).

The risk head is already "the mean over calibrated members", so the 50/50 mix of two heads
with the same member count and the same input features is their members in one list: the
browser and `WebScorer` score it unchanged. The tactic head and label space come from the
partner (the retrain); the base keeps the alert line calm.
"""

from __future__ import annotations

import copy
import json

import numpy as np
import pytest

from qorgan.classifier.linear_train import train_linear
from qorgan.classifier.web_bundle import WEB_BUNDLE_FILENAME, WebScorer, export_web_bundle, load_linear_from_web
from qorgan.classifier.web_ensemble import EnsembleMismatchError, fuse_risk_heads
from qorgan.data.schema import Dialogue, Label, TacticTag, Utterance, spans_from_phrases

_THRESHOLDS = {"risk": 0.59, "enter": 0.59, "exit": 0.49}


def _d(did, text, *, risk, tags=(), phrases=()):
    return Dialogue(
        id=did, language="ru", utterances=(Utterance(speaker="caller", text=text),),
        label=Label(risk=risk, tactic_tags=tuple(TacticTag(id=t) for t in tags), trigger_spans=spans_from_phrases(phrases, text)),
    )


def _corpus(scam_word: str):
    scams = [_d(f"s{i}", f"Продиктуйте код из SMS {scam_word} {i}", risk=0.9, tags=["otp_request", "urgency"], phrases=["код из SMS"]) for i in range(10)]
    legit = [_d(f"n{i}", f"Обычный разговор про погоду {i}", risk=0.03) for i in range(10)]
    return scams + legit


@pytest.fixture()
def base(fake_embedder):
    bundle = train_linear(_corpus("сейчас"), label_space=("otp_request", "urgency"), embedder=fake_embedder, hard_signal=True)
    return export_web_bundle(bundle, thresholds=_THRESHOLDS)


@pytest.fixture()
def partner(fake_embedder):
    bundle = train_linear(_corpus("срочно"), label_space=("otp_request", "urgency", "safe_account"), embedder=fake_embedder, hard_signal=True)
    return export_web_bundle(bundle, thresholds=_THRESHOLDS)


def _rows(scorer: WebScorer, n: int = 8) -> np.ndarray:
    return np.random.default_rng(7).normal(size=(n, scorer.feature_dim))


def test_fused_risk_is_the_equal_mix_of_both_heads(base, partner):
    fused = WebScorer.from_dict(fuse_risk_heads(base, partner, sources={"base": "a", "partner": "b"}))
    a, b = WebScorer.from_dict(base), WebScorer.from_dict(partner)
    x = _rows(a)
    np.testing.assert_allclose(fused.risk_proba(x), 0.5 * a.risk_proba(x) + 0.5 * b.risk_proba(x), rtol=0, atol=1e-12)


def test_tactic_head_and_label_space_come_from_the_partner(base, partner):
    fused = fuse_risk_heads(base, partner, sources={"base": "a", "partner": "b"})
    assert fused["label_space"] == partner["label_space"]
    assert fused["tactic_head"] == partner["tactic_head"]
    assert fused["lexicon"] == partner["lexicon"]


def test_records_its_sources_and_weights(base, partner):
    fused = fuse_risk_heads(base, partner, sources={"base": "sha-a", "partner": "sha-b"})
    assert fused["risk_ensemble"] == {"weights": {"base": 0.5, "partner": 0.5},
                                      "members": {"base": 3, "partner": 3},
                                      "sources": {"base": "sha-a", "partner": "sha-b"}}


def test_inputs_are_not_mutated(base, partner):
    before = copy.deepcopy((base, partner))
    fuse_risk_heads(base, partner, sources={"base": "a", "partner": "b"})
    assert (base, partner) == before


def test_the_fused_file_serves(base, partner, tmp_path):
    path = tmp_path / WEB_BUNDLE_FILENAME
    path.write_text(json.dumps(fuse_risk_heads(base, partner, sources={"base": "a", "partner": "b"}), ensure_ascii=False), encoding="utf-8")
    served = load_linear_from_web(path)
    assert served.label_space == tuple(partner["label_space"])


@pytest.mark.parametrize("field, value", [
    ("feature_order", ["embedding"]),
    ("embed_backend", "sentence-transformers"),
    ("embed_model_name", "intfloat/multilingual-e5-small"),
    ("cue_matcher_version", 1),
    ("feature_version", "hybrid-v0"),
])
def test_refuses_heads_with_different_inputs(base, partner, field, value):
    other = {**partner, field: value}
    with pytest.raises(EnsembleMismatchError, match=field):
        fuse_risk_heads(base, other, sources={"base": "a", "partner": "b"})


def test_refuses_a_different_lexicon(base, partner):
    lexicon = {**partner["lexicon"], "cue_lexicon_hash": "different"}
    with pytest.raises(EnsembleMismatchError, match="lexicon"):
        fuse_risk_heads(base, {**partner, "lexicon": lexicon}, sources={"base": "a", "partner": "b"})


def test_refuses_unequal_member_counts(base, partner):
    head = {**partner["risk_head"], "members": partner["risk_head"]["members"][:2]}
    with pytest.raises(EnsembleMismatchError, match="member"):
        fuse_risk_heads(base, {**partner, "risk_head": head}, sources={"base": "a", "partner": "b"})


def test_refuses_to_fuse_an_ensemble_again(base, partner):
    fused = fuse_risk_heads(base, partner, sources={"base": "a", "partner": "b"})
    with pytest.raises(EnsembleMismatchError, match="already"):
        fuse_risk_heads(fused, partner, sources={"base": "a", "partner": "b"})
