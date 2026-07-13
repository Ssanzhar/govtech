"""SMOKE tests for `qorgan.classifier.predict` — backend routing, never hits a network."""

from types import SimpleNamespace

import pytest

from qorgan.classifier import predict
from qorgan.data.demo_transcripts import DEMO_TRANSCRIPTS
from qorgan.data.schema import ScoreResult


def test_score_mock_backend_returns_deterministic_result_for_demo_transcript():
    text = DEMO_TRANSCRIPTS["scam_bank_ru"]
    first = predict.score(text, backend="mock")
    second = predict.score(text, backend="mock")

    assert first == second
    assert first.backend == "mock"
    assert first.risk == pytest.approx(0.96)
    tag_ids = {tag.id for tag in first.tags}
    assert "otp_request" in tag_ids
    assert "safe_account" in tag_ids
    for span in first.attributions:
        assert text[span.start : span.end] == span.text


def test_score_mock_backend_hard_negative_low_risk():
    text = DEMO_TRANSCRIPTS["hard_negative_bank_call_ru"]
    result = predict.score(text, backend="mock")

    assert result.risk < 0.5
    assert result.tags == ()
    assert result.attributions == ()


def test_score_mock_heuristic_fallback_for_arbitrary_text_with_hard_signal():
    text = "Здравствуйте. Продиктуйте код из SMS, чтобы подтвердить операцию."
    result = predict.score(text, backend="mock")

    assert result.backend == "mock"
    assert result.risk >= 0.9
    assert any(tag.id == "otp_request" for tag in result.tags)
    assert result.attributions
    for span in result.attributions:
        assert text[span.start : span.end] == span.text


def test_score_mock_heuristic_fallback_no_match_low_risk():
    text = "Привет, как дела? Давай встретимся в кафе завтра вечером."
    result = predict.score(text, backend="mock")

    assert result.risk < 0.2
    assert result.tags == ()
    assert result.attributions == ()


def test_score_llm_backend_routes_to_llm_classifier(monkeypatch):
    sentinel = ScoreResult(risk=0.42, backend="llm")
    calls = []

    def fake_classify(transcript, **kwargs):
        calls.append(transcript)
        return sentinel

    monkeypatch.setattr(predict.llm_classifier, "classify", fake_classify)

    result = predict.score("some transcript", backend="llm")

    assert result is sentinel
    assert calls == ["some transcript"]


def test_score_xlmr_backend_missing_model_raises_clear_error(monkeypatch, tmp_path):
    # Point the bundle loader at an empty dir -> a clear, actionable error (not a crash).
    monkeypatch.setenv("QORGAN_XLMR_MODEL_DIR", str(tmp_path / "no_model"))
    predict._XLMR_BUNDLE_CACHE.clear()
    with pytest.raises(predict.XlmrModelNotFoundError):
        predict.score("Продиктуйте код из SMS", backend="xlmr")


def test_xlmr_score_with_injected_bundle_returns_scoreresult(tiny_encoder, fake_tokenizer):
    import torch

    from qorgan.classifier.model import ScamClassifierModel

    label_space = ("otp_request", "urgency", "safe_account")
    model = ScamClassifierModel(tiny_encoder, num_tactics=len(label_space))
    bundle = predict.XlmrBundle(
        model=model,
        tokenizer=fake_tokenizer,
        label_space=label_space,
        max_length=32,
        tactic_threshold=0.5,
        temperature=1.5,
        device=torch.device("cpu"),
    )
    transcript = "Продиктуйте код из SMS и переведите деньги на безопасный счёт"

    result = predict._xlmr_score(transcript, bundle=bundle)

    assert isinstance(result, ScoreResult)
    assert result.backend == "xlmr"
    assert 0.0 <= result.risk <= 1.0
    assert result.raw_confidence is not None and 0.5 <= result.raw_confidence <= 1.0
    for tag in result.tags:
        assert tag.id in label_space
    for span in result.attributions:
        assert transcript[span.start : span.end] == span.text  # grounded, verbatim


def test_score_unknown_backend_raises_unknown_backend_error():
    with pytest.raises(predict.UnknownBackendError):
        predict.score("some transcript", backend="bogus")


def test_score_linear_backend_missing_model_raises(monkeypatch, tmp_path):
    monkeypatch.setenv("QORGAN_LINEAR_MODEL_DIR", str(tmp_path / "no_linear"))
    predict._LINEAR_BUNDLE_CACHE.clear()
    with pytest.raises(FileNotFoundError):
        predict.score("Продиктуйте код из SMS", backend="linear")


def test_linear_score_with_injected_bundle_returns_scoreresult(fake_embedder):
    from qorgan.classifier.linear_train import train_linear
    from qorgan.data.schema import Dialogue, Label, TacticTag, Utterance

    label_space = ("otp_request", "urgency", "safe_account")

    def d(did, text, risk, tags=()):
        return Dialogue(
            id=did, language="ru", utterances=(Utterance(speaker="c", text=text),),
            label=Label(risk=risk, tactic_tags=tuple(TacticTag(id=t) for t in tags)),
        )

    train = (
        [d(f"s{i}", f"Продиктуйте код из SMS {i}", 0.9, ["otp_request"]) for i in range(10)]
        + [d(f"n{i}", f"Разговор про погоду {i}", 0.03) for i in range(10)]
    )
    bundle = train_linear(train, label_space=label_space, embedder=fake_embedder)

    scam = "Здравствуйте это банк\nПродиктуйте код из SMS сейчас"
    result = predict._linear_score(scam, bundle=bundle, embedder=fake_embedder)

    assert isinstance(result, ScoreResult)
    assert result.backend == "linear"
    assert 0.0 <= result.risk <= 1.0
    assert result.raw_confidence is not None
    for tag in result.tags:
        assert tag.id in label_space
    for span in result.attributions:
        assert scam[span.start : span.end] == span.text  # grounded, verbatim

    legit = predict._linear_score("Как дела на выходных", bundle=bundle, embedder=fake_embedder)
    assert result.risk > legit.risk  # scam scored above a benign line


def test_score_empty_transcript_raises_value_error():
    with pytest.raises(ValueError):
        predict.score("   ", backend="mock")


def test_score_uses_config_default_backend_when_not_specified(monkeypatch):
    fake_cfg = SimpleNamespace(classifier_backend="mock")
    monkeypatch.setattr(predict, "get_config", lambda: fake_cfg)

    result = predict.score(DEMO_TRANSCRIPTS["hard_negative_bank_call_ru"])

    assert result.backend == "mock"


def test_score_backend_override_beats_config_default(monkeypatch):
    fake_cfg = SimpleNamespace(classifier_backend="llm")
    monkeypatch.setattr(predict, "get_config", lambda: fake_cfg)
    monkeypatch.setattr(
        predict.llm_classifier, "classify", lambda transcript, **kwargs: (_ for _ in ()).throw(
            AssertionError("llm backend should not be called when override is 'mock'")
        )
    )

    result = predict.score(DEMO_TRANSCRIPTS["hard_negative_bank_call_ru"], backend="mock")

    assert result.backend == "mock"
