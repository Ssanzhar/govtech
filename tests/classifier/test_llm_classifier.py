"""SMOKE tests for `qorgan.classifier.llm_classifier` — mocked Gemini client only.

Never calls the network. Asserts parse/validation/routing/caching behavior, not model
quality (CLAUDE SS7 / IMPLEMENTATION_PLAN TDD-vs-smoke policy).
"""

import json
from types import SimpleNamespace

import pytest

from qorgan.classifier.llm_classifier import LLMClassifierError, classify
from qorgan.data.schema import ScoreResult


class FakeModels:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        if len(self._responses) == 1:
            return self._responses[0]
        return self._responses.pop(0)


class FakeClient:
    def __init__(self, responses):
        self.models = FakeModels(responses)


def _tool_use_response(payload: dict, tool_name: str = "classify_scam_call"):
    # Gemini JSON-mode response: the payload is a JSON object in `.text`.
    return SimpleNamespace(text=json.dumps(payload), parsed=None)


TRANSCRIPT = "Это служба безопасности банка. Продиктуйте код из SMS немедленно."


def test_classify_returns_valid_score_result_from_mocked_client(tmp_path):
    payload = {
        "risk": 0.95,
        "tactic_tags": [{"id": "otp_request", "weight": 1.0}],
        "trigger_phrases": ["код из SMS"],
        "confidence": 0.8,
    }
    client = FakeClient([_tool_use_response(payload)])

    result = classify(TRANSCRIPT, client=client, cache_dir=tmp_path)

    assert isinstance(result, ScoreResult)
    assert result.risk == 0.95
    assert result.backend == "llm"
    assert result.raw_confidence == 0.8
    assert result.tags[0].id == "otp_request"
    assert result.attributions[0].text == "код из SMS"
    assert TRANSCRIPT[result.attributions[0].start : result.attributions[0].end] == "код из SMS"


def test_classify_uses_cache_on_second_call(tmp_path):
    payload = {"risk": 0.2, "tactic_tags": [], "trigger_phrases": [], "confidence": 0.5}
    client = FakeClient([_tool_use_response(payload)])

    first = classify(TRANSCRIPT, client=client, cache_dir=tmp_path)
    second = classify(TRANSCRIPT, client=client, cache_dir=tmp_path)

    assert first == second
    assert len(client.models.calls) == 1


def test_classify_cache_disabled_calls_client_each_time(tmp_path):
    payload = {"risk": 0.2, "tactic_tags": [], "trigger_phrases": [], "confidence": 0.5}
    client = FakeClient([_tool_use_response(payload), _tool_use_response(payload)])

    classify(TRANSCRIPT, client=client, cache_dir=tmp_path, use_cache=False)
    classify(TRANSCRIPT, client=client, cache_dir=tmp_path, use_cache=False)

    assert len(client.models.calls) == 2


def test_classify_raises_on_empty_json_response(tmp_path):
    response = SimpleNamespace(text=None, parsed=None)
    client = FakeClient([response])

    with pytest.raises(LLMClassifierError):
        classify(TRANSCRIPT, client=client, cache_dir=tmp_path)


def test_classify_repairs_span_with_wrong_offsets_but_correct_text(tmp_path):
    # The model only ever returns phrase *text*; classify() always computes offsets
    # itself via a substring search, so "wrong offsets from the model" can't happen --
    # this test documents/locks that contract.
    payload = {
        "risk": 0.9,
        "tactic_tags": [{"id": "secrecy"}],
        "trigger_phrases": ["код из SMS"],
        "confidence": 0.6,
    }
    client = FakeClient([_tool_use_response(payload)])

    result = classify(TRANSCRIPT, client=client, cache_dir=tmp_path)

    span = result.attributions[0]
    assert TRANSCRIPT[span.start : span.end] == span.text == "код из SMS"


def test_classify_drops_span_with_hallucinated_text_not_in_transcript(tmp_path):
    payload = {
        "risk": 0.9,
        "tactic_tags": [{"id": "otp_request"}],
        "trigger_phrases": ["код из SMS", "текст которого нет в транскрипте"],
        "confidence": 0.6,
    }
    client = FakeClient([_tool_use_response(payload)])

    result = classify(TRANSCRIPT, client=client, cache_dir=tmp_path)

    assert len(result.attributions) == 1
    assert result.attributions[0].text == "код из SMS"


def test_classify_raises_on_missing_required_field(tmp_path):
    payload = {"tactic_tags": [], "trigger_phrases": [], "confidence": 0.5}  # no "risk"
    client = FakeClient([_tool_use_response(payload)])

    with pytest.raises(LLMClassifierError):
        classify(TRANSCRIPT, client=client, cache_dir=tmp_path)


def test_classify_clamps_out_of_range_risk(tmp_path):
    payload = {"risk": 1.4, "tactic_tags": [], "trigger_phrases": [], "confidence": 0.5}
    client = FakeClient([_tool_use_response(payload)])

    result = classify(TRANSCRIPT, client=client, cache_dir=tmp_path)

    assert result.risk == 1.0


def test_classify_empty_transcript_raises_value_error(tmp_path):
    with pytest.raises(ValueError):
        classify("   ", client=FakeClient([]), cache_dir=tmp_path)


def test_classify_invalid_confidence_falls_back_to_none(tmp_path):
    payload = {
        "risk": 0.3,
        "tactic_tags": [],
        "trigger_phrases": [],
        "confidence": "not-a-number",
    }
    client = FakeClient([_tool_use_response(payload)])

    result = classify(TRANSCRIPT, client=client, cache_dir=tmp_path)

    assert result.raw_confidence is None


def test_classify_different_transcripts_get_different_cache_entries(tmp_path):
    payload_a = {"risk": 0.1, "tactic_tags": [], "trigger_phrases": [], "confidence": 0.5}
    payload_b = {"risk": 0.9, "tactic_tags": [], "trigger_phrases": [], "confidence": 0.5}
    client = FakeClient([_tool_use_response(payload_a), _tool_use_response(payload_b)])

    result_a = classify("transcript A", client=client, cache_dir=tmp_path)
    result_b = classify("transcript B", client=client, cache_dir=tmp_path)

    assert result_a.risk != result_b.risk
    assert len(client.models.calls) == 2
