"""TDD tests for `qorgan.data.schema` — the frozen data contracts.

Covers: `Span` self-consistency, verbatim-substring validation (`validate_verbatim_spans`),
`Dialogue`/`Incident` transcript-vs-trigger-span grounding, and bounds/immutability across
all nine contract models.
"""

import pytest
from pydantic import ValidationError

from qorgan.data.schema import (
    Dialogue,
    Explanation,
    Incident,
    Label,
    Organization,
    ScoreResult,
    Span,
    SpanValidationError,
    TacticTag,
    Utterance,
    validate_verbatim_spans,
)

# --- Span ---------------------------------------------------------------------------


def test_span_valid_construction():
    span = Span(text="hello", start=0, end=5)
    assert span.text == "hello"
    assert span.start == 0
    assert span.end == 5


def test_span_end_before_start_raises():
    with pytest.raises(ValidationError):
        Span(text="x", start=5, end=2)


def test_span_end_equal_start_raises():
    with pytest.raises(ValidationError):
        Span(text="", start=2, end=2)


def test_span_length_mismatch_raises():
    with pytest.raises(ValidationError):
        Span(text="hello", start=0, end=4)


def test_span_empty_text_raises():
    with pytest.raises(ValidationError):
        Span(text="   ", start=0, end=3)


def test_span_negative_start_raises():
    with pytest.raises(ValidationError):
        Span(text="hi", start=-1, end=1)


def test_span_is_frozen():
    span = Span(text="hi", start=0, end=2)
    with pytest.raises(ValidationError):
        span.start = 1


# --- validate_verbatim_spans ---------------------------------------------------------


def test_validate_verbatim_spans_passes_for_exact_slice():
    transcript = "Продиктуйте код из SMS немедленно"
    phrase = "код из SMS"
    start = transcript.index(phrase)
    span = Span(text=phrase, start=start, end=start + len(phrase))
    validate_verbatim_spans([span], transcript)  # should not raise


def test_validate_verbatim_spans_raises_when_slice_does_not_match():
    transcript = "Продиктуйте код из SMS немедленно"
    bad_span = Span(text="код из SMS", start=0, end=10)
    with pytest.raises(SpanValidationError):
        validate_verbatim_spans([bad_span], transcript)


def test_validate_verbatim_spans_empty_list_is_fine():
    validate_verbatim_spans([], "any transcript")


# --- TacticTag -----------------------------------------------------------------------


def test_tactic_tag_default_weight():
    tag = TacticTag(id="otp_request")
    assert tag.weight == 1.0


def test_tactic_tag_weight_out_of_range_raises():
    with pytest.raises(ValidationError):
        TacticTag(id="otp_request", weight=1.5)


def test_tactic_tag_blank_id_raises():
    with pytest.raises(ValidationError):
        TacticTag(id="   ")


# --- Utterance -----------------------------------------------------------------------


def test_utterance_valid():
    u = Utterance(speaker="caller", text="Hello there")
    assert u.speaker == "caller"


def test_utterance_blank_text_raises():
    with pytest.raises(ValidationError):
        Utterance(speaker="caller", text="   ")


def test_utterance_blank_speaker_raises():
    with pytest.raises(ValidationError):
        Utterance(speaker="  ", text="hi")


# --- Label ---------------------------------------------------------------------------


def test_label_risk_bounds_valid():
    label = Label(risk=0.5)
    assert label.risk == 0.5
    assert label.tactic_tags == ()
    assert label.trigger_spans == ()
    assert label.is_hard_negative is False


def test_label_risk_out_of_range_raises():
    with pytest.raises(ValidationError):
        Label(risk=1.5)
    with pytest.raises(ValidationError):
        Label(risk=-0.1)


# --- Dialogue --------------------------------------------------------------------------


def _dialogue(text: str, spans: list[Span] | None = None) -> Dialogue:
    return Dialogue(
        id="d1",
        language="ru",
        utterances=[Utterance(speaker="caller", text=text)],
        label=Label(risk=0.9, trigger_spans=spans or []),
    )


def test_dialogue_requires_at_least_one_utterance():
    with pytest.raises(ValidationError):
        Dialogue(id="d1", language="ru", utterances=[], label=Label(risk=0.1))


def test_dialogue_transcript_joins_utterances():
    dialogue = Dialogue(
        id="d1",
        language="ru",
        utterances=[
            Utterance(speaker="caller", text="Привет"),
            Utterance(speaker="callee", text="Здравствуйте"),
        ],
        label=Label(risk=0.1),
    )
    assert dialogue.transcript() == "Привет\nЗдравствуйте"


def test_dialogue_trigger_span_verbatim_passes():
    text = "Продиктуйте код из SMS"
    start = text.index("код из SMS")
    span = Span(text="код из SMS", start=start, end=start + len("код из SMS"))
    dialogue = _dialogue(text, [span])
    assert dialogue.label.trigger_spans[0].text == "код из SMS"


def test_dialogue_trigger_span_not_verbatim_raises():
    text = "Продиктуйте код из SMS"
    bad_span = Span(text="код из SMS", start=0, end=len("код из SMS"))
    with pytest.raises(ValidationError):
        _dialogue(text, [bad_span])


def test_dialogue_language_must_be_supported():
    with pytest.raises(ValidationError):
        Dialogue(
            id="d1",
            language="en",
            utterances=[Utterance(speaker="caller", text="hi")],
            label=Label(risk=0.1),
        )


def test_dialogue_is_frozen():
    dialogue = _dialogue("hello world")
    with pytest.raises(ValidationError):
        dialogue.id = "other"


# --- Incident --------------------------------------------------------------------------


def test_incident_verbatim_validation_passes():
    transcript = "Переведите деньги на безопасный счёт"
    phrase = "безопасный счёт"
    start = transcript.index(phrase)
    span = Span(text=phrase, start=start, end=start + len(phrase))
    incident = Incident(
        id="i1",
        dialogue_id="d1",
        transcript=transcript,
        label=Label(risk=0.95, trigger_spans=[span]),
    )
    assert incident.transcript == transcript


def test_incident_verbatim_validation_fails():
    transcript = "Переведите деньги на безопасный счёт"
    bad_span = Span(text="безопасный счёт", start=0, end=15)
    with pytest.raises(ValidationError):
        Incident(
            id="i1",
            dialogue_id="d1",
            transcript=transcript,
            label=Label(risk=0.95, trigger_spans=[bad_span]),
        )


def test_incident_optional_fields_default_none():
    incident = Incident(id="i1", dialogue_id="d1", transcript="hi", label=Label(risk=0.1))
    assert incident.phone_number is None
    assert incident.timestamp is None
    assert incident.script_family is None


# --- ScoreResult -----------------------------------------------------------------------


def test_score_result_valid():
    result = ScoreResult(
        risk=0.8,
        tags=[TacticTag(id="otp_request")],
        attributions=[Span(text="hi", start=0, end=2)],
        backend="llm",
        raw_confidence=0.7,
    )
    assert result.risk == 0.8
    assert result.backend == "llm"
    assert result.raw_confidence == 0.7
    assert isinstance(result.tags, tuple)
    assert isinstance(result.attributions, tuple)


def test_score_result_backend_empty_raises():
    with pytest.raises(ValidationError):
        ScoreResult(risk=0.1, backend="  ")


def test_score_result_risk_bounds():
    with pytest.raises(ValidationError):
        ScoreResult(risk=1.1, backend="llm")


def test_score_result_raw_confidence_optional_none():
    result = ScoreResult(risk=0.1, backend="mock", raw_confidence=None)
    assert result.raw_confidence is None


def test_score_result_raw_confidence_out_of_range_raises():
    with pytest.raises(ValidationError):
        ScoreResult(risk=0.1, backend="llm", raw_confidence=1.2)


def test_score_result_defaults_to_empty_tags_and_attributions():
    result = ScoreResult(risk=0.1, backend="mock")
    assert result.tags == ()
    assert result.attributions == ()


# --- Explanation -----------------------------------------------------------------------


def test_explanation_valid():
    explanation = Explanation(
        reason="Detected OTP request.",
        highlights=[Span(text="hi", start=0, end=2)],
        tags=[TacticTag(id="otp_request")],
        confidence=0.5,
        caveat="Model may be wrong.",
        human_note="A human must decide.",
    )
    assert explanation.reason == "Detected OTP request."
    assert explanation.confidence == 0.5


def test_explanation_confidence_optional():
    explanation = Explanation(
        reason="No signal.",
        caveat="caveat",
        human_note="note",
    )
    assert explanation.confidence is None
    assert explanation.highlights == ()
    assert explanation.tags == ()


# --- Organization -----------------------------------------------------------------------


def test_organization_valid():
    org = Organization(
        id="org1",
        members=["d1", "d2"],
        numbers=["+77001234567"],
        representative_script="bank_impersonation_otp",
        priority=0.9,
        is_novel=False,
    )
    assert org.members == ("d1", "d2")
    assert org.is_novel is False


def test_organization_defaults():
    org = Organization(id="org1", members=["d1"])
    assert org.numbers == ()
    assert org.representative_script is None
    assert org.priority == 0.0
    assert org.is_novel is False


def test_organization_is_frozen():
    org = Organization(id="org1", members=["d1"])
    with pytest.raises(ValidationError):
        org.priority = 1.0
