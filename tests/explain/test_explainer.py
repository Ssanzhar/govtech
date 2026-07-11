"""TDD tests for `qorgan.explain.explainer` — grounded, templated, localized explanations."""

import pytest

from qorgan.data.schema import ScoreResult, Span, TacticTag
from qorgan.explain.explainer import ExplainerError, explain


def _span(transcript: str, phrase: str) -> Span:
    start = transcript.index(phrase)
    return Span(text=phrase, start=start, end=start + len(phrase))


TRANSCRIPT = "Это служба безопасности банка. Продиктуйте код из SMS немедленно."


def test_explain_ru_with_tags_and_spans_builds_grounded_reason():
    result = ScoreResult(
        risk=0.9,
        tags=[TacticTag(id="otp_request"), TacticTag(id="impersonation_bank")],
        attributions=[_span(TRANSCRIPT, "код из SMS")],
        backend="llm",
        raw_confidence=0.7,
    )

    explanation = explain(result, TRANSCRIPT, "ru")

    assert "код из SMS" in explanation.reason
    assert explanation.highlights[0].text == "код из SMS"
    assert explanation.tags == result.tags
    assert explanation.confidence == 0.7
    assert explanation.caveat
    assert explanation.human_note


def test_explain_kk_locale_uses_kazakh_tag_names():
    result = ScoreResult(
        risk=0.9,
        tags=[TacticTag(id="otp_request")],
        attributions=[_span(TRANSCRIPT, "код из SMS")],
        backend="llm",
    )

    ru_explanation = explain(result, TRANSCRIPT, "ru")
    kk_explanation = explain(result, TRANSCRIPT, "kk")

    assert ru_explanation.reason != kk_explanation.reason


def test_explain_no_tags_uses_no_signal_reason():
    result = ScoreResult(risk=0.05, backend="llm")

    explanation = explain(result, TRANSCRIPT, "ru")

    assert explanation.highlights == ()
    assert explanation.tags == ()
    assert explanation.reason  # non-empty, templated no-signal message


def test_explain_unsupported_locale_raises():
    result = ScoreResult(risk=0.1, backend="llm")
    with pytest.raises(ExplainerError):
        explain(result, TRANSCRIPT, "en")


def test_explain_empty_transcript_raises():
    result = ScoreResult(risk=0.1, backend="llm")
    with pytest.raises(ExplainerError):
        explain(result, "   ", "ru")


def test_explain_span_not_in_transcript_raises():
    bad_result = ScoreResult(
        risk=0.9,
        tags=[TacticTag(id="otp_request")],
        attributions=[Span(text="код из SMS", start=0, end=10)],
        backend="llm",
    )
    with pytest.raises(Exception):
        explain(bad_result, TRANSCRIPT, "ru")


def test_explain_caveat_mentions_uncalibrated_for_llm_backend():
    result = ScoreResult(risk=0.9, tags=[TacticTag(id="otp_request")], backend="llm")
    explanation = explain(result, TRANSCRIPT, "ru")
    assert "llm" in explanation.caveat.lower()


def test_explain_caveat_omits_uncalibrated_suffix_for_xlmr_backend():
    result = ScoreResult(risk=0.9, tags=[TacticTag(id="otp_request")], backend="xlmr")
    explanation = explain(result, TRANSCRIPT, "ru")
    assert "xlmr" not in explanation.caveat.lower()


def test_explain_human_note_present_both_locales():
    result = ScoreResult(risk=0.1, backend="llm")
    ru = explain(result, TRANSCRIPT, "ru")
    kk = explain(result, TRANSCRIPT, "kk")
    assert ru.human_note
    assert kk.human_note
    assert ru.human_note != kk.human_note


def test_explain_highlights_sorted_by_start():
    later = _span(TRANSCRIPT, "немедленно")
    earlier = _span(TRANSCRIPT, "служба безопасности")
    result = ScoreResult(
        risk=0.9,
        tags=[TacticTag(id="impersonation_bank")],
        attributions=[later, earlier],
        backend="llm",
    )

    explanation = explain(result, TRANSCRIPT, "ru")

    assert explanation.highlights[0].start < explanation.highlights[1].start


def test_explain_surfaces_localized_confidence_label_high_for_calibrated():
    result = ScoreResult(
        risk=0.9, tags=[TacticTag(id="otp_request")], backend="xlmr", raw_confidence=0.92
    )
    explanation = explain(result, TRANSCRIPT, "ru")
    assert explanation.confidence_label  # non-empty localized band
    assert "высок" in explanation.confidence_label.lower()  # "высокая"


def test_explain_confidence_label_unknown_when_no_confidence():
    result = ScoreResult(risk=0.9, tags=[TacticTag(id="otp_request")], backend="llm")
    explanation = explain(result, TRANSCRIPT, "ru")
    assert explanation.confidence_label  # unknown-band string, not empty/None


def test_explain_confidence_label_differs_by_locale():
    result = ScoreResult(
        risk=0.9, tags=[TacticTag(id="otp_request")], backend="xlmr", raw_confidence=0.5
    )
    ru = explain(result, TRANSCRIPT, "ru")
    kk = explain(result, TRANSCRIPT, "kk")
    assert ru.confidence_label != kk.confidence_label


def test_explain_reason_lists_localized_tactic_name():
    result = ScoreResult(
        risk=0.9,
        tags=[TacticTag(id="otp_request")],
        attributions=[_span(TRANSCRIPT, "код из SMS")],
        backend="llm",
    )
    explanation = explain(result, TRANSCRIPT, "ru")
    # the RU display name for otp_request must appear in the reason (from taxonomy)
    from qorgan.taxonomy import get_taxonomy

    ru_name = get_taxonomy().display_name("otp_request", "ru")
    assert ru_name in explanation.reason


def test_explain_unknown_tag_id_skipped_gracefully():
    result = ScoreResult(
        risk=0.9,
        tags=[TacticTag(id="not_a_real_tactic")],
        backend="llm",
    )

    explanation = explain(result, TRANSCRIPT, "ru")

    assert explanation.tags[0].id == "not_a_real_tactic"  # passthrough, not crashed
    assert explanation.reason  # still produces something sensible
