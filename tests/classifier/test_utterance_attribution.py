"""TDD tests for `select_top_utterance_spans` — utterance-level grounded attribution."""

import pytest

from qorgan.classifier.attribution import select_top_utterance_spans
from qorgan.data.schema import UTTERANCE_JOIN, Span, validate_verbatim_spans

_UTTERANCES = [
    "Здравствуйте это банк",
    "Продиктуйте код из SMS",
    "Спасибо за информацию",
]
_TRANSCRIPT = UTTERANCE_JOIN.join(_UTTERANCES)


def test_returns_spans_for_highest_scoring_utterances():
    spans = select_top_utterance_spans(_UTTERANCES, [0.1, 0.9, 0.2], top_k=1)
    assert len(spans) == 1
    assert spans[0].text == "Продиктуйте код из SMS"


def test_spans_are_verbatim_substrings_of_the_joined_transcript():
    spans = select_top_utterance_spans(_UTTERANCES, [0.9, 0.8, 0.1], top_k=2)
    validate_verbatim_spans(spans, _TRANSCRIPT)  # raises if any offset is wrong
    assert all(isinstance(s, Span) for s in spans)


def test_spans_sorted_by_start():
    spans = select_top_utterance_spans(_UTTERANCES, [0.2, 0.9, 0.8], top_k=2)
    assert [s.start for s in spans] == sorted(s.start for s in spans)


def test_min_score_filters_low_utterances():
    spans = select_top_utterance_spans(_UTTERANCES, [0.1, 0.2, 0.15], top_k=3, min_score=0.5)
    assert spans == ()


def test_top_k_caps_count():
    spans = select_top_utterance_spans(_UTTERANCES, [0.9, 0.8, 0.7], top_k=2)
    assert len(spans) == 2


def test_blank_utterance_skipped():
    utterances = ["   ", "Реальная фраза здесь"]
    spans = select_top_utterance_spans(utterances, [0.99, 0.5], top_k=2)
    # the blank top-scorer is dropped (can't be a Span), the real one survives
    assert [s.text for s in spans] == ["Реальная фраза здесь"]


def test_handles_duplicate_utterances_via_computed_offsets():
    utterances = ["повтор", "повтор", "уникальный"]
    spans = select_top_utterance_spans(utterances, [0.9, 0.8, 0.1], top_k=2)
    transcript = UTTERANCE_JOIN.join(utterances)
    validate_verbatim_spans(spans, transcript)
    # the two spans point at the two DIFFERENT occurrences (different start offsets)
    assert spans[0].start != spans[1].start


def test_length_mismatch_raises():
    with pytest.raises(ValueError):
        select_top_utterance_spans(_UTTERANCES, [0.1, 0.2], top_k=1)


def test_top_k_non_positive_raises():
    with pytest.raises(ValueError):
        select_top_utterance_spans(_UTTERANCES, [0.1, 0.2, 0.3], top_k=0)
