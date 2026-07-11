"""TDD tests for `qorgan.classifier.attribution` -- the pure token->char span aligner.
The Captum integrated-gradients scorer (needs torch/captum) lands later in the same file.
"""

import pytest

from qorgan.classifier.attribution import align_token_attributions_to_spans
from qorgan.data.schema import Span

TRANSCRIPT = "Продиктуйте код из SMS прямо сейчас"
#              0         1         2         3
#              0123456789012345678901234567890123456


def _offsets_for(*words: str) -> list[tuple[int, int]]:
    """Helper: build offset_mapping entries for each `word`, located verbatim in
    TRANSCRIPT (test-only convenience, not part of the module under test)."""
    offsets = []
    for word in words:
        start = TRANSCRIPT.index(word)
        offsets.append((start, start + len(word)))
    return offsets


# --- special tokens skipped ---------------------------------------------------------------


def test_special_tokens_with_zero_width_offsets_are_skipped():
    token_scores = [0.0, 0.9, 0.0]
    offset_mapping = [(0, 0)] + _offsets_for("код") + [(0, 0)]
    spans = align_token_attributions_to_spans(
        token_scores, offset_mapping, TRANSCRIPT, top_k=5
    )
    assert len(spans) == 1
    assert spans[0].text == "код"


# --- top_k limits count ---------------------------------------------------------------


def test_top_k_limits_number_of_selected_tokens():
    words = ["Продиктуйте", "код", "SMS", "прямо", "сейчас"]
    token_scores = [0.9, 0.5, 0.8, 0.3, 0.2]
    offset_mapping = _offsets_for(*words)
    spans = align_token_attributions_to_spans(
        token_scores, offset_mapping, TRANSCRIPT, top_k=2
    )
    # top_k=2 -> "Продиктуйте" (0.9) and "SMS" (0.8); neither is adjacent -> 2 separate spans
    assert len(spans) == 2
    texts = {span.text for span in spans}
    assert texts == {"Продиктуйте", "SMS"}


# --- contiguous tokens merged ---------------------------------------------------------------


def test_contiguous_tokens_are_merged_into_one_span():
    # "код" and " из" are adjacent character ranges in the transcript
    code_start = TRANSCRIPT.index("код")
    code_end = code_start + len("код")
    tail_start = code_end
    tail_end = tail_start + len(" из")
    assert TRANSCRIPT[code_start:tail_end] == "код из"

    token_scores = [0.9, 0.8]
    offset_mapping = [(code_start, code_end), (tail_start, tail_end)]
    spans = align_token_attributions_to_spans(
        token_scores, offset_mapping, TRANSCRIPT, top_k=2
    )
    assert len(spans) == 1
    assert spans[0].text == TRANSCRIPT[code_start:tail_end]
    assert spans[0].start == code_start
    assert spans[0].end == tail_end


def test_overlapping_token_ranges_are_merged():
    token_scores = [0.9, 0.8]
    offset_mapping = [(0, 10), (5, 15)]
    spans = align_token_attributions_to_spans(
        token_scores, offset_mapping, TRANSCRIPT, top_k=2
    )
    assert len(spans) == 1
    assert spans[0].start == 0
    assert spans[0].end == 15
    assert spans[0].text == TRANSCRIPT[0:15]


# --- non-adjacent tokens stay separate ---------------------------------------------------------------


def test_non_adjacent_tokens_stay_separate_spans():
    words = ["Продиктуйте", "сейчас"]
    token_scores = [0.9, 0.8]
    offset_mapping = _offsets_for(*words)
    spans = align_token_attributions_to_spans(
        token_scores, offset_mapping, TRANSCRIPT, top_k=2
    )
    assert len(spans) == 2
    assert [span.text for span in spans] == ["Продиктуйте", "сейчас"]
    # sorted by start
    assert spans[0].start < spans[1].start


# --- min_score filters ---------------------------------------------------------------


def test_min_score_filters_low_scoring_tokens():
    words = ["Продиктуйте", "код", "SMS"]
    token_scores = [0.9, 0.1, 0.8]
    offset_mapping = _offsets_for(*words)
    spans = align_token_attributions_to_spans(
        token_scores, offset_mapping, TRANSCRIPT, top_k=5, min_score=0.5
    )
    texts = {span.text for span in spans}
    assert texts == {"Продиктуйте", "SMS"}
    assert "код" not in texts


# --- validation errors ---------------------------------------------------------------


def test_length_mismatch_raises():
    with pytest.raises(ValueError):
        align_token_attributions_to_spans([0.1, 0.2], [(0, 1)], TRANSCRIPT, top_k=1)


@pytest.mark.parametrize("bad_top_k", [0, -1, -5])
def test_non_positive_top_k_raises(bad_top_k):
    with pytest.raises(ValueError):
        align_token_attributions_to_spans([0.1], [(0, 1)], TRANSCRIPT, top_k=bad_top_k)


def test_empty_selection_returns_empty_tuple():
    token_scores = [0.0, 0.0]
    offset_mapping = [(0, 0), (0, 0)]
    spans = align_token_attributions_to_spans(
        token_scores, offset_mapping, TRANSCRIPT, top_k=5
    )
    assert spans == ()


def test_empty_token_scores_returns_empty_tuple():
    assert align_token_attributions_to_spans([], [], TRANSCRIPT, top_k=1) == ()


def test_returned_spans_are_span_instances_and_verbatim():
    words = ["сейчас"]
    token_scores = [0.9]
    offset_mapping = _offsets_for(*words)
    spans = align_token_attributions_to_spans(
        token_scores, offset_mapping, TRANSCRIPT, top_k=1
    )
    assert len(spans) == 1
    assert isinstance(spans[0], Span)
    assert TRANSCRIPT[spans[0].start : spans[0].end] == spans[0].text


def test_blank_span_after_slicing_is_skipped():
    # a whitespace-only token range should never produce a blank Span
    transcript = "код   из"
    token_scores = [0.9]
    offset_mapping = [(3, 6)]  # the three spaces between "код" and "из"
    spans = align_token_attributions_to_spans(
        token_scores, offset_mapping, transcript, top_k=1
    )
    assert spans == ()
