"""TDD tests for `qorgan.classifier.labels` -- the multi-label tactic label space and
target/weight encoding shared by the trained XLM-R head.

Convention: positive class = scam = 1, negative = legitimate = 0 (same as
`qorgan.eval.metrics`). All functions here are pure: no I/O, no mutation of inputs.
"""

import pytest

from qorgan.classifier.labels import (
    build_label_space,
    decode_tactics,
    default_label_space,
    encode_tactics,
    pos_weight,
    tactic_pos_weights,
)

# --- build_label_space ---------------------------------------------------------------------


def test_build_label_space_dedups_preserving_first_seen_order():
    ids = ["urgency", "otp_request", "urgency", "secrecy", "otp_request"]
    assert build_label_space(ids) == ("urgency", "otp_request", "secrecy")


def test_build_label_space_single_id():
    assert build_label_space(["urgency"]) == ("urgency",)


def test_build_label_space_empty_raises():
    with pytest.raises(ValueError):
        build_label_space([])


def test_build_label_space_all_blank_raises():
    with pytest.raises(ValueError):
        build_label_space(["", "   ", ""])


# --- default_label_space ---------------------------------------------------------------------


def test_default_label_space_matches_real_taxonomy():
    from qorgan.taxonomy import get_taxonomy

    space = default_label_space()
    assert space == build_label_space(get_taxonomy().tactic_ids())
    assert len(space) == len(set(space))
    assert len(space) > 0


# --- encode_tactics ---------------------------------------------------------------------


def test_encode_tactics_multi_hot_basic():
    label_space = ("urgency", "otp_request", "secrecy")
    encoded = encode_tactics(["secrecy", "urgency"], label_space)
    assert encoded == [1.0, 0.0, 1.0]


def test_encode_tactics_no_matches_all_zero():
    label_space = ("urgency", "otp_request", "secrecy")
    assert encode_tactics([], label_space) == [0.0, 0.0, 0.0]


def test_encode_tactics_ignores_unknown_ids():
    label_space = ("urgency", "otp_request")
    encoded = encode_tactics(["urgency", "not_a_real_tactic"], label_space)
    assert encoded == [1.0, 0.0]


def test_encode_tactics_returns_floats():
    label_space = ("urgency",)
    encoded = encode_tactics(["urgency"], label_space)
    assert all(isinstance(value, float) for value in encoded)


# --- decode_tactics ---------------------------------------------------------------------


def test_decode_tactics_sorted_by_prob_desc_then_id_asc():
    label_space = ("urgency", "otp_request", "secrecy")
    probs = [0.4, 0.9, 0.4]
    result = decode_tactics(probs, label_space, threshold=0.3)
    # otp_request highest; urgency/secrecy tie at 0.4 -> id ASC ("secrecy" < "urgency")
    assert result == [("otp_request", 0.9), ("secrecy", 0.4), ("urgency", 0.4)]


def test_decode_tactics_threshold_boundary_is_inclusive():
    label_space = ("urgency", "otp_request")
    probs = [0.5, 0.2]
    result = decode_tactics(probs, label_space, threshold=0.5)
    assert result == [("urgency", 0.5)]


def test_decode_tactics_below_threshold_excluded():
    label_space = ("urgency", "otp_request")
    probs = [0.1, 0.2]
    result = decode_tactics(probs, label_space, threshold=0.5)
    assert result == []


def test_decode_tactics_length_mismatch_raises():
    with pytest.raises(ValueError):
        decode_tactics([0.1, 0.2], ("urgency",), threshold=0.5)


@pytest.mark.parametrize("bad_threshold", [-0.01, 1.01, -1, 2])
def test_decode_tactics_out_of_range_threshold_raises(bad_threshold):
    with pytest.raises(ValueError):
        decode_tactics([0.1], ("urgency",), threshold=bad_threshold)


def test_decode_tactics_threshold_endpoints_valid():
    assert decode_tactics([0.0], ("urgency",), threshold=0.0) == [("urgency", 0.0)]
    assert decode_tactics([1.0], ("urgency",), threshold=1.0) == [("urgency", 1.0)]


# --- pos_weight ---------------------------------------------------------------------


def test_pos_weight_balanced_targets():
    assert pos_weight([1, 1, 0, 0]) == pytest.approx(1.0)


def test_pos_weight_imbalanced_targets():
    # 1 positive, 3 negatives -> 3/1 = 3.0
    assert pos_weight([1, 0, 0, 0]) == pytest.approx(3.0)


def test_pos_weight_all_negative_returns_default_one():
    assert pos_weight([0, 0, 0]) == 1.0


def test_pos_weight_all_positive_returns_zero():
    # num_zeros=0, num_ones=3 -> 0/3 = 0.0
    assert pos_weight([1, 1, 1]) == 0.0


def test_pos_weight_empty_raises():
    with pytest.raises(ValueError):
        pos_weight([])


def test_pos_weight_invalid_value_raises():
    with pytest.raises(ValueError):
        pos_weight([1, 0, 2])


# --- tactic_pos_weights ---------------------------------------------------------------------


def test_tactic_pos_weights_basic_matrix():
    # column 0: [1,0,0] -> 1 pos, 2 neg -> 2.0
    # column 1: [1,1,0] -> 2 pos, 1 neg -> 0.5
    matrix = [
        [1.0, 1.0],
        [0.0, 1.0],
        [0.0, 0.0],
    ]
    assert tactic_pos_weights(matrix) == pytest.approx([2.0, 0.5])


def test_tactic_pos_weights_treats_ge_half_as_positive():
    matrix = [
        [0.5, 0.49],
        [0.0, 0.0],
    ]
    # column 0: one positive (0.5 >= 0.5), one negative -> 1/1 = 1.0
    # column 1: zero positives (0.49 < 0.5) -> default 1.0
    assert tactic_pos_weights(matrix) == pytest.approx([1.0, 1.0])


def test_tactic_pos_weights_no_positives_in_column_returns_one():
    # column 0 has no positives -> defaults to 1.0; column 1 is all positives -> 0/2 = 0.0
    matrix = [[0.0, 1.0], [0.0, 1.0]]
    assert tactic_pos_weights(matrix) == pytest.approx([1.0, 0.0])


def test_tactic_pos_weights_empty_matrix_raises():
    with pytest.raises(ValueError):
        tactic_pos_weights([])


def test_tactic_pos_weights_ragged_rows_raises():
    with pytest.raises(ValueError):
        tactic_pos_weights([[1.0, 0.0], [1.0]])
