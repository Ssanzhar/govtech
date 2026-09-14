"""TDD tests for `qorgan.eval.metrics` -- the FPR-first evaluation primitives.

Convention throughout: positive class = scam = 1, negative = legitimate = 0.
FPR is the project's primary metric (CLAUDE.md SS3.5, SS6): a false alarm on a real
bank call destroys trust, so it is asserted first everywhere it matters.
"""

import pytest

from qorgan.eval.metrics import (
    adjusted_rand,
    binarize,
    binary_report,
    confusion_counts,
    f1,
    false_positive_rate,
    per_label_f1,
    pr_auc,
    precision,
    purity,
    recall,
)

# --- confusion_counts --------------------------------------------------------------------


def test_confusion_counts_perfect_classifier():
    y_true = [1, 1, 0, 0]
    y_pred = [1, 1, 0, 0]
    assert confusion_counts(y_true, y_pred) == (2, 0, 2, 0)


def test_confusion_counts_known_matrix():
    # tp=2 (idx0,3), fp=1 (idx1), tn=2 (idx2,5), fn=1 (idx4)
    y_true = [1, 0, 0, 1, 1, 0]
    y_pred = [1, 1, 0, 1, 0, 0]
    assert confusion_counts(y_true, y_pred) == (2, 1, 2, 1)


def test_confusion_counts_length_mismatch_raises():
    with pytest.raises(ValueError):
        confusion_counts([1, 0], [1])


def test_confusion_counts_non_binary_raises():
    with pytest.raises(ValueError):
        confusion_counts([1, 2], [1, 0])
    with pytest.raises(ValueError):
        confusion_counts([1, 0], [1, -1])


def test_confusion_counts_empty_is_all_zero():
    assert confusion_counts([], []) == (0, 0, 0, 0)


# --- false_positive_rate (PRIMARY METRIC) -------------------------------------------------


def test_fpr_perfect_classifier_is_zero():
    assert false_positive_rate([1, 1, 0, 0], [1, 1, 0, 0]) == 0.0


def test_fpr_known_matrix():
    # fp=1, tn=2 -> fpr = 1/3
    y_true = [1, 0, 0, 1, 1, 0]
    y_pred = [1, 1, 0, 1, 0, 0]
    assert false_positive_rate(y_true, y_pred) == pytest.approx(1 / 3)


def test_fpr_all_positive_ground_truth_returns_zero():
    # no true negatives at all -> fp + tn == 0 -> defined as 0.0
    assert false_positive_rate([1, 1, 1], [1, 0, 1]) == 0.0


def test_fpr_all_negative_ground_truth_well_defined():
    # all legitimate calls; some misclassified as scam
    y_true = [0, 0, 0, 0]
    y_pred = [1, 0, 1, 0]
    assert false_positive_rate(y_true, y_pred) == pytest.approx(0.5)


def test_fpr_worst_case_all_false_positives():
    y_true = [0, 0, 0]
    y_pred = [1, 1, 1]
    assert false_positive_rate(y_true, y_pred) == 1.0


# --- precision / recall / f1 --------------------------------------------------------------


def test_precision_recall_f1_perfect_classifier():
    y_true = [1, 1, 0, 0]
    y_pred = [1, 1, 0, 0]
    assert precision(y_true, y_pred) == 1.0
    assert recall(y_true, y_pred) == 1.0
    assert f1(y_true, y_pred) == 1.0


def test_precision_zero_when_no_predicted_positives():
    assert precision([1, 0, 1], [0, 0, 0]) == 0.0


def test_recall_zero_when_no_actual_positives_predicted():
    assert recall([1, 0, 1], [0, 0, 0]) == 0.0


def test_f1_zero_when_precision_and_recall_both_zero():
    assert f1([1, 0], [0, 1]) == 0.0


def test_f1_known_matrix():
    # tp=2, fp=1, fn=1 -> precision=2/3, recall=2/3 -> f1=2/3
    y_true = [1, 0, 0, 1, 1, 0]
    y_pred = [1, 1, 0, 1, 0, 0]
    assert f1(y_true, y_pred) == pytest.approx(2 / 3)


# --- binarize ------------------------------------------------------------------------------


def test_binarize_threshold_boundary_is_inclusive():
    scores = [0.1, 0.5, 0.5, 0.9]
    assert binarize(scores, 0.5) == [0, 1, 1, 1]


def test_binarize_basic():
    assert binarize([0.0, 0.3, 0.7, 1.0], 0.6) == [0, 0, 1, 1]


@pytest.mark.parametrize("bad_threshold", [-0.01, 1.01, -1, 2])
def test_binarize_out_of_range_threshold_raises(bad_threshold):
    with pytest.raises(ValueError):
        binarize([0.1, 0.9], bad_threshold)


def test_binarize_threshold_endpoints_valid():
    assert binarize([0.0, 1.0], 0.0) == [1, 1]
    assert binarize([0.0, 1.0], 1.0) == [0, 1]


# --- pr_auc --------------------------------------------------------------------------------


def test_pr_auc_perfectly_separable_scores_is_one():
    y_true = [0, 0, 1, 1]
    y_scores = [0.1, 0.2, 0.8, 0.9]
    assert pr_auc(y_true, y_scores) == pytest.approx(1.0)


def test_pr_auc_empty_returns_zero():
    assert pr_auc([], []) == 0.0


def test_pr_auc_single_class_returns_zero():
    assert pr_auc([1, 1, 1], [0.2, 0.5, 0.9]) == 0.0
    assert pr_auc([0, 0, 0], [0.2, 0.5, 0.9]) == 0.0


# --- per_label_f1 --------------------------------------------------------------------------


def test_per_label_f1_multi_label_example():
    true_labels = [{"otp_request"}, {"urgency", "authority"}, set(), {"otp_request", "urgency"}]
    pred_labels = [{"otp_request"}, {"urgency"}, {"authority"}, {"otp_request"}]
    label_ids = ["otp_request", "urgency", "authority"]

    result = per_label_f1(true_labels, pred_labels, label_ids)

    # otp_request: true=[1,0,0,1], pred=[1,0,0,1] -> perfect -> f1=1.0
    assert result["otp_request"] == pytest.approx(1.0)
    # urgency: true=[0,1,0,1], pred=[0,1,0,0] -> tp=1,fp=0,fn=1 -> p=1,r=0.5 -> f1=2/3
    assert result["urgency"] == pytest.approx(2 / 3)
    # authority: true=[0,1,0,0], pred=[0,0,1,0] -> tp=0,fp=1,fn=1 -> f1=0.0
    assert result["authority"] == pytest.approx(0.0)


def test_per_label_f1_length_mismatch_raises():
    with pytest.raises(ValueError):
        per_label_f1([{"a"}], [{"a"}, {"b"}], ["a", "b"])


def test_per_label_f1_empty_label_ids_returns_empty_dict():
    assert per_label_f1([{"a"}], [{"a"}], []) == {}


# --- purity --------------------------------------------------------------------------------


def test_purity_perfect_clustering_is_one():
    labels_true = ["a", "a", "b", "b"]
    labels_pred = [0, 0, 1, 1]
    assert purity(labels_true, labels_pred) == 1.0


def test_purity_known_imperfect_clustering():
    # cluster 0 -> {a,a,b} majority a (2/3), cluster 1 -> {b,b} majority b (2/2)
    labels_true = ["a", "a", "b", "b", "b"]
    labels_pred = [0, 0, 0, 1, 1]
    assert purity(labels_true, labels_pred) == pytest.approx(4 / 5)


def test_purity_empty_returns_zero():
    assert purity([], []) == 0.0


def test_purity_length_mismatch_raises():
    with pytest.raises(ValueError):
        purity(["a", "b"], [0])


# --- adjusted_rand -------------------------------------------------------------------------


def test_adjusted_rand_identical_labelings_is_one():
    labels_true = [0, 0, 1, 1, 2, 2]
    labels_pred = [0, 0, 1, 1, 2, 2]
    assert adjusted_rand(labels_true, labels_pred) == pytest.approx(1.0)


def test_adjusted_rand_empty_returns_zero():
    assert adjusted_rand([], []) == 0.0


def test_adjusted_rand_length_mismatch_raises():
    with pytest.raises(ValueError):
        adjusted_rand([0, 1], [0])


# --- binary_report (headline aggregator) --------------------------------------------------


def test_binary_report_key_order_is_fpr_first():
    y_true = [1, 0, 0, 1]
    y_scores = [0.9, 0.4, 0.1, 0.6]
    report = binary_report(y_true, y_scores, threshold=0.5)
    assert list(report.keys()) == [
        "fpr",
        "fpr_ci",
        "precision",
        "precision_ci",
        "recall",
        "recall_ci",
        "f1",
        "pr_auc",
        "pr_auc_ci",
        "tp",
        "fp",
        "tn",
        "fn",
        "support",
        "threshold",
    ]


def test_binary_report_intervals_are_exact_binomial_on_the_right_denominators():
    # 24 negatives, 0 false positives -> FPR 0.000 but the 95 % interval reaches 0.142.
    y_true = [0] * 24 + [1] * 18
    y_scores = [0.1] * 24 + [0.9] * 18
    report = binary_report(y_true, y_scores, threshold=0.5)
    assert report["fpr"] == 0.0
    assert report["fpr_ci"] == (0.0, pytest.approx(0.1423, abs=1e-3))
    assert report["recall"] == 1.0
    assert report["recall_ci"] == (pytest.approx(0.8147, abs=1e-3), 1.0)
    assert report["precision_ci"] == (pytest.approx(0.8147, abs=1e-3), 1.0)
    assert report["pr_auc_ci"] == (pytest.approx(1.0), pytest.approx(1.0))


def test_binary_report_intervals_are_none_when_a_class_is_absent():
    report = binary_report([1, 1], [0.9, 0.8], threshold=0.5)
    assert report["fpr_ci"] is None  # no negatives -> FPR undefined, not "[0, 1]"
    assert report["pr_auc_ci"] is None
    assert report["recall_ci"] == (pytest.approx(0.1581, abs=1e-3), 1.0)


def test_binary_report_bootstrap_resamples_is_configurable():
    y_true = [1, 0, 1, 0]
    y_scores = [0.9, 0.6, 0.4, 0.1]
    fast = binary_report(y_true, y_scores, threshold=0.5, bootstrap_resamples=20)
    assert fast["pr_auc_ci"] is not None


def test_binary_report_values_perfect_classifier():
    y_true = [1, 1, 0, 0]
    y_scores = [0.9, 0.8, 0.2, 0.1]
    report = binary_report(y_true, y_scores, threshold=0.5)
    assert report["fpr"] == 0.0
    assert report["precision"] == 1.0
    assert report["recall"] == 1.0
    assert report["f1"] == 1.0
    assert report["pr_auc"] == pytest.approx(1.0)
    assert report["tp"] == 2
    assert report["fp"] == 0
    assert report["tn"] == 2
    assert report["fn"] == 0
    assert report["support"] == 4
    assert report["threshold"] == 0.5
    assert isinstance(report["tp"], int)
    assert isinstance(report["fpr"], float)


def test_binary_report_support_equals_length_of_y_true():
    y_true = [1, 0, 1, 0, 1]
    y_scores = [0.9, 0.1, 0.6, 0.4, 0.55]
    report = binary_report(y_true, y_scores, threshold=0.5)
    assert report["support"] == 5
