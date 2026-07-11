"""SMOKE tests for `qorgan.eval.run` — the FPR-first eval harness, driven by an injected
score_fn against a tiny on-disk split (no network, no real backend)."""

import pytest

from qorgan.data.generate import write_dialogues_jsonl
from qorgan.data.schema import Dialogue, Label, ScoreResult, Span, TacticTag, Utterance, spans_from_phrases
from qorgan.eval.run import (
    evaluate_by_language,
    evaluate_split,
    format_report,
    load_split,
    run,
    tune_alert_threshold,
)
from qorgan.eval.threshold import ThresholdChoice


def _dialogue(did, text, *, risk, tags=(), phrases=(), hard_negative=False):
    utterances = (Utterance(speaker="caller", text=text),)
    return Dialogue(
        id=did,
        language="ru",
        utterances=utterances,
        label=Label(
            risk=risk,
            tactic_tags=tuple(TacticTag(id=t) for t in tags),
            trigger_spans=spans_from_phrases(phrases, text),
            is_hard_negative=hard_negative,
        ),
    )


def _perfect_score_fn(transcript):
    """A stand-in classifier that mirrors ground truth for the fixtures below."""
    if "код из SMS" in transcript:
        start = transcript.index("код из SMS")
        return ScoreResult(
            risk=0.95,
            tags=(TacticTag(id="otp_request"),),
            attributions=(Span(text="код из SMS", start=start, end=start + len("код из SMS")),),
            backend="mock",
        )
    return ScoreResult(risk=0.03, tags=(), attributions=(), backend="mock")


def _make_split(tmp_path, name):
    dialogues = [
        _dialogue("scam1", "Продиктуйте код из SMS сейчас", risk=0.9, tags=["otp_request"], phrases=["код из SMS"]),
        _dialogue("neg1", "Подтверждаем перевод на 5000 тенге", risk=0.02, hard_negative=True),
    ]
    path = tmp_path / f"{name}.jsonl"
    write_dialogues_jsonl(dialogues, path)
    return dialogues


def test_load_split_reads_jsonl(tmp_path):
    _make_split(tmp_path, "test")
    loaded = load_split(tmp_path, "test")
    assert [d.id for d in loaded] == ["scam1", "neg1"]
    assert all(isinstance(d, Dialogue) for d in loaded)


def test_load_split_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_split(tmp_path, "nope")


def test_evaluate_split_reports_fpr_first_and_perfect_scores(tmp_path):
    dialogues = _make_split(tmp_path, "test")
    result = evaluate_split(dialogues, score_fn=_perfect_score_fn, alert_threshold=0.7)

    # FPR is the headline metric and must be present.
    assert "fpr" in result
    assert result["fpr"] == 0.0  # the hard negative is not flagged
    assert result["recall"] == 1.0
    assert result["precision"] == 1.0
    assert result["support"] == 2
    assert "per_tactic_f1" in result
    assert result["per_tactic_f1"]["otp_request"] == 1.0


def test_evaluate_split_counts_a_false_positive(tmp_path):
    dialogues = _make_split(tmp_path, "test")

    def over_eager(_transcript):
        return ScoreResult(risk=0.99, tags=(), attributions=(), backend="mock")

    result = evaluate_split(dialogues, score_fn=over_eager, alert_threshold=0.7)
    assert result["fpr"] == 1.0  # the hard negative got flagged
    assert result["fp"] == 1


def test_run_evaluates_each_split_separately(tmp_path):
    _make_split(tmp_path, "test")
    _make_split(tmp_path, "real_heldout")
    results = run(
        tmp_path, ["test", "real_heldout"], score_fn=_perfect_score_fn, alert_threshold=0.7
    )
    assert set(results) == {"test", "real_heldout"}
    assert results["test"]["fpr"] == 0.0
    assert results["real_heldout"]["fpr"] == 0.0


def test_format_report_is_markdown_with_fpr_and_split_names(tmp_path):
    _make_split(tmp_path, "test")
    results = run(tmp_path, ["test"], score_fn=_perfect_score_fn, alert_threshold=0.7)
    report = format_report(results)
    assert "FPR" in report
    assert "test" in report
    assert "|" in report  # markdown table


def test_format_report_includes_per_tactic_section(tmp_path):
    _make_split(tmp_path, "test")
    results = run(tmp_path, ["test"], score_fn=_perfect_score_fn, alert_threshold=0.7)
    report = format_report(results)
    assert "Per-tactic F1" in report
    assert "otp_request" in report


def test_evaluate_by_language_groups_and_reports_each(tmp_path):
    from qorgan.data.schema import Dialogue, Label, TacticTag, Utterance, spans_from_phrases

    def d(did, text, lang, risk, tags=(), phrases=()):
        return Dialogue(
            id=did, language=lang, utterances=(Utterance(speaker="c", text=text),),
            label=Label(risk=risk, tactic_tags=tuple(TacticTag(id=t) for t in tags),
                        trigger_spans=spans_from_phrases(phrases, text)),
        )

    dialogues = [
        d("r1", "Продиктуйте код из SMS", "ru", 0.9, ["otp_request"], ["код из SMS"]),
        d("r2", "Перевод на 5000 тенге", "ru", 0.02),
        d("k1", "Продиктуйте код из SMS қазақша", "kk", 0.9, ["otp_request"], ["код из SMS"]),
    ]
    by_lang = evaluate_by_language(dialogues, score_fn=_perfect_score_fn, alert_threshold=0.7)
    assert set(by_lang) == {"ru", "kk"}
    assert "fpr" in by_lang["ru"] and "fpr" in by_lang["kk"]
    assert by_lang["ru"]["support"] == 2


def test_tune_alert_threshold_returns_low_fpr_choice(tmp_path):
    dialogues = _make_split(tmp_path, "real_heldout")
    choice = tune_alert_threshold(dialogues, score_fn=_perfect_score_fn, max_fpr=0.05)
    assert isinstance(choice, ThresholdChoice)
    assert choice.fpr <= 0.05
    assert 0.0 <= choice.threshold <= 1.0
