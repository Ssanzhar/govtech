"""TDD tests for `qorgan.eval.asr_realism` (PLAN A10): the same dialogues scored clean and
ASR-styled; FPR first, then recall, both with intervals; decision flips; and whether the
hard-signal cues and reassurance patterns survive the styling."""

from __future__ import annotations

from qorgan.data.schema import Dialogue, Label, ScoreResult, TacticTag, Utterance
from qorgan.eval.asr_realism import paired_split, render_table


def _d(did, text, *, risk, language="ru"):
    return Dialogue(
        id=did, language=language, utterances=(Utterance(speaker="caller", text=text),),
        label=Label(risk=risk, tactic_tags=(TacticTag(id="otp_request"),) if risk >= 0.5 else ()),
    )


def _score_fn(alert_texts):
    def score(text):
        return ScoreResult(risk=0.95 if text in alert_texts else 0.1, backend="fake")
    return score


def test_paired_split_reports_fpr_first_recall_flips_and_cue_survival():
    dialogues = [
        _d("s1", "Продиктуйте код из SMS!", risk=0.9),
        _d("s2", "Переведите 5000 тенге.", risk=0.9),
        _d("n1", "Мы никогда не просим код из SMS.", risk=0.05),
        _d("n2", "Ваша карта готова.", risk=0.05),
    ]
    styled = {"продиктуйте код из sms", "переведите пять тысяч тенге", "мы никогда не просим код из sms", "ваша карта готова"}
    # clean: both scams alert, no legit alerts; styled: s2 drops, n2 starts alerting
    score = _score_fn({"Продиктуйте код из SMS!", "Переведите 5000 тенге.", "продиктуйте код из sms", "ваша карта готова"})
    cue_hits = lambda text: {"otp_request"} if "код из sms" in text.lower() else set()  # noqa: E731
    reassures = lambda text: "никогда не просим" in text.lower()  # noqa: E731

    result = paired_split(dialogues, score_fn=score, alert_threshold=0.5, cue_hits=cue_hits, reassures=reassures)

    assert result["n_positives"] == 2 and result["n_negatives"] == 2
    assert result["clean"]["fpr"] == 0.0 and result["styled"]["fpr"] == 0.5
    assert result["clean"]["recall"] == 1.0 and result["styled"]["recall"] == 0.5
    assert result["clean"]["fpr_interval"].high < 1.0 and result["styled"]["recall_interval"].low >= 0.0
    assert result["flips_to_clear"] == 1 and result["flips_to_alert"] == 1
    assert result["cue_hits_clean"] == 2 and result["cue_hits_preserved"] == 2  # both survive lowercasing
    assert result["reassurance_clean"] == 1 and result["reassurance_preserved"] == 1
    assert result["styled_texts"] == styled  # what the scorer actually saw


def test_render_table_puts_fpr_before_recall_and_names_the_split():
    dialogues = [_d("s1", "A", risk=0.9), _d("n1", "B", risk=0.05)]
    result = paired_split(dialogues, score_fn=_score_fn({"A", "a"}), alert_threshold=0.5, cue_hits=lambda t: set(), reassures=lambda t: False)
    table = render_table({"test": result})
    header = table.splitlines()[0]
    assert header.index("FPR") < header.index("Recall")
    assert "| test |" in table and "cues" in table.lower()


def test_dialogues_that_cannot_be_styled_are_skipped_and_counted():
    dialogues = [_d("s1", "Назовите код", risk=0.9), _d("n1", "Kaspi", risk=0.05)]  # n1 is Latin-only: nothing left under drop_latin
    result = paired_split(dialogues, score_fn=_score_fn({"Назовите код", "назовите код"}), alert_threshold=0.5,
                          cue_hits=lambda t: set(), reassures=lambda t: False, drop_latin=True)
    assert result["n_positives"] == 1 and result["n_negatives"] == 0
    assert result["skipped_ids"] == ("n1",)
    assert "1 skipped" in render_table({"test": result})
