"""TDD tests for `qorgan.eval.adversarial` (PLAN A9): paired source-vs-paraphrase recall
with intervals, per language, and the recall-drop gate."""

from __future__ import annotations

from qorgan.data.adversarial import ADVERSARIAL_ID_PREFIX
from qorgan.data.schema import Dialogue, Label, ScoreResult, TacticTag, Utterance
from qorgan.eval.adversarial import RECALL_DROP_GATE_POINTS, paired_recall, render_table


def _d(dialogue_id: str, text: str, language: str = "ru") -> Dialogue:
    return Dialogue(
        id=dialogue_id, language=language, utterances=(Utterance(speaker="caller", text=text),),
        label=Label(risk=0.9, tactic_tags=(TacticTag(id="otp_request"),)),
    )


def _score_fn(flag_texts: set[str]):
    def score(text: str) -> ScoreResult:
        return ScoreResult(risk=0.95 if text in flag_texts else 0.1, backend="fake")
    return score


def test_paired_recall_pairs_by_id_and_reports_the_drop():
    sources = [_d("s1", "source one"), _d("s2", "source two", "kk"), _d("s3", "source three")]
    adversarial = [_d(f"{ADVERSARIAL_ID_PREFIX}s1", "adv one"), _d(f"{ADVERSARIAL_ID_PREFIX}s2", "adv two", "kk")]
    score = _score_fn({"source one", "source two", "source three", "adv one"})

    result = paired_recall(sources, adversarial, score_fn=score, alert_threshold=0.5)

    assert result["n_pairs"] == 2
    assert result["source"]["recall"] == 1.0 and result["adversarial"]["recall"] == 0.5
    assert result["recall_drop_points"] == 50.0
    assert result["flips_to_clear"] == 1 and result["flips_to_scam"] == 0
    assert result["source"]["interval"].low <= 1.0 <= result["source"]["interval"].high
    assert result["by_language"]["kk"]["adversarial"]["recall"] == 0.0
    assert result["gate_failed"] is True
    assert RECALL_DROP_GATE_POINTS == 15.0


def test_render_table_mentions_the_gate():
    sources = [_d("s1", "a")]
    adversarial = [_d(f"{ADVERSARIAL_ID_PREFIX}s1", "b")]
    result = paired_recall(sources, adversarial, score_fn=_score_fn({"a", "b"}), alert_threshold=0.5)
    table = render_table(result)
    assert "| Set | N | Recall [95% CI] |" in table and "drop" in table.lower()
    assert result["gate_failed"] is False and "within the 15-point gate" in table
