"""The court_quotes eval harness (ADR D68): one-turn replay, rates with exact intervals, and a
report that never prints quote text. Runs on the deterministic `mock` backend."""

import pytest

from qorgan.data.court_quotes import QuoteRow
from qorgan.eval.court_quotes import rate, render, score_quotes, stream_verdicts, summarize


def _row(i: int, text: str, verdict: str = "v1", speech: str = "reported") -> QuoteRow:
    return QuoteRow(
        id=f"cq-{i}", language="ru", text=text, speech=speech, tactic_guess="", source="sudact.ru",
        source_url="u", court="c", date="2025-01-01", verdict_id=verdict,
    )


_ROWS = (
    _row(0, "назовите код из смс, который вам сейчас придет"),
    _row(1, "ваши деньги нужно срочно перевести на безопасный счет", speech="direct"),
    _row(2, "у вас всё в порядке, хорошего дня", verdict="v2"),
)


def _no_cues(_text: str) -> tuple[str, ...]:
    return ()


def test_every_quote_is_scored_once_with_its_ids():
    scores = score_quotes(_ROWS, cue_fn=_no_cues, backend="mock")
    assert [s.id for s in scores] == ["cq-0", "cq-1", "cq-2"]
    assert all(0.0 <= s.risk <= 1.0 and 0.0 <= s.meter_score <= 100.0 for s in scores)


def test_cue_firing_comes_from_the_injected_matcher():
    scores = score_quotes(_ROWS, cue_fn=lambda t: ("otp_request",) if "код" in t else (), backend="mock")
    assert [s.cue_fired for s in scores] == [True, False, False]
    assert scores[0].cue_tactics == ("otp_request",)


def test_rate_carries_a_clopper_pearson_interval():
    r = rate(3, 10)
    assert r.value == pytest.approx(0.3)
    assert 0.0 < r.low < 0.3 < r.high < 1.0
    assert rate(0, 0).n == 0


def test_summary_and_report_hold_numbers_not_text():
    scores = score_quotes(_ROWS, cue_fn=_no_cues, backend="mock")
    summary = summarize(scores)
    assert summary["risk>=threshold"].n == 3
    report = render(scores, stream_verdicts(_ROWS, backend="mock"), "mock")
    for row in _ROWS:
        assert row.text not in report
    assert "speech=direct" in report and "pseudo-calls" in report


def test_pseudo_calls_group_by_verdict():
    assert set(stream_verdicts(_ROWS, backend="mock")) == {"v1", "v2"}
