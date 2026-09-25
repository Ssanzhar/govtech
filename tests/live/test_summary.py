"""TDD tests for `qorgan.live.summary` — post-call summary + consent-gated report
(design spec §§07, 11).

The report is a *draft*: built only when asked, every field editable, submitted only by
an explicit `submit_report()` call — nothing here fires automatically.
"""

from datetime import UTC, datetime

import pytest

from qorgan.asr.stream import CommittedUtterance
from qorgan.data.schema import validate_verbatim_spans
from qorgan.live.session import advance, initial_session
from qorgan.live.summary import (
    CallSummary,
    ReportDraft,
    build_report,
    report_to_incident,
    submit_report,
    summarize,
)

KEY = b"tests-only-key"

SCAM_TURNS = (
    "Алло, здравствуйте. Это служба безопасности вашего банка.",
    "Зафиксирована подозрительная операция, действовать нужно прямо сейчас.",
    "Продиктуйте код из SMS для отмены операции.",
    "И переведите деньги на безопасный счёт.",
)


def _scam_state(locale: str = "ru"):
    state = initial_session(locale, backend="mock")
    for text in SCAM_TURNS:
        state, _update = advance(state, CommittedUtterance(text=text, confidence=1.0))
    return state


def _benign_state():
    state = initial_session("ru", backend="mock")
    state, _update = advance(
        state, CommittedUtterance(text="Привет! Как выходные прошли?", confidence=1.0)
    )
    return state


# --- summarize ------------------------------------------------------------------------------


def test_scam_summary_reports_score_band_tactics_and_actions():
    summary = summarize(_scam_state())

    assert summary.final_score >= 81.0
    assert summary.band == "critical"
    assert summary.tactic_names  # localized display names, not raw ids
    assert all(" " in name or name.istitle() for name in summary.tactic_names)
    assert summary.recommended_actions
    assert summary.human_note  # "a human decides" is part of the contract


def test_benign_summary_is_calm_and_actionless():
    summary = summarize(_benign_state())

    assert summary.band == "low"
    assert summary.tactic_names == ()
    assert summary.recommended_actions == ()


def test_summary_is_localized():
    ru = summarize(_scam_state("ru"))
    kk = summarize(_scam_state("kk"))

    assert ru.tactic_names != kk.tactic_names


# --- build_report ----------------------------------------------------------------------------


def test_report_draft_carries_all_editable_fields():
    state = _scam_state()
    draft = build_report(state, phone_number="+7 700 000 00 00")

    assert draft.phone_number == "+7 700 000 00 00"
    assert draft.transcript == state.transcript()
    assert draft.flagged_phrases  # evidence surfaced during the call
    assert "otp_request" in draft.tactic_ids
    assert draft.risk_score >= 81.0
    assert draft.timestamp.tzinfo is not None  # tz-aware default


def test_report_timestamp_is_overridable():
    stamp = datetime(2026, 7, 14, 12, 0, tzinfo=UTC)
    draft = build_report(_scam_state(), timestamp=stamp)

    assert draft.timestamp == stamp


# --- report_to_incident -----------------------------------------------------------------------


def test_report_converts_to_a_valid_l2_incident(tmp_path):
    draft = build_report(_scam_state(), phone_number="+7 700 000 00 00")
    stored = submit_report(draft, hmac_key=KEY, reports_path=tmp_path / "r.jsonl")
    incident = report_to_incident(stored, incident_id="report-001")

    assert incident.id == "report-001"
    assert incident.transcript == stored.transcript
    assert incident.number_hash == stored.number_hash and incident.number_prefix == "+7 700 ***"
    assert incident.label.risk == pytest.approx(draft.risk_score / 100.0)
    validate_verbatim_spans(incident.label.trigger_spans, incident.transcript)


def test_incident_drops_phrases_not_verbatim_in_transcript(tmp_path):
    draft = build_report(_scam_state())
    edited = draft.model_copy(update={"flagged_phrases": ("не из этого разговора",)})

    stored = submit_report(edited, hmac_key=KEY, reports_path=tmp_path / "r.jsonl")
    incident = report_to_incident(stored, incident_id="report-002")

    assert incident.label.trigger_spans == ()  # never fabricate evidence


# --- submit_report ----------------------------------------------------------------------------


def test_submit_report_appends_one_jsonl_line_per_submission(tmp_path):
    import json

    reports_path = tmp_path / "citizen_reports.jsonl"
    draft = build_report(_scam_state(), phone_number="+7 700 000 00 00")

    submit_report(draft, hmac_key=KEY, reports_path=reports_path)
    submit_report(draft.model_copy(update={"phone_number": None}), hmac_key=KEY, reports_path=reports_path)

    lines = reports_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    first = json.loads(lines[0])
    assert "phone_number" not in first  # the raw number is never written
    assert first["number_prefix"] == "+7 700 ***" and len(first["number_hash"]) == 24
    assert "000 00 00" not in lines[0] and "0000000" not in lines[0]
    assert first["risk_score"] >= 81.0
    assert json.loads(lines[1])["number_hash"] is None


def test_submit_report_with_a_number_requires_the_hashing_key(tmp_path):
    from qorgan.privacy.numbers import MissingHmacKeyError

    draft = build_report(_scam_state(), phone_number="+7 700 000 00 00")
    with pytest.raises(MissingHmacKeyError):
        submit_report(draft, hmac_key=None, reports_path=tmp_path / "r.jsonl")
    assert not (tmp_path / "r.jsonl").exists()


def test_models_are_frozen():
    summary = summarize(_benign_state())
    draft = build_report(_benign_state())

    with pytest.raises(Exception):
        summary.band = "critical"  # type: ignore[misc]
    with pytest.raises(Exception):
        draft.transcript = "edited"  # type: ignore[misc]
