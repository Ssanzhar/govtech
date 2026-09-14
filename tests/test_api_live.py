"""TDD tests for `qorgan.api_live` — the live-call replay HTTP API.

Always forces `backend: "mock"` for determinism (no model weights / network needed): the
`mock` backend's taxonomy-keyword heuristic deterministically flags the bundled scam demo
call and keeps the hard-negative call clear (the FPR demo scene).
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from qorgan.api import app
from qorgan.data.demo_transcripts import LIVE_DEMO_CALLS


@pytest.fixture()
def client() -> TestClient:
    return TestClient(app)


def _create_session(client: TestClient, *, locale: str = "ru", backend: str = "mock") -> str:
    res = client.post("/api/live/session", json={"locale": locale, "backend": backend})
    assert res.status_code == 200, res.text
    return res.json()["session_id"]


def _lines(scenario_id: str) -> list[str]:
    return [line.strip() for line in LIVE_DEMO_CALLS[scenario_id].splitlines() if line.strip()]


# --- scenarios --------------------------------------------------------------------------------


def test_scenarios_lists_both_demo_calls_with_nonempty_lines(client: TestClient) -> None:
    body = client.get("/api/live/scenarios").json()

    ids = {s["id"] for s in body["scenarios"]}
    assert {"live_scam_bank_ru", "live_hard_negative_bank_ru"} <= ids
    for scenario in body["scenarios"]:
        assert scenario["label"]
        assert scenario["lines"], f"{scenario['id']} must have non-empty lines"


# --- session creation ---------------------------------------------------------------------------


def test_create_session_returns_an_id(client: TestClient) -> None:
    res = client.post("/api/live/session", json={"locale": "ru", "backend": "mock"})

    assert res.status_code == 200
    body = res.json()
    assert body["session_id"]
    assert body["locale"] == "ru"
    assert body["backend"] == "mock"


def test_create_session_bad_locale_is_422(client: TestClient) -> None:
    res = client.post("/api/live/session", json={"locale": "en", "backend": "mock"})

    assert res.status_code == 422


def test_create_session_unknown_backend_is_4xx(client: TestClient) -> None:
    res = client.post("/api/live/session", json={"locale": "ru", "backend": "not_a_backend"})

    assert 400 <= res.status_code < 500


# --- replay: the two demo scenes -----------------------------------------------------------------


def test_scam_replay_escalates_to_high_or_critical_and_latches(client: TestClient) -> None:
    session_id = _create_session(client)

    last = None
    for line in _lines("live_scam_bank_ru"):
        res = client.post(f"/api/live/session/{session_id}/utterance", json={"text": line})
        assert res.status_code == 200, res.text
        last = res.json()

    assert last["band"] in {"high", "critical"}
    assert last["latched"] is True
    assert last["turn"] == len(_lines("live_scam_bank_ru"))


def test_hard_negative_replay_stays_clear(client: TestClient) -> None:
    session_id = _create_session(client)

    last = None
    for line in _lines("live_hard_negative_bank_ru"):
        res = client.post(f"/api/live/session/{session_id}/utterance", json={"text": line})
        assert res.status_code == 200, res.text
        last = res.json()

    assert last["band"] in {"low", "medium"}
    assert last["latched"] is False


def test_new_evidence_text_is_a_substring_of_the_just_posted_line(client: TestClient) -> None:
    session_id = _create_session(client)

    for line in _lines("live_scam_bank_ru"):
        res = client.post(f"/api/live/session/{session_id}/utterance", json={"text": line})
        body = res.json()
        for evidence in body["new_evidence"]:
            assert evidence["text"] in line


def test_turn_increments_each_utterance(client: TestClient) -> None:
    session_id = _create_session(client)
    lines = _lines("live_scam_bank_ru")

    for index, line in enumerate(lines, start=1):
        res = client.post(f"/api/live/session/{session_id}/utterance", json={"text": line})
        assert res.json()["turn"] == index


# --- errors -----------------------------------------------------------------------------------


def test_utterance_on_unknown_session_is_404(client: TestClient) -> None:
    res = client.post("/api/live/session/does-not-exist/utterance", json={"text": "hello"})

    assert res.status_code == 404


def test_blank_utterance_is_422(client: TestClient) -> None:
    session_id = _create_session(client)

    res = client.post(f"/api/live/session/{session_id}/utterance", json={"text": "   "})

    assert res.status_code == 422


# --- end / summary ------------------------------------------------------------------------------


def test_end_returns_summary_and_pops_session(client: TestClient) -> None:
    session_id = _create_session(client)
    lines = _lines("live_scam_bank_ru")
    last_meter = None
    for line in lines:
        res = client.post(f"/api/live/session/{session_id}/utterance", json={"text": line})
        last_meter = res.json()["meter"]

    res = client.post(f"/api/live/session/{session_id}/end")

    assert res.status_code == 200
    body = res.json()
    assert body["final_score"] == pytest.approx(last_meter)
    assert body["human_note"]
    assert body["band"] in {"low", "medium", "high", "critical"}


def test_ending_twice_is_404_the_second_time(client: TestClient) -> None:
    session_id = _create_session(client)
    client.post(f"/api/live/session/{session_id}/utterance", json={"text": "Алло."})

    first = client.post(f"/api/live/session/{session_id}/end")
    second = client.post(f"/api/live/session/{session_id}/end")

    assert first.status_code == 200
    assert second.status_code == 404


def test_end_on_unknown_session_is_404(client: TestClient) -> None:
    res = client.post("/api/live/session/does-not-exist/end")

    assert res.status_code == 404


# --- consent-gated report (→ analyst dashboard intake) ------------------------------------------


def _reports_file(tmp_path):
    return tmp_path / "processed" / "citizen_reports.jsonl"


def _tmp_data_dir(tmp_path, monkeypatch) -> None:
    """Point the data dir at tmp but keep the real committed taxonomy + lexicons."""
    monkeypatch.setenv("QORGAN_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("QORGAN_TAXONOMY_PATH", "data/taxonomy/tactics.yaml")
    monkeypatch.setenv("QORGAN_CUE_LEXICON_PATH", "data/lexicon/hard_signal_cues.yaml")
    monkeypatch.setenv(
        "QORGAN_REASSURANCE_PATTERNS_PATH", "data/lexicon/reassurance_patterns.yaml"
    )
    monkeypatch.setenv("QORGAN_NUMBER_HMAC_KEY", "tests-only-key")


def test_report_after_end_appends_a_citizen_report(
    client: TestClient, tmp_path, monkeypatch
) -> None:
    _tmp_data_dir(tmp_path, monkeypatch)
    session_id = _create_session(client)
    scam_line = _lines("live_scam_bank_ru")[0]
    client.post(f"/api/live/session/{session_id}/utterance", json={"text": scam_line})
    client.post(f"/api/live/session/{session_id}/end")

    res = client.post(
        f"/api/live/session/{session_id}/report", json={"phone_number": "+7 700 000 11 22"}
    )

    assert res.status_code == 200, res.text
    body = res.json()
    assert body["report_id"].startswith("report-")
    assert body["receipt_id"] == body["receipt_id"].lower() and len(body["receipt_id"]) == 24
    assert body["number_prefix"] == "+7 700 ***"
    lines = _reports_file(tmp_path).read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    from qorgan.reports.model import StoredReport

    stored = StoredReport.model_validate_json(lines[0])
    assert scam_line in stored.transcript
    assert stored.number_prefix == "+7 700 ***"
    assert "000 11 22" not in lines[0] and "0001122" not in lines[0]  # raw number never persisted


def test_report_works_without_calling_end_first(
    client: TestClient, tmp_path, monkeypatch
) -> None:
    _tmp_data_dir(tmp_path, monkeypatch)
    session_id = _create_session(client)
    client.post(f"/api/live/session/{session_id}/utterance", json={"text": "Алло, это банк."})

    res = client.post(f"/api/live/session/{session_id}/report", json={})

    assert res.status_code == 200
    assert _reports_file(tmp_path).exists()


def test_report_twice_is_404_the_second_time(client: TestClient, tmp_path, monkeypatch) -> None:
    _tmp_data_dir(tmp_path, monkeypatch)
    session_id = _create_session(client)
    client.post(f"/api/live/session/{session_id}/utterance", json={"text": "Алло, это банк."})
    client.post(f"/api/live/session/{session_id}/end")

    first = client.post(f"/api/live/session/{session_id}/report", json={})
    second = client.post(f"/api/live/session/{session_id}/report", json={})

    assert first.status_code == 200
    assert second.status_code == 404


def test_report_on_unknown_session_is_404(client: TestClient) -> None:
    res = client.post("/api/live/session/does-not-exist/report", json={})

    assert res.status_code == 404


def test_report_with_no_utterances_is_422(client: TestClient, tmp_path, monkeypatch) -> None:
    _tmp_data_dir(tmp_path, monkeypatch)
    session_id = _create_session(client)

    res = client.post(f"/api/live/session/{session_id}/report", json={})

    assert res.status_code == 422
    assert not _reports_file(tmp_path).exists()


def test_reported_call_shows_up_as_pending_on_the_admin_overview(
    client: TestClient, tmp_path, monkeypatch
) -> None:
    from qorgan.analytics.pipeline import write_organizations_jsonl
    from qorgan.data.incident_seed import write_incidents_jsonl
    from qorgan.data.schema import Incident, Label, Organization, TacticTag

    _tmp_data_dir(tmp_path, monkeypatch)
    incident = Incident(
        id="i1",
        dialogue_id="i1",
        transcript="это служба безопасности банка, продиктуйте код",
        label=Label(risk=0.9, tactic_tags=(TacticTag(id="impersonation_bank"),)),
    )
    processed = tmp_path / "processed"
    write_organizations_jsonl([Organization(id="org_0", members=("i1",))], processed / "organizations.jsonl")
    write_incidents_jsonl([incident], processed / "incidents.jsonl")

    session_id = _create_session(client)
    client.post(
        f"/api/live/session/{session_id}/utterance",
        json={"text": _lines("live_scam_bank_ru")[0]},
    )
    client.post(f"/api/live/session/{session_id}/end")
    client.post(f"/api/live/session/{session_id}/report", json={"phone_number": "+7 707 1 2 3"})

    body = client.get("/api/admin/overview").json()

    assert body["kpis"]["pending_reports"] == 1


def test_report_with_a_number_is_refused_when_no_hashing_key_is_configured(
    client: TestClient, tmp_path, monkeypatch
) -> None:
    """Never store a raw number: a misconfigured server refuses instead (ADR D14)."""
    _tmp_data_dir(tmp_path, monkeypatch)
    monkeypatch.setenv("QORGAN_NUMBER_HMAC_KEY", "")
    session_id = _create_session(client)
    client.post(f"/api/live/session/{session_id}/utterance", json={"text": _lines("live_scam_bank_ru")[0]})

    res = client.post(f"/api/live/session/{session_id}/report", json={"phone_number": "+7 700 000 11 22"})

    assert res.status_code == 503
    assert not _reports_file(tmp_path).exists()


def test_report_without_a_number_needs_no_key(client: TestClient, tmp_path, monkeypatch) -> None:
    _tmp_data_dir(tmp_path, monkeypatch)
    monkeypatch.setenv("QORGAN_NUMBER_HMAC_KEY", "")
    session_id = _create_session(client)
    client.post(f"/api/live/session/{session_id}/utterance", json={"text": _lines("live_scam_bank_ru")[0]})

    res = client.post(f"/api/live/session/{session_id}/report", json={})

    assert res.status_code == 200, res.text
    assert res.json()["number_prefix"] is None


def test_report_with_an_unparseable_number_is_a_client_error(client: TestClient, tmp_path, monkeypatch) -> None:
    _tmp_data_dir(tmp_path, monkeypatch)
    session_id = _create_session(client)
    client.post(f"/api/live/session/{session_id}/utterance", json={"text": _lines("live_scam_bank_ru")[0]})

    res = client.post(f"/api/live/session/{session_id}/report", json={"phone_number": "call me"})

    assert res.status_code == 422
