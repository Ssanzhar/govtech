"""TDD tests for `qorgan.api_live_ws` — the microphone WebSocket.

Scripted fake recognizers (the vosk `KaldiRecognizer` contract, as in
`tests/asr/test_vosk_stream.py`) are injected through the `_RECOGNIZER_FACTORY` test
seam — no real vosk models, no audio hardware. The classifier backend is always the
deterministic `mock`.
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from qorgan.api import app

SCAM_TEXT = "это служба безопасности банка переведите деньги на безопасный счёт"
CHUNK = b"\x00\x01" * 160


class FakeRecognizer:
    """Scripted vosk.KaldiRecognizer stand-in (AcceptWaveform/Result/PartialResult/FinalResult)."""

    def __init__(self, accepts=(), results=(), partials=(), finals=()):
        self.accepts = list(accepts)
        self.results = list(results)
        self.partials = list(partials)
        self.finals = list(finals)

    def AcceptWaveform(self, chunk):  # noqa: N802 - vosk API name
        return self.accepts.pop(0) if self.accepts else False

    def Result(self):  # noqa: N802
        return self.results.pop(0) if self.results else "{}"

    def PartialResult(self):  # noqa: N802
        return self.partials.pop(0) if self.partials else json.dumps({"partial": ""})

    def FinalResult(self):  # noqa: N802
        return self.finals.pop(0) if self.finals else "{}"


def _result(text: str) -> str:
    return json.dumps(
        {"text": text, "result": [{"word": word, "conf": 0.9} for word in text.split()]}
    )


def _use_fakes(monkeypatch, *, ru: FakeRecognizer, kk: FakeRecognizer | None = None) -> None:
    import qorgan.api_live_ws as ws_mod

    silent_kk = kk or FakeRecognizer()
    monkeypatch.setattr(ws_mod, "_RECOGNIZER_FACTORY", lambda: {"kk": silent_kk, "ru": ru})


@pytest.fixture()
def client() -> TestClient:
    return TestClient(app)


def _start(ws, *, locale: str = "ru", backend: str | None = "mock") -> dict:
    ws.send_text(json.dumps({"type": "start", "locale": locale, "backend": backend}))
    return ws.receive_json()


# --- capabilities -------------------------------------------------------------------------------


def test_capabilities_reports_microphone_flag(client: TestClient) -> None:
    body = client.get("/api/live/capabilities").json()

    assert isinstance(body["microphone"], bool)


# --- protocol -----------------------------------------------------------------------------------


def test_ws_ready_carries_session_and_sample_rate(client: TestClient, monkeypatch) -> None:
    _use_fakes(monkeypatch, ru=FakeRecognizer())

    with client.websocket_connect("/api/live/ws") as ws:
        ready = _start(ws)
        ws.send_text(json.dumps({"type": "stop"}))
        assert ready["type"] == "ready"
        assert ready["session_id"]
        assert ready["backend"] == "mock"
        assert ready["sample_rate"] == 16000


def test_ws_scam_audio_drives_the_meter_and_summary(client: TestClient, monkeypatch) -> None:
    _use_fakes(monkeypatch, ru=FakeRecognizer(accepts=[True], results=[_result(SCAM_TEXT)]))

    with client.websocket_connect("/api/live/ws") as ws:
        _start(ws)
        ws.send_bytes(CHUNK)
        ws.send_text(json.dumps({"type": "stop"}))

        utterance = ws.receive_json()
        assert utterance["type"] == "utterance"
        assert utterance["text"] == SCAM_TEXT
        assert utterance["language"] == "ru"
        assert utterance["turn"] == 1
        assert utterance["meter"] > 0

        summary = ws.receive_json()
        assert summary["type"] == "summary"
        assert summary["final_score"] == pytest.approx(utterance["meter"])
        assert summary["human_note"]


def test_ws_forwards_partials(client: TestClient, monkeypatch) -> None:
    ru = FakeRecognizer(
        accepts=[False], partials=[json.dumps({"partial": "это служба"})]
    )
    _use_fakes(monkeypatch, ru=ru)

    with client.websocket_connect("/api/live/ws") as ws:
        _start(ws)
        ws.send_bytes(CHUNK)
        ws.send_text(json.dumps({"type": "stop"}))

        first = ws.receive_json()
        assert first["type"] == "partial"
        assert first["text"] == "это служба"


def test_ws_unknown_backend_yields_error(client: TestClient, monkeypatch) -> None:
    _use_fakes(monkeypatch, ru=FakeRecognizer())

    with client.websocket_connect("/api/live/ws") as ws:
        reply = _start(ws, backend="not_a_backend")

        assert reply["type"] == "error"
        assert reply["message"]


def test_ws_without_vosk_yields_error(client: TestClient, monkeypatch) -> None:
    import qorgan.api_live_ws as ws_mod

    monkeypatch.setattr(ws_mod, "_RECOGNIZER_FACTORY", None)
    monkeypatch.setattr(ws_mod, "_vosk_available", lambda: False)

    with client.websocket_connect("/api/live/ws") as ws:
        reply = _start(ws)

        assert reply["type"] == "error"
        assert "live" in reply["message"]


def test_ws_session_is_reportable_after_summary(
    client: TestClient, tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("QORGAN_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("QORGAN_TAXONOMY_PATH", "data/taxonomy/tactics.yaml")
    _use_fakes(monkeypatch, ru=FakeRecognizer(accepts=[True], results=[_result(SCAM_TEXT)]))

    with client.websocket_connect("/api/live/ws") as ws:
        ready = _start(ws)
        ws.send_bytes(CHUNK)
        ws.send_text(json.dumps({"type": "stop"}))
        ws.receive_json()  # utterance
        summary = ws.receive_json()
        assert summary["type"] == "summary"

    res = client.post(
        f"/api/live/session/{ready['session_id']}/report",
        json={"phone_number": "+7 700 000 11 22"},
    )

    assert res.status_code == 200, res.text
    reports = (tmp_path / "processed" / "citizen_reports.jsonl").read_text(encoding="utf-8")
    assert SCAM_TEXT in reports
