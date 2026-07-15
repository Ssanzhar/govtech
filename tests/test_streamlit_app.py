"""SMOKE test for `app/streamlit_app.py` — runs the real script via Streamlit's AppTest
harness (no server, no browser, no network/API key needed since the mock backend
auto-activates with no GEMINI_API_KEY)."""

from pathlib import Path

from streamlit.testing.v1 import AppTest

APP_PATH = Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"

# Generous timeout: the first AppTest run in the process pays a one-time import/compile
# cost for qorgan + its dependencies, which can exceed AppTest's 3s default.
_RUN_TIMEOUT = 30


def test_app_imports_and_renders_with_defaults(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "")
    monkeypatch.setenv("GOOGLE_API_KEY", "")
    monkeypatch.setenv("QORGAN_CLASSIFIER_BACKEND", "mock")
    at = AppTest.from_file(str(APP_PATH))

    at.run(timeout=_RUN_TIMEOUT)

    assert not at.exception
    assert at.title[0].value.startswith("Qorgan")
    assert any("offline demo" in info.value for info in at.info)


def test_app_scores_a_bundled_demo_transcript_end_to_end(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "")
    monkeypatch.setenv("GOOGLE_API_KEY", "")
    monkeypatch.setenv("QORGAN_CLASSIFIER_BACKEND", "mock")
    at = AppTest.from_file(str(APP_PATH))
    at.run(timeout=_RUN_TIMEOUT)

    at.selectbox[0].select("scam_bank_ru").run(timeout=_RUN_TIMEOUT)
    at.button[0].click().run(timeout=_RUN_TIMEOUT)

    assert not at.exception
    assert at.metric[0].value == "96%"
    assert any("High risk" in err.value for err in at.error)
    assert any("otp_request" in write.value for write in at.markdown)


def test_app_hard_negative_does_not_trigger_alert(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "")
    monkeypatch.setenv("GOOGLE_API_KEY", "")
    monkeypatch.setenv("QORGAN_CLASSIFIER_BACKEND", "mock")
    at = AppTest.from_file(str(APP_PATH))
    at.run(timeout=_RUN_TIMEOUT)

    at.selectbox[0].select("hard_negative_bank_call_ru").run(timeout=_RUN_TIMEOUT)
    at.button[0].click().run(timeout=_RUN_TIMEOUT)

    assert not at.exception
    assert any("Low risk" in ok.value for ok in at.success)
    assert len(at.error) == 0


def test_live_tab_scam_replay_reaches_critical_and_offers_report(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "")
    monkeypatch.setenv("GOOGLE_API_KEY", "")
    monkeypatch.setenv("QORGAN_CLASSIFIER_BACKEND", "mock")
    at = AppTest.from_file(str(APP_PATH))
    at.run(timeout=_RUN_TIMEOUT)

    at.selectbox[1].select("live_scam_bank_ru").run(timeout=_RUN_TIMEOUT)
    at.button[1].click().run(timeout=_RUN_TIMEOUT)

    assert not at.exception
    final_state = at.session_state["live_final_state"]
    assert final_state.meter.score >= 81.0  # two+ hard signals -> Critical floor
    assert final_state.meter.latched is True
    assert any("Post-call summary" in h.value for h in at.subheader)
    assert any("Report this call" in h.value for h in at.subheader)
    assert len(at.error) >= 1  # the live warning banner fired during the call


def test_live_tab_hard_negative_replay_stays_low_and_quiet(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "")
    monkeypatch.setenv("GOOGLE_API_KEY", "")
    monkeypatch.setenv("QORGAN_CLASSIFIER_BACKEND", "mock")
    at = AppTest.from_file(str(APP_PATH))
    at.run(timeout=_RUN_TIMEOUT)

    at.selectbox[1].select("live_hard_negative_bank_ru").run(timeout=_RUN_TIMEOUT)
    at.button[1].click().run(timeout=_RUN_TIMEOUT)

    assert not at.exception
    final_state = at.session_state["live_final_state"]
    assert final_state.meter.score <= 30.0  # stays in the Low band
    assert final_state.meter.latched is False
    assert len(at.error) == 0  # the real-bank-call scene must not trigger


def test_live_tab_mic_mode_degrades_gracefully_without_optional_deps(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "")
    monkeypatch.setenv("GOOGLE_API_KEY", "")
    monkeypatch.setenv("QORGAN_CLASSIFIER_BACKEND", "mock")
    import app.mic_live as mic_live

    monkeypatch.setattr(mic_live, "_find_spec", lambda _name: None)  # simulate no [live] extras
    at = AppTest.from_file(str(APP_PATH))
    at.run(timeout=_RUN_TIMEOUT)

    at.radio[2].set_value("Microphone — browser").run(timeout=_RUN_TIMEOUT)

    assert not at.exception
    assert any('pip install -e ".[live]"' in info.value for info in at.info)


def test_live_tab_local_mic_mode_renders_when_live_extras_installed(monkeypatch):
    """Positive path for the mic dispatch + deps check + fragment. Deliberately the
    *local* vector: webrtc_streamer requires the real Streamlit runtime and cannot run
    under AppTest's mocked one — the browser vector is verified on a live server."""
    import pytest

    pytest.importorskip("vosk")
    pytest.importorskip("sounddevice")
    monkeypatch.setenv("GEMINI_API_KEY", "")
    monkeypatch.setenv("GOOGLE_API_KEY", "")
    monkeypatch.setenv("QORGAN_CLASSIFIER_BACKEND", "mock")
    at = AppTest.from_file(str(APP_PATH))
    at.run(timeout=_RUN_TIMEOUT)

    at.radio[2].set_value("Microphone — local").run(timeout=_RUN_TIMEOUT)

    assert not at.exception
    assert any("Start recording" in b.label for b in at.button)  # mic not started in tests


def test_app_level2_tab_does_not_crash_on_empty_organizations_file(monkeypatch, tmp_path):
    data_dir = tmp_path / "data"
    processed_dir = data_dir / "processed"
    processed_dir.mkdir(parents=True)
    (processed_dir / "organizations.jsonl").write_text("\n", encoding="utf-8")

    monkeypatch.setenv("QORGAN_DATA_DIR", str(data_dir))
    monkeypatch.setenv("GEMINI_API_KEY", "")
    monkeypatch.setenv("GOOGLE_API_KEY", "")
    monkeypatch.setenv("QORGAN_CLASSIFIER_BACKEND", "mock")
    at = AppTest.from_file(str(APP_PATH))

    at.run(timeout=_RUN_TIMEOUT)

    assert not at.exception
    assert any("no organizations" in info.value.lower() for info in at.info)
