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
    at = AppTest.from_file(str(APP_PATH))

    at.run(timeout=_RUN_TIMEOUT)

    assert not at.exception
    assert at.title[0].value.startswith("Qorgan")
    assert any("offline demo" in info.value for info in at.info)


def test_app_scores_a_bundled_demo_transcript_end_to_end(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "")
    monkeypatch.setenv("GOOGLE_API_KEY", "")
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
    at = AppTest.from_file(str(APP_PATH))
    at.run(timeout=_RUN_TIMEOUT)

    at.selectbox[0].select("hard_negative_bank_call_ru").run(timeout=_RUN_TIMEOUT)
    at.button[0].click().run(timeout=_RUN_TIMEOUT)

    assert not at.exception
    assert any("Low risk" in ok.value for ok in at.success)
    assert len(at.error) == 0
