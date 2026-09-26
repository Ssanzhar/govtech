"""SMOKE test for `app/streamlit_app.py` — runs the real script via Streamlit's AppTest
harness (no server, no browser, no network/API key needed since the mock backend
auto-activates with no GEMINI_API_KEY)."""

from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

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


def _write_l2_fixture(data_dir):
    """Two organizations (one novel) + tagged, dated incidents for the analyst tab."""
    from datetime import UTC, datetime

    from support.numbers import hashed
    from qorgan.data.schema import Incident, Label, Organization, Span, TacticTag

    processed = data_dir / "processed"
    processed.mkdir(parents=True)
    transcript = "Это служба безопасности вашего банка. Продиктуйте код из SMS."
    phrase = "Продиктуйте код из SMS"
    start = transcript.index(phrase)
    incidents = [
        Incident(
            id=f"i{n}",
            dialogue_id=f"i{n}",
            transcript=transcript,
            label=Label(
                risk=0.9,
                tactic_tags=(TacticTag(id="impersonation_bank"), TacticTag(id="otp_request")),
                trigger_spans=(Span(text=phrase, start=start, end=start + len(phrase)),),
            ),
            number_hash=hashed("+7 700 101 20 30"), number_prefix="+7 700 ***",
            timestamp=datetime(2026, 7, 10 + n, 12, 0, tzinfo=UTC),
        )
        for n in range(3)
    ] + [
        Incident(
            id="i9",
            dialogue_id="i9",
            transcript="Раздача криптовалюты, отправьте монеты на кошелёк.",
            label=Label(risk=0.9, tactic_tags=(TacticTag(id="payment_redirect"),)),
            number_hash=hashed("+7 708 909 10 11"), number_prefix="+7 708 ***",
            timestamp=datetime(2026, 7, 14, 9, 0, tzinfo=UTC),
        )
    ]
    organizations = [
        Organization(
            id="org_0",
            members=("i0", "i1", "i2"),
            numbers=("+7 700 101 20 30",),
            representative_script=transcript,
            priority=0.82,
        ),
        Organization(
            id="org_1",
            members=("i9",),
            numbers=("+7 708 909 10 11",),
            representative_script="Раздача криптовалюты, отправьте монеты на кошелёк.",
            priority=0.4,
            is_novel=True,
        ),
    ]
    (processed / "incidents.jsonl").write_text(
        "\n".join(i.model_dump_json() for i in incidents) + "\n", encoding="utf-8"
    )
    (processed / "organizations.jsonl").write_text(
        "\n".join(o.model_dump_json() for o in organizations) + "\n", encoding="utf-8"
    )


def test_analyst_tab_renders_kpis_names_and_drilldown(monkeypatch, tmp_path):
    data_dir = tmp_path / "data"
    _write_l2_fixture(data_dir)
    monkeypatch.setenv("QORGAN_DATA_DIR", str(data_dir))
    monkeypatch.setenv("QORGAN_TAXONOMY_PATH", "data/taxonomy/tactics.yaml")
    monkeypatch.setenv("GEMINI_API_KEY", "")
    monkeypatch.setenv("GOOGLE_API_KEY", "")
    monkeypatch.setenv("QORGAN_CLASSIFIER_BACKEND", "mock")
    at = AppTest.from_file(str(APP_PATH))

    at.run(timeout=_RUN_TIMEOUT)

    assert not at.exception
    metric_values = [m.value for m in at.metric]
    assert "4" in metric_values  # incidents analyzed
    assert "2" in metric_values  # organizations
    body = " ".join(m.value for m in at.markdown)
    assert "Выдаёт себя за банк" in body  # tactic-derived display name, not org_0
    assert any("NEW SCHEME" in w.value for w in at.warning)  # novel callout


def test_analyst_tab_offers_and_wires_ingest_for_pending_reports(monkeypatch, tmp_path):
    from datetime import UTC, datetime

    from qorgan.analytics.intake import IngestSummary, Placement
    from support.numbers import stored_report

    data_dir = tmp_path / "data"
    _write_l2_fixture(data_dir)
    draft = stored_report(
        number="+7 700 101 20 30", flagged_phrases=(), tactic_ids=(),
        timestamp=datetime(2026, 7, 15, 10, 0, tzinfo=UTC), risk_score=84.0,
    )
    (data_dir / "processed" / "citizen_reports.jsonl").write_text(
        draft.model_dump_json() + "\n", encoding="utf-8"
    )
    monkeypatch.setenv("QORGAN_DATA_DIR", str(data_dir))
    monkeypatch.setenv("QORGAN_TAXONOMY_PATH", "data/taxonomy/tactics.yaml")
    monkeypatch.setenv("GEMINI_API_KEY", "")
    monkeypatch.setenv("GOOGLE_API_KEY", "")
    monkeypatch.setenv("QORGAN_CLASSIFIER_BACKEND", "mock")

    # Stub the ingest so the test never loads the real embedding model; the wiring
    # (button -> ingest -> rerun -> success banner with org names) is what's under test.
    import app.analyst_view as analyst_view

    stub_summary = IngestSummary(
        ingested=1,
        placements=(Placement(incident_id="report-abc123", org_id="org_0", org_is_novel=False),),
        organizations_total=2,
    )
    monkeypatch.setattr(analyst_view, "ingest_pending", lambda **_kwargs: stub_summary)

    at = AppTest.from_file(str(APP_PATH))
    at.run(timeout=_RUN_TIMEOUT)

    assert not at.exception
    assert any("awaiting analysis" in info.value for info in at.info)
    assert any("1" == m.value for m in at.metric)  # pending KPI

    ingest_button = next(b for b in at.button if b.label == "Ingest into analysis")
    ingest_button.click().run(timeout=_RUN_TIMEOUT)

    assert not at.exception
    assert any("Ingested" in s.value and "report-abc123" in s.value for s in at.success)


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
