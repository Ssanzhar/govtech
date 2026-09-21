"""Tests for `qorgan.web.client_config` -- everything the JS core needs besides weights,
exported from the same sources Python uses (taxonomy, templates, advice, config) so the
two implementations cannot drift."""

import json
from pathlib import Path

from qorgan.config import get_config
from qorgan.explain.recommend import load_advice
from qorgan.explain.templates import load_templates
from qorgan.taxonomy import get_taxonomy
from qorgan.web.client_config import CLIENT_CONFIG_FILENAME, export_client_config, write_client_config


def test_export_is_json_and_mirrors_python_sources():
    data = export_client_config()
    json.dumps(data, ensure_ascii=False)
    taxonomy = get_taxonomy()
    assert data["format_version"] == 1
    assert [t["id"] for t in data["taxonomy"]["tactics"]] == list(taxonomy.tactic_ids())
    assert data["taxonomy"]["hard_signal_ids"] == list(taxonomy.hard_signal_ids())
    by_id = {t["id"]: t for t in data["taxonomy"]["tactics"]}
    assert by_id["otp_request"]["names"]["ru"] == taxonomy.display_name("otp_request", "ru")
    assert by_id["otp_request"]["names"]["kk"] == taxonomy.display_name("otp_request", "kk")
    assert set(data["locales"]) == set(get_config().supported_locales)


def test_export_carries_templates_and_advice_per_locale():
    data = export_client_config()
    from pathlib import Path

    templates_dir = Path(__file__).resolve().parents[2] / "src" / "qorgan" / "explain"
    for locale in ("ru", "kk"):
        t = load_templates(locale, templates_dir)
        assert data["templates"][locale]["reason_template"] == t.reason_template
        assert data["templates"][locale]["confidence"] == t.confidence
        a = load_advice(locale)
        assert data["advice"][locale]["tactic_advice"] == a.tactic_advice
        assert data["advice"][locale]["verification_questions"] == list(a.verification_questions)
        assert data["advice"][locale]["low_confidence_note"] == a.low_confidence_note


def test_export_carries_meter_and_scoring_constants():
    data = export_client_config()
    cfg = get_config()
    m = data["meter"]
    assert m["alpha_up"] == cfg.meter_alpha_up and m["alpha_down"] == cfg.meter_alpha_down
    assert m["enter"] == cfg.risk_threshold_enter and m["exit"] == cfg.risk_threshold_exit
    assert m["bands"] == {"low_max": 30.0, "medium_max": 60.0, "high_max": 80.0, "score_max": 100.0}
    assert m["hard_signal_confidence_floor"] == 0.8
    assert m["single_hard_signal_floor"] == 61.0 and m["double_hard_signal_floor"] == 81.0
    s = data["scoring"]
    assert s["risk_threshold"] == cfg.risk_threshold
    assert s["attribution_top_k"] == 3 and s["attribution_min_score"] == 0.5
    assert s["calibrated_backends"] == ["linear", "xlmr"]
    assert s["confidence_bands"] == {"high": 0.8, "medium": 0.6}
    assert s["low_confidence_floor"] == 0.6
    w = data["window"]
    assert w["max_chars"] == 6000 and w["head_utterances"] == 4 and w["join"] == "\n"


def test_write_is_deterministic(tmp_path):
    a = write_client_config(tmp_path / "a.json").read_bytes()
    b = write_client_config(tmp_path / "b.json").read_bytes()
    assert a == b and CLIENT_CONFIG_FILENAME.endswith(".json")


def test_committed_client_config_is_current():
    """`site/core/qorgan-config.json` must be regenerated after any taxonomy / template /
    advice / config change: `python -m qorgan.web.client_config`."""
    from pathlib import Path

    committed = Path(__file__).resolve().parents[2] / "site" / "core" / CLIENT_CONFIG_FILENAME
    assert committed.exists(), "run: python -m qorgan.web.client_config"
    assert json.loads(committed.read_text(encoding="utf-8")) == export_client_config()


def test_served_weights_match_the_trained_bundle_when_present():
    """`site/models/weights.json` is what the browser loads; it must be the export of the
    bundle in `models/linear` (regenerate with scripts/export_parity_fixtures.py)."""
    import hashlib
    from pathlib import Path

    import pytest

    root = Path(__file__).resolve().parents[2]
    trained = root / "models" / "linear" / "web" / "weights.json"
    served = root / "site" / "models" / "weights.json"
    if not trained.exists():
        pytest.skip("no local trained bundle")
    assert served.exists()
    assert hashlib.sha256(served.read_bytes()).hexdigest() == hashlib.sha256(trained.read_bytes()).hexdigest()


def test_export_names_the_embedder_the_browser_must_load(monkeypatch):
    from qorgan.web.client_config import web_model_id

    monkeypatch.setenv("QORGAN_EMBED_ONNX_DIR", "site/models/Xenova/multilingual-e5-base")
    data = export_client_config()
    assert data["embedder"] == {
        "model_id": "Xenova/multilingual-e5-base", "dtype": "q8", "prefix": "query: ",
        "model_name": "intfloat/multilingual-e5-base",
    }
    assert web_model_id(Path("site/models/qorgan/multilingual-e5-base-static-int8")) == "qorgan/multilingual-e5-base-static-int8"
    assert web_model_id(Path("/elsewhere/vendor/model")) == "vendor/model"


def test_export_names_the_on_device_asr_models():
    data = export_client_config()
    cfg = get_config()
    assert data["asr"]["models"]["kk"] == {"id": cfg.vosk_model_kk, "url": f"vosk/{cfg.vosk_model_kk}.tar.gz"}
    assert data["asr"]["models"]["ru"]["id"] == cfg.vosk_model_ru
    assert data["asr"]["sample_rate"] == cfg.asr_sample_rate
