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
    assert [t["id"] for t in data["taxonomy"]["tactics"]] == list(taxonomy.all_tactic_ids())  # v1 + v2 (D63)
    assert data["taxonomy"]["hard_signal_ids"] == list(taxonomy.hard_signal_ids())
    assert data["taxonomy"]["active_label_space"] == taxonomy.active_label_space
    assert len(data["roles"]) == 19 and all(set(r["names"]) == set(data["locales"]) for r in data["roles"])
    assert all(data["templates"][loc]["role_template"] for loc in data["locales"])
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


def test_served_weights_are_current_for_the_repo_lexicons():
    """The committed `site/models/weights.json` is the shipped model (the browser, and the
    server under `QORGAN_LINEAR_WEIGHTS=web|auto`). It must load, and the lexicons embedded in
    it must be the repo's: editing `data/lexicon/*.yaml` without retraining and re-exporting
    (scripts/export_parity_fixtures.py) fails here, on every checkout -- not only on a machine
    that happens to hold a trained bundle."""
    from pathlib import Path

    from qorgan.classifier.cue_lexicon import lexicon_hash, load_cue_lexicon
    from qorgan.classifier.reassurance import load_reassurance_patterns, reassurance_hash
    from qorgan.classifier.web_bundle import load_linear_from_web

    served = load_linear_from_web(Path(__file__).resolve().parents[2] / "site" / "models" / "weights.json")
    assert served.cue_lexicon_hash == lexicon_hash(load_cue_lexicon())
    assert served.reassurance_hash == reassurance_hash(load_reassurance_patterns())


def test_served_weights_match_a_current_trained_bundle_when_present():
    """When `models/linear` holds a bundle that is valid for the current lexicons, the
    browser weights must be its export (a retrain not yet published fails here). A bundle
    that is stale for the current lexicons (e.g. an older Hub download) is not the shipped
    model, so it is not compared."""
    import hashlib
    from pathlib import Path

    import pytest

    from qorgan.classifier.linear_train import LinearFeatureMismatchError, load_linear

    root = Path(__file__).resolve().parents[2]
    trained = root / "models" / "linear" / "web" / "weights.json"
    served = root / "site" / "models" / "weights.json"
    if not trained.exists():
        pytest.skip("no local trained bundle")
    try:
        load_linear(root / "models" / "linear")
    except LinearFeatureMismatchError:
        pytest.skip("the local bundle is stale for the current lexicons; the committed weights are served")
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


def test_english_is_a_third_handicapped_recogniser_and_can_be_switched_off(monkeypatch):
    # ADR D62: English joins the voice mode last in the vote order, ranked with a handicap so its
    # confident garbage on kk/ru speech does not win; QORGAN_VOSK_MODEL_EN="" removes it.
    cfg = get_config()
    data = export_client_config()
    # The JSON writer sorts keys (models arrive as en, kk, ru); the vote order travels as a list.
    assert data["asr"]["order"] == ["kk", "ru", "en"]
    assert json.loads(json.dumps(data, sort_keys=True))["asr"]["order"] == ["kk", "ru", "en"]
    assert data["asr"]["models"]["en"] == {"id": cfg.vosk_model_en, "url": f"vosk/{cfg.vosk_model_en}.tar.gz"}
    assert data["asr"]["handicap"] == {"en": cfg.asr_en_handicap}
    assert 0 < cfg.asr_en_handicap < 1

    monkeypatch.setenv("QORGAN_VOSK_MODEL_EN", "")
    off = export_client_config()
    assert off["asr"]["order"] == ["kk", "ru"] and set(off["asr"]["models"]) == {"kk", "ru"}
    assert off["asr"]["handicap"] == {}
