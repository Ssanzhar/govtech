"""TDD tests for `qorgan.config` — centralized constants, env loading, validation."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from qorgan.config import Config, ConfigError, get_config, load_config


def test_defaults_with_empty_env():
    cfg = load_config({})
    assert cfg.classifier_backend == "linear"
    assert cfg.gemini_api_key is None
    assert cfg.risk_threshold == 0.59
    assert cfg.risk_threshold_enter == 0.59
    assert cfg.risk_threshold_exit == 0.49
    assert cfg.default_seed == 42
    assert cfg.supported_locales == ("ru", "kk")
    assert cfg.default_locale == "ru"
    assert cfg.llm_model_quality == "gemini-2.5-pro"
    assert cfg.llm_model_bulk == "gemini-2.5-flash"
    assert cfg.whisper_model_size == "small"


def test_paths_are_path_objects_and_derived_from_data_dir():
    cfg = load_config({})
    assert isinstance(cfg.data_dir, Path)
    assert cfg.data_dir.name == "data"
    assert cfg.taxonomy_path == cfg.data_dir / "taxonomy" / "tactics.yaml"
    assert cfg.cache_dir == cfg.data_dir / "cache" / "llm_predictions"
    assert cfg.model_dir.name == "models"
    assert cfg.corpus_config_path == cfg.repo_root / "configs" / "corpus.yaml"


def test_env_overrides_applied():
    cfg = load_config({"QORGAN_CLASSIFIER_BACKEND": "mock", "QORGAN_RISK_THRESHOLD": "0.85"})
    assert cfg.classifier_backend == "mock"
    assert cfg.risk_threshold == 0.85


def test_custom_data_dir_cascades_to_taxonomy_and_cache_paths():
    cfg = load_config({"QORGAN_DATA_DIR": "/tmp/custom_data"})
    assert cfg.taxonomy_path == Path("/tmp/custom_data/taxonomy/tactics.yaml")
    assert cfg.cache_dir == Path("/tmp/custom_data/cache/llm_predictions")


def test_explicit_taxonomy_path_wins_over_data_dir_default():
    cfg = load_config({"QORGAN_DATA_DIR": "/tmp/x", "QORGAN_TAXONOMY_PATH": "/tmp/custom.yaml"})
    assert cfg.taxonomy_path == Path("/tmp/custom.yaml")


def test_invalid_backend_raises_config_error():
    with pytest.raises(ConfigError):
        load_config({"QORGAN_CLASSIFIER_BACKEND": "bogus"})


def test_risk_threshold_out_of_range_raises():
    with pytest.raises(ConfigError):
        load_config({"QORGAN_RISK_THRESHOLD": "1.5"})


def test_risk_threshold_not_a_number_raises():
    with pytest.raises(ConfigError):
        load_config({"QORGAN_RISK_THRESHOLD": "not-a-number"})


def test_hysteresis_exit_above_enter_raises():
    with pytest.raises(ConfigError):
        load_config(
            {"QORGAN_RISK_THRESHOLD_ENTER": "0.5", "QORGAN_RISK_THRESHOLD_EXIT": "0.6"}
        )


def test_hysteresis_exit_equal_enter_is_allowed():
    cfg = load_config({"QORGAN_RISK_THRESHOLD_ENTER": "0.5", "QORGAN_RISK_THRESHOLD_EXIT": "0.5"})
    assert cfg.risk_threshold_exit == cfg.risk_threshold_enter == 0.5


def test_default_locale_must_be_in_supported_locales():
    with pytest.raises(ConfigError):
        load_config({"QORGAN_SUPPORTED_LOCALES": "ru", "QORGAN_DEFAULT_LOCALE": "kk"})


def test_supported_locales_parsed_from_csv():
    cfg = load_config({"QORGAN_SUPPORTED_LOCALES": "ru, kk, en", "QORGAN_DEFAULT_LOCALE": "en"})
    assert cfg.supported_locales == ("ru", "kk", "en")


def test_supported_locales_empty_raises():
    with pytest.raises(ConfigError):
        load_config({"QORGAN_SUPPORTED_LOCALES": "  "})


def test_gemini_api_key_empty_string_normalizes_to_none():
    cfg = load_config({"GEMINI_API_KEY": ""})
    assert cfg.gemini_api_key is None


def test_gemini_api_key_read_when_present():
    cfg = load_config({"GEMINI_API_KEY": "test-key-123"})
    assert cfg.gemini_api_key == "test-key-123"


def test_google_api_key_used_as_gemini_fallback():
    cfg = load_config({"GOOGLE_API_KEY": "test-key-456"})
    assert cfg.gemini_api_key == "test-key-456"


def test_xlmr_model_dir_default_and_override():
    cfg = load_config({})
    assert cfg.xlmr_model_dir == cfg.model_dir / "xlmr"
    override = load_config({"QORGAN_XLMR_MODEL_DIR": "/tmp/my_xlmr"})
    assert override.xlmr_model_dir == Path("/tmp/my_xlmr")


def test_linear_model_dir_and_embed_model_defaults_and_overrides():
    cfg = load_config({})
    assert cfg.linear_model_dir == cfg.model_dir / "linear"
    assert cfg.embed_model_name == "intfloat/multilingual-e5-base"
    override = load_config(
        {"QORGAN_LINEAR_MODEL_DIR": "/tmp/lin", "QORGAN_EMBED_MODEL_NAME": "intfloat/multilingual-e5-small"}
    )
    assert override.linear_model_dir == Path("/tmp/lin")
    assert override.embed_model_name == "intfloat/multilingual-e5-small"


def test_split_fraction_defaults():
    cfg = load_config({})
    assert cfg.split_train_fraction == 0.7
    assert cfg.split_val_fraction == 0.15
    # test fraction is the remainder
    assert cfg.split_test_fraction == pytest.approx(0.15)


def test_split_fractions_env_overrides():
    cfg = load_config({"QORGAN_SPLIT_TRAIN_FRACTION": "0.8", "QORGAN_SPLIT_VAL_FRACTION": "0.1"})
    assert cfg.split_train_fraction == 0.8
    assert cfg.split_val_fraction == 0.1
    assert cfg.split_test_fraction == pytest.approx(0.1)


def test_split_fractions_summing_to_one_or_more_raises():
    with pytest.raises(ConfigError):
        load_config({"QORGAN_SPLIT_TRAIN_FRACTION": "0.7", "QORGAN_SPLIT_VAL_FRACTION": "0.3"})


def test_split_train_fraction_out_of_range_raises():
    with pytest.raises(ConfigError):
        load_config({"QORGAN_SPLIT_TRAIN_FRACTION": "0"})
    with pytest.raises(ConfigError):
        load_config({"QORGAN_SPLIT_TRAIN_FRACTION": "1.0"})


def test_split_val_fraction_out_of_range_raises():
    with pytest.raises(ConfigError):
        load_config({"QORGAN_SPLIT_VAL_FRACTION": "-0.1"})


def test_config_is_immutable():
    cfg = load_config({})
    with pytest.raises(ValidationError):
        cfg.risk_threshold = 0.1


def test_get_config_reflects_current_environment(monkeypatch):
    monkeypatch.setenv("QORGAN_CLASSIFIER_BACKEND", "mock")
    assert get_config().classifier_backend == "mock"
    monkeypatch.setenv("QORGAN_CLASSIFIER_BACKEND", "llm")
    assert get_config().classifier_backend == "llm"


def test_repo_root_is_absolute_and_contains_pyproject():
    cfg = load_config({})
    assert cfg.repo_root.is_absolute()
    assert (cfg.repo_root / "pyproject.toml").exists()


def test_load_config_returns_config_instance():
    assert isinstance(load_config({}), Config)


# --- privacy knobs (PLAN_2026-09 C2/C3) ------------------------------------------------------


def test_number_hmac_key_defaults_to_none_and_reads_env():
    assert load_config({}).number_hmac_key is None
    assert load_config({"QORGAN_NUMBER_HMAC_KEY": ""}).number_hmac_key is None
    assert load_config({"QORGAN_NUMBER_HMAC_KEY": "s3cret"}).number_hmac_key == b"s3cret"


def test_report_retention_days_default_and_override():
    assert load_config({}).report_retention_days == 180
    assert load_config({"QORGAN_REPORT_RETENTION_DAYS": "30"}).report_retention_days == 30


def test_report_retention_days_must_be_positive():
    with pytest.raises(ConfigError):
        load_config({"QORGAN_REPORT_RETENTION_DAYS": "0"})


# --- partner intake API (PLAN_2026-09 C5) ---------------------------------------------------


def test_partner_registry_defaults_to_closed():
    cfg = load_config({})
    assert cfg.partner_credentials == ()
    assert cfg.partner_quota_window_hours == 24


def test_partner_registry_is_parsed_with_default_and_explicit_quotas():
    cfg = load_config({
        "QORGAN_PARTNER_API_KEYS": "bank_a:0123456789abcdefghij:5,telecom_b:abcdefghij0123456789",
        "QORGAN_PARTNER_DAILY_QUOTA": "42",
    })
    assert [(c.id, c.daily_quota) for c in cfg.partner_credentials] == [("bank_a", 5), ("telecom_b", 42)]
    assert "0123456789abcdefghij" not in repr(cfg)


def test_weak_partner_key_fails_fast():
    with pytest.raises(ConfigError):
        load_config({"QORGAN_PARTNER_API_KEYS": "bank_a:short"})


def test_asr_style_train_fraction_defaults_and_env_override():
    assert load_config({}).asr_style_train_fraction == 1.0
    assert load_config({"QORGAN_ASR_STYLE_TRAIN_FRACTION": "0.25"}).asr_style_train_fraction == 0.25
    with pytest.raises(ValueError):
        load_config({"QORGAN_ASR_STYLE_TRAIN_FRACTION": "1.5"})
