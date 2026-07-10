"""Centralized configuration for Qorgan.

ALL constants (model routing, classifier backend, risk thresholds, hysteresis, paths,
seeds, locales) live here. No other module may hardcode these values -- read them from a
`Config` instance instead.

Values are sourced from environment variables (optionally loaded from a `.env` file at the
repo root), each with a documented default so the project runs out of the box with no
`.env` file and no `GEMINI_API_KEY`.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field, model_validator

# Anchor for all repo-relative defaults: src/qorgan/config.py -> src/qorgan -> src -> repo root.
_REPO_ROOT = Path(__file__).resolve().parents[2]

_DEFAULT_LLM_MODEL_QUALITY = "gemini-2.5-pro"
_DEFAULT_LLM_MODEL_BULK = "gemini-2.5-flash"
_DEFAULT_CLASSIFIER_BACKEND = "llm"
_DEFAULT_WHISPER_MODEL_SIZE = "small"
_DEFAULT_RISK_THRESHOLD = 0.7
_DEFAULT_RISK_THRESHOLD_ENTER = 0.7
_DEFAULT_RISK_THRESHOLD_EXIT = 0.55
_DEFAULT_SEED = 42
_DEFAULT_SUPPORTED_LOCALES: tuple[str, ...] = ("ru", "kk")
_DEFAULT_LOCALE = "ru"

ClassifierBackend = Literal["llm", "xlmr", "mock"]


class ConfigError(ValueError):
    """Raised when configuration values are missing, malformed, or mutually inconsistent."""


class Config(BaseModel):
    """Immutable, validated application configuration.

    Construct via `load_config()` / `get_config()` -- never instantiate directly with
    unvalidated values.
    """

    model_config = ConfigDict(frozen=True)

    repo_root: Path

    # --- LLM / classifier routing ---
    gemini_api_key: str | None
    llm_model_quality: str
    llm_model_bulk: str
    classifier_backend: ClassifierBackend

    # --- Paths ---
    data_dir: Path
    model_dir: Path
    taxonomy_path: Path
    cache_dir: Path
    corpus_config_path: Path

    # --- ASR ---
    whisper_model_size: str

    # --- Risk thresholds / hysteresis (gap G8) ---
    risk_threshold: float = Field(ge=0.0, le=1.0)
    risk_threshold_enter: float = Field(ge=0.0, le=1.0)
    risk_threshold_exit: float = Field(ge=0.0, le=1.0)

    # --- Reproducibility / localization ---
    default_seed: int
    supported_locales: tuple[str, ...]
    default_locale: str

    @model_validator(mode="after")
    def _hysteresis_exit_not_above_enter(self) -> "Config":
        if self.risk_threshold_exit > self.risk_threshold_enter:
            raise ValueError(
                "risk_threshold_exit must be <= risk_threshold_enter "
                f"(got exit={self.risk_threshold_exit}, enter={self.risk_threshold_enter})"
            )
        return self

    @model_validator(mode="after")
    def _supported_locales_not_empty(self) -> "Config":
        if not self.supported_locales:
            raise ValueError("supported_locales must not be empty")
        return self

    @model_validator(mode="after")
    def _default_locale_is_supported(self) -> "Config":
        if self.default_locale not in self.supported_locales:
            raise ValueError(
                f"default_locale {self.default_locale!r} must be one of {self.supported_locales}"
            )
        return self


def _read_str(env: Mapping[str, str], key: str, default: str) -> str:
    raw = env.get(key)
    return default if not raw else raw


def _read_float(env: Mapping[str, str], key: str, default: float) -> float:
    raw = env.get(key)
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ConfigError(f"{key}={raw!r} is not a valid float") from exc


def _read_int(env: Mapping[str, str], key: str, default: int) -> int:
    raw = env.get(key)
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ConfigError(f"{key}={raw!r} is not a valid int") from exc


def _read_path(env: Mapping[str, str], key: str, default: Path) -> Path:
    raw = env.get(key)
    return default if not raw else Path(raw)


def _read_csv_tuple(env: Mapping[str, str], key: str, default: tuple[str, ...]) -> tuple[str, ...]:
    raw = env.get(key)
    if not raw:
        return default
    return tuple(part.strip() for part in raw.split(",") if part.strip())


def load_config(env: Mapping[str, str] | None = None) -> Config:
    """Build a validated `Config` from environment variables.

    When `env` is omitted, `.env` (if present at the repo root) is loaded into
    `os.environ` -- without overriding already-set variables -- and `os.environ` is used
    as the source. Pass an explicit mapping (e.g. in tests) to bypass the process
    environment and `.env` entirely.

    Raises `ConfigError` if any value is malformed or mutually inconsistent.
    """
    if env is None:
        load_dotenv(dotenv_path=_REPO_ROOT / ".env", override=False)
        source: Mapping[str, str] = os.environ
    else:
        source = env

    data_dir = _read_path(source, "QORGAN_DATA_DIR", _REPO_ROOT / "data")

    try:
        return Config(
            repo_root=_REPO_ROOT,
            gemini_api_key=source.get("GEMINI_API_KEY") or source.get("GOOGLE_API_KEY") or None,
            llm_model_quality=_read_str(source, "QORGAN_LLM_MODEL_QUALITY", _DEFAULT_LLM_MODEL_QUALITY),
            llm_model_bulk=_read_str(source, "QORGAN_LLM_MODEL_BULK", _DEFAULT_LLM_MODEL_BULK),
            classifier_backend=_read_str(
                source, "QORGAN_CLASSIFIER_BACKEND", _DEFAULT_CLASSIFIER_BACKEND
            ),
            data_dir=data_dir,
            model_dir=_read_path(source, "QORGAN_MODEL_DIR", _REPO_ROOT / "models"),
            taxonomy_path=_read_path(
                source, "QORGAN_TAXONOMY_PATH", data_dir / "taxonomy" / "tactics.yaml"
            ),
            cache_dir=_read_path(
                source, "QORGAN_CACHE_DIR", data_dir / "cache" / "llm_predictions"
            ),
            corpus_config_path=_read_path(
                source, "QORGAN_CORPUS_CONFIG_PATH", _REPO_ROOT / "configs" / "corpus.yaml"
            ),
            whisper_model_size=_read_str(
                source, "QORGAN_WHISPER_MODEL_SIZE", _DEFAULT_WHISPER_MODEL_SIZE
            ),
            risk_threshold=_read_float(source, "QORGAN_RISK_THRESHOLD", _DEFAULT_RISK_THRESHOLD),
            risk_threshold_enter=_read_float(
                source, "QORGAN_RISK_THRESHOLD_ENTER", _DEFAULT_RISK_THRESHOLD_ENTER
            ),
            risk_threshold_exit=_read_float(
                source, "QORGAN_RISK_THRESHOLD_EXIT", _DEFAULT_RISK_THRESHOLD_EXIT
            ),
            default_seed=_read_int(source, "QORGAN_SEED", _DEFAULT_SEED),
            supported_locales=_read_csv_tuple(
                source, "QORGAN_SUPPORTED_LOCALES", _DEFAULT_SUPPORTED_LOCALES
            ),
            default_locale=_read_str(source, "QORGAN_DEFAULT_LOCALE", _DEFAULT_LOCALE),
        )
    except ConfigError:
        raise
    except ValueError as exc:
        raise ConfigError(str(exc)) from exc


def get_config() -> Config:
    """Return the application configuration, re-read from the current environment.

    Deliberately uncached: reading a handful of env vars (plus an optional `.env` stat) is
    cheap, and always reflecting the live environment keeps tests and callers that mutate
    `os.environ` (e.g. via `monkeypatch`) correct without any cache-invalidation dance.
    """
    return load_config()
