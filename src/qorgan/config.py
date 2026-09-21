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

from qorgan.partners import DEFAULT_PARTNER_DAILY_QUOTA, PartnerCredential, parse_partner_credentials

# Anchor for all repo-relative defaults: src/qorgan/config.py -> src/qorgan -> src -> repo root.
_REPO_ROOT = Path(__file__).resolve().parents[2]

_DEFAULT_LLM_MODEL_QUALITY = "gemini-2.5-pro"
_DEFAULT_LLM_MODEL_BULK = "gemini-2.5-flash"
_DEFAULT_CLASSIFIER_BACKEND = "linear"
_DEFAULT_WHISPER_MODEL_SIZE = "small"
# Live streaming ASR (design spec §06): small Vosk models for KK + RU run in parallel and
# vote per utterance. Names resolve via vosk's model auto-download (~/.cache/vosk).
_DEFAULT_VOSK_MODEL_KK = "vosk-model-small-kz-0.42"
_DEFAULT_VOSK_MODEL_RU = "vosk-model-small-ru-0.22"
_DEFAULT_ASR_SAMPLE_RATE = 16000
_DEFAULT_EMBED_MODEL_NAME = "intfloat/multilingual-e5-base"
# "sentence-transformers" (fp32 PyTorch) or "onnx" (the int8 graph the browser ships;
# PLAN_2026-09 A4 -- server and device then embed identically).
_DEFAULT_EMBED_BACKEND = "onnx"
_EMBED_BACKENDS = ("sentence-transformers", "onnx")
_DEFAULT_EMBED_ONNX_SUBDIR = Path("site") / "models" / "Xenova" / "multilingual-e5-base"
# Shipped default (2026-09-14, PLAN_2026-09 A4/A5): heads trained on the int8 ONNX
# embeddings the browser ships (server + device embed identically). At 0.59 every FPR
# gate is 0 (test / authored_heldout / ood) with test recall 0.953 and authored recall
# 1.000; the one ood negative that crossed 0.55 sat at 0.551. ood recall is 0.844 (was
# 0.889 with fp32-trained heads) -- reported, not hidden. See docs/eval_report.md.
_DEFAULT_RISK_THRESHOLD = 0.59
# Consented reports are kept this long before the purge removes them (PLAN_2026-09 C3).
_DEFAULT_REPORT_RETENTION_DAYS = 180
# Partner intake API (PLAN_2026-09 C5): rolling window for the per-partner report budget.
_DEFAULT_PARTNER_QUOTA_WINDOW_HOURS = 24
_DEFAULT_RISK_THRESHOLD_ENTER = 0.59
_DEFAULT_RISK_THRESHOLD_EXIT = 0.49
# Live suspicion-meter smoothing (design spec §08): the displayed 0-100 score follows an
# asymmetric EMA -- it rises fast (two consistent turns reach the target band) and decays
# slowly (a scammer changing topic doesn't reset accumulated evidence). Launch defaults,
# to be re-tuned on pilot recordings with the FPR-first harness.
_DEFAULT_METER_ALPHA_UP = 0.5
_DEFAULT_METER_ALPHA_DOWN = 0.12
# PLAN_2026-09 A6 (2026-09-19): the first one or two windows of a call are short and noisy --
# a bank's opener reads like a scam opener until context arrives -- so the warning latch
# cannot engage before this many committed utterances unless a confident hard signal fired
# (measured: removes the transient false latches at turn 2 without losing an alert).
_DEFAULT_METER_MIN_TURNS_TO_ARM = 3
# Optional damping of the rise on short windows: alpha_up is scaled by min(1, turn / N);
# 1 = off (it trades one hairline scam for three fewer transient latches on test).
_DEFAULT_METER_SHORT_WINDOW_TURNS = 1
_DEFAULT_SEED = 42
_DEFAULT_SUPPORTED_LOCALES: tuple[str, ...] = ("ru", "kk")
_DEFAULT_LOCALE = "ru"
# Corpus split fractions (test fraction is the remainder). Consumed by
# `qorgan.data.build_corpus` for the deterministic train/val/test partition.
_DEFAULT_SPLIT_TRAIN_FRACTION = 0.7
# Share of train rows that also get an ASR-styled copy at corpus build (PLAN A10, ADR D31).
_DEFAULT_ASR_STYLE_TRAIN_FRACTION = 1.0
_DEFAULT_SPLIT_VAL_FRACTION = 0.15

ClassifierBackend = Literal["linear", "llm", "xlmr", "mock"]


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
    cue_lexicon_path: Path
    reassurance_patterns_path: Path
    # Ids of authored_heldout anchors read during feature engineering (PLAN_2026-09 A2).
    inspection_ledger_path: Path

    # --- ASR ---
    whisper_model_size: str
    vosk_model_kk: str
    vosk_model_ru: str
    asr_sample_rate: int = Field(gt=0)

    # --- Fine-tuned XLM-R backend (D3) ---
    xlmr_model_dir: Path

    # --- Embeddings + linear classifier backend ("linear") ---
    linear_model_dir: Path
    embed_model_name: str
    embed_backend: str
    embed_onnx_dir: Path

    # --- Risk thresholds / hysteresis (gap G8) ---
    risk_threshold: float = Field(ge=0.0, le=1.0)
    risk_threshold_enter: float = Field(ge=0.0, le=1.0)
    risk_threshold_exit: float = Field(ge=0.0, le=1.0)

    # --- Live suspicion-meter smoothing (design spec §08) ---
    meter_alpha_up: float = Field(gt=0.0, le=1.0)
    meter_alpha_down: float = Field(gt=0.0, le=1.0)
    meter_min_turns_to_arm: int = Field(ge=1)
    meter_short_window_turns: int = Field(ge=1)

    # --- Corpus splits (test fraction is the remainder) ---
    split_train_fraction: float = Field(gt=0.0, lt=1.0)
    asr_style_train_fraction: float = Field(ge=0.0, le=1.0)
    split_val_fraction: float = Field(gt=0.0, lt=1.0)

    # --- Privacy (PLAN_2026-09 C2/C3, ADR D14) ---
    # Salt for phone-number HMACs; None means numbers cannot be accepted at all.
    number_hmac_key: bytes | None
    report_retention_days: int = Field(gt=0)

    # --- Partner intake API (PLAN_2026-09 C5, ADR D19) ---
    # Parsed `QORGAN_PARTNER_API_KEYS`; empty means the /api/v1 ingress is closed.
    partner_credentials: tuple[PartnerCredential, ...]
    partner_quota_window_hours: int = Field(gt=0)

    # --- Reproducibility / localization ---
    default_seed: int
    supported_locales: tuple[str, ...]
    default_locale: str

    @property
    def split_test_fraction(self) -> float:
        """The held-out test fraction: whatever is left after train + val."""
        return 1.0 - self.split_train_fraction - self.split_val_fraction

    @model_validator(mode="after")
    def _embed_backend_known(self) -> "Config":
        if self.embed_backend not in _EMBED_BACKENDS:
            raise ValueError(f"embed_backend must be one of {_EMBED_BACKENDS}, got {self.embed_backend!r}")
        return self

    @model_validator(mode="after")
    def _hysteresis_exit_not_above_enter(self) -> "Config":
        if self.risk_threshold_exit > self.risk_threshold_enter:
            raise ValueError(
                "risk_threshold_exit must be <= risk_threshold_enter "
                f"(got exit={self.risk_threshold_exit}, enter={self.risk_threshold_enter})"
            )
        return self

    @model_validator(mode="after")
    def _meter_decay_not_above_rise(self) -> "Config":
        if self.meter_alpha_down > self.meter_alpha_up:
            raise ValueError(
                "meter_alpha_down must be <= meter_alpha_up "
                f"(got down={self.meter_alpha_down}, up={self.meter_alpha_up})"
            )
        return self

    @model_validator(mode="after")
    def _supported_locales_not_empty(self) -> "Config":
        if not self.supported_locales:
            raise ValueError("supported_locales must not be empty")
        return self

    @model_validator(mode="after")
    def _split_fractions_leave_room_for_test(self) -> "Config":
        if self.split_train_fraction + self.split_val_fraction >= 1.0:
            raise ValueError(
                "split_train_fraction + split_val_fraction must be < 1.0 to leave a "
                f"non-empty test split (got train={self.split_train_fraction}, "
                f"val={self.split_val_fraction})"
            )
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


def _read_secret_bytes(env: Mapping[str, str], key: str) -> bytes | None:
    """An optional secret as bytes; unset or blank means "not configured" (None)."""
    value = env.get(key, "").strip()
    return value.encode("utf-8") if value else None


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
            cue_lexicon_path=_read_path(
                source, "QORGAN_CUE_LEXICON_PATH", data_dir / "lexicon" / "hard_signal_cues.yaml"
            ),
            reassurance_patterns_path=_read_path(
                source, "QORGAN_REASSURANCE_PATTERNS_PATH", data_dir / "lexicon" / "reassurance_patterns.yaml"
            ),
            inspection_ledger_path=_read_path(
                source, "QORGAN_INSPECTION_LEDGER_PATH", data_dir / "anchors" / "inspection_ledger.yaml"
            ),
            whisper_model_size=_read_str(
                source, "QORGAN_WHISPER_MODEL_SIZE", _DEFAULT_WHISPER_MODEL_SIZE
            ),
            vosk_model_kk=_read_str(source, "QORGAN_VOSK_MODEL_KK", _DEFAULT_VOSK_MODEL_KK),
            vosk_model_ru=_read_str(source, "QORGAN_VOSK_MODEL_RU", _DEFAULT_VOSK_MODEL_RU),
            asr_sample_rate=_read_int(source, "QORGAN_ASR_SAMPLE_RATE", _DEFAULT_ASR_SAMPLE_RATE),
            xlmr_model_dir=_read_path(
                source, "QORGAN_XLMR_MODEL_DIR", _read_path(source, "QORGAN_MODEL_DIR", _REPO_ROOT / "models") / "xlmr"
            ),
            linear_model_dir=_read_path(
                source, "QORGAN_LINEAR_MODEL_DIR", _read_path(source, "QORGAN_MODEL_DIR", _REPO_ROOT / "models") / "linear"
            ),
            embed_model_name=_read_str(source, "QORGAN_EMBED_MODEL_NAME", _DEFAULT_EMBED_MODEL_NAME),
            embed_backend=_read_str(source, "QORGAN_EMBED_BACKEND", _DEFAULT_EMBED_BACKEND),
            embed_onnx_dir=_read_path(source, "QORGAN_EMBED_ONNX_DIR", _REPO_ROOT / _DEFAULT_EMBED_ONNX_SUBDIR),
            risk_threshold=_read_float(source, "QORGAN_RISK_THRESHOLD", _DEFAULT_RISK_THRESHOLD),
            risk_threshold_enter=_read_float(
                source, "QORGAN_RISK_THRESHOLD_ENTER", _DEFAULT_RISK_THRESHOLD_ENTER
            ),
            risk_threshold_exit=_read_float(
                source, "QORGAN_RISK_THRESHOLD_EXIT", _DEFAULT_RISK_THRESHOLD_EXIT
            ),
            meter_alpha_up=_read_float(source, "QORGAN_METER_ALPHA_UP", _DEFAULT_METER_ALPHA_UP),
            meter_alpha_down=_read_float(
                source, "QORGAN_METER_ALPHA_DOWN", _DEFAULT_METER_ALPHA_DOWN
            ),
            meter_min_turns_to_arm=_read_int(
                source, "QORGAN_METER_MIN_TURNS_TO_ARM", _DEFAULT_METER_MIN_TURNS_TO_ARM
            ),
            meter_short_window_turns=_read_int(
                source, "QORGAN_METER_SHORT_WINDOW_TURNS", _DEFAULT_METER_SHORT_WINDOW_TURNS
            ),
            split_train_fraction=_read_float(
                source, "QORGAN_SPLIT_TRAIN_FRACTION", _DEFAULT_SPLIT_TRAIN_FRACTION
            ),
            asr_style_train_fraction=_read_float(
                source, "QORGAN_ASR_STYLE_TRAIN_FRACTION", _DEFAULT_ASR_STYLE_TRAIN_FRACTION
            ),
            split_val_fraction=_read_float(
                source, "QORGAN_SPLIT_VAL_FRACTION", _DEFAULT_SPLIT_VAL_FRACTION
            ),
            default_seed=_read_int(source, "QORGAN_SEED", _DEFAULT_SEED),
            number_hmac_key=_read_secret_bytes(source, "QORGAN_NUMBER_HMAC_KEY"),
            report_retention_days=_read_int(
                source, "QORGAN_REPORT_RETENTION_DAYS", _DEFAULT_REPORT_RETENTION_DAYS
            ),
            partner_credentials=parse_partner_credentials(
                source.get("QORGAN_PARTNER_API_KEYS", ""),
                default_quota=_read_int(source, "QORGAN_PARTNER_DAILY_QUOTA", DEFAULT_PARTNER_DAILY_QUOTA),
            ),
            partner_quota_window_hours=_read_int(
                source, "QORGAN_PARTNER_QUOTA_WINDOW_HOURS", _DEFAULT_PARTNER_QUOTA_WINDOW_HOURS
            ),
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
