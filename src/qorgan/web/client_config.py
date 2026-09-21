"""Export the non-weight half of the on-device client: taxonomy (ids, hard-signal flags,
localized names), explanation templates, advice strings, meter/window/scoring constants.

Generated from the exact Python sources the server uses, so the JS core in `site/core/`
cannot drift from them. Regenerate with `python -m qorgan.web.client_config` after any
taxonomy / template / advice / config change (the JS parity tests catch a stale copy).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from qorgan.classifier import predict
from qorgan.config import get_config
from qorgan.data.schema import UTTERANCE_JOIN
from qorgan.explain import explainer, recommend, templates as templates_mod
from qorgan.live import meter, session
from qorgan.taxonomy import get_taxonomy

CLIENT_CONFIG_FILENAME = "qorgan-config.json"
SCENARIOS_FILENAME = "scenarios.json"
FORMAT_VERSION = 1
_TEMPLATES_DIR = Path(explainer.__file__).resolve().parent


def export_client_config() -> dict[str, Any]:
    cfg = get_config()
    taxonomy = get_taxonomy()
    locales = list(cfg.supported_locales)
    return {
        "format_version": FORMAT_VERSION,
        "locales": locales,
        "default_locale": cfg.default_locale,
        "taxonomy": {
            "tactics": [
                {
                    "id": tactic.id,
                    "hard_signal": bool(tactic.hard_signal),
                    "names": {locale: taxonomy.display_name(tactic.id, locale) for locale in locales},
                }
                for tactic in taxonomy.tactics
            ],
            "hard_signal_ids": list(taxonomy.hard_signal_ids()),
        },
        "templates": {locale: _templates(locale) for locale in locales},
        "advice": {locale: _advice(locale) for locale in locales},
        "scoring": {
            "risk_threshold": cfg.risk_threshold,
            "attribution_top_k": predict._LINEAR_ATTRIBUTION_TOP_K,
            "attribution_min_score": predict._LINEAR_ATTRIBUTION_MIN_SCORE,
            "calibrated_backends": sorted(explainer._CALIBRATED_BACKENDS),
            "confidence_bands": {"high": templates_mod._HIGH_CONFIDENCE, "medium": templates_mod._MEDIUM_CONFIDENCE},
            "low_confidence_floor": recommend.LOW_CONFIDENCE_FLOOR,
        },
        "meter": {
            "alpha_up": cfg.meter_alpha_up,
            "alpha_down": cfg.meter_alpha_down,
            "min_turns_to_arm": cfg.meter_min_turns_to_arm,
            "short_window_turns": cfg.meter_short_window_turns,
            "enter": cfg.risk_threshold_enter,
            "exit": cfg.risk_threshold_exit,
            "bands": {
                "low_max": meter.BAND_LOW_MAX,
                "medium_max": meter.BAND_MEDIUM_MAX,
                "high_max": meter.BAND_HIGH_MAX,
                "score_max": meter.SCORE_MAX,
            },
            "hard_signal_confidence_floor": meter.HARD_SIGNAL_CONFIDENCE_FLOOR,
            "single_hard_signal_floor": meter.SINGLE_HARD_SIGNAL_FLOOR,
            "double_hard_signal_floor": meter.DOUBLE_HARD_SIGNAL_FLOOR,
        },
        "window": {
            "max_chars": session.MAX_WINDOW_CHARS,
            "head_utterances": session.WINDOW_HEAD_UTTERANCES,
            "join": UTTERANCE_JOIN,
        },
        # The browser loads exactly the graph the server embeds with (ADR D17): the model id
        # is the ONNX dir relative to site/models/ (transformers.js `localModelPath`).
        "embedder": {
            "model_id": web_model_id(cfg.embed_onnx_dir),
            "dtype": "q8",
            "prefix": "query: ",
            "model_name": cfg.embed_model_name,
        },
        # On-device speech recognition (PLAN B9): the same small Vosk models the July
        # server path used, self-hosted as USTAR tarballs under site/models/vosk/.
        "asr": {
            "models": {
                "kk": {"id": cfg.vosk_model_kk, "url": f"{WEB_ASR_MODELS_SUBDIR}/{cfg.vosk_model_kk}.tar.gz"},
                "ru": {"id": cfg.vosk_model_ru, "url": f"{WEB_ASR_MODELS_SUBDIR}/{cfg.vosk_model_ru}.tar.gz"},
            },
            "sample_rate": cfg.asr_sample_rate,
        },
    }


# Relative to site/models/ (the page resolves it against `models/`).
WEB_ASR_MODELS_SUBDIR = "vosk"


def web_model_id(onnx_dir: Path) -> str:
    """`site/models/<id>` -> `<id>`; a dir outside site/models keeps its last two parts."""
    parts = onnx_dir.resolve().parts
    if "models" in parts and "site" in parts:
        idx = len(parts) - 1 - parts[::-1].index("models")
        return "/".join(parts[idx + 1 :])
    return "/".join(parts[-2:])


def _templates(locale: str) -> dict[str, Any]:
    t = templates_mod.load_templates(locale, _TEMPLATES_DIR)
    return {
        "reason_template": t.reason_template,
        "no_signal_reason": t.no_signal_reason,
        "spans_placeholder": t.spans_placeholder,
        "caveat": t.caveat,
        "uncalibrated_suffix": t.uncalibrated_suffix,
        "human_note": t.human_note,
        "confidence": dict(t.confidence),
    }


def _advice(locale: str) -> dict[str, Any]:
    a = recommend.load_advice(locale)
    return {
        "tactic_advice": dict(a.tactic_advice),
        "verification_questions": list(a.verification_questions),
        "low_confidence_note": a.low_confidence_note,
    }


def write_client_config(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(export_client_config(), sort_keys=True, separators=(",", ":"), ensure_ascii=False),
        encoding="utf-8",
    )
    return path


def export_scenarios() -> list[dict[str, Any]]:
    """The bundled replay scripts (same source as `GET /api/live/scenarios`)."""
    from qorgan.api_live import _SCENARIO_LABELS
    from qorgan.data.demo_transcripts import LIVE_DEMO_CALLS

    return [
        {"id": sid, "label": _SCENARIO_LABELS.get(sid, sid), "lines": [ln.strip() for ln in script.splitlines() if ln.strip()]}
        for sid, script in LIVE_DEMO_CALLS.items()
    ]


def write_scenarios(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"scenarios": export_scenarios()}, ensure_ascii=False, indent=1), encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> None:  # pragma: no cover - CLI
    import argparse

    parser = argparse.ArgumentParser(description="Export the on-device client config + scenarios JSON.")
    parser.add_argument("--out-dir", type=Path, default=Path("site/core"))
    args = parser.parse_args(argv)
    print(f"exported -> {write_client_config(args.out_dir / CLIENT_CONFIG_FILENAME)}")
    print(f"exported -> {write_scenarios(args.out_dir / SCENARIOS_FILENAME)}")


if __name__ == "__main__":  # pragma: no cover
    main()
