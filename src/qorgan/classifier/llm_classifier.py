"""Gemini structured-output scam classifier (baseline + fallback; ships first, D1-5).

Calls Gemini in JSON mode with a response schema that forces `{risk, tactic_tags,
trigger_phrases, confidence}` structured output, validates it into a `ScoreResult`, and
caches the result on disk keyed by a hash of (model, transcript) so demos/evals never
re-pay for identical calls (gap G9).

The Gemini client is injected by callers (see `client` param) -- this module never
constructs a network client at import time, so importing it never requires
`GEMINI_API_KEY` to be set.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from qorgan.config import get_config
from qorgan.data.schema import ScoreResult, TacticTag, spans_from_phrases
from qorgan.llm_tools import LLMResponseError, generate_json, thinking_budget_for

# Headroom so Gemini 2.5 thinking tokens (which count against this budget) never truncate
# the JSON verdict -- this is the shipping/demo backend, so truncation here breaks the demo.
_MAX_TOKENS = 2048
_CACHE_VERSION = "v1"

_SYSTEM_PROMPT = (
    "You are a fraud-detection assistant for Qorgan, a Kazakhstani scam-call screening "
    "tool. You will be given a phone-call transcript (Kazakh, Russian, or code-switched). "
    "Assess how likely it is that the CALLER is running a social-engineering scam against "
    "the callee. Respond with a single JSON object with keys: risk (number 0-1), "
    "tactic_tags (array of {id, weight}), trigger_phrases (array of strings), confidence "
    "(number 0-1). Only list a tactic_tags id and only quote a trigger_phrases entry if "
    "the exact phrase appears verbatim in the transcript -- never paraphrase or invent "
    "phrases. If the call looks legitimate (e.g. a real bank confirmation, a family "
    "request, a genuine government/service call), return a low risk and empty "
    "tactic_tags/trigger_phrases."
)

_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "risk": {
            "type": "number",
            "description": "Scam probability in [0, 1]. 0 = certainly legitimate, 1 = certainly a scam.",
        },
        "tactic_tags": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string", "description": "A tactic id from the Qorgan taxonomy."},
                    "weight": {"type": "number", "description": "Contribution weight in [0, 1]."},
                },
                "required": ["id"],
            },
        },
        "trigger_phrases": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Verbatim phrases copied from the transcript that triggered the tags.",
        },
        "confidence": {
            "type": "number",
            "description": "Self-reported confidence in [0, 1] (uncalibrated).",
        },
    },
    "required": ["risk", "tactic_tags", "trigger_phrases", "confidence"],
}


class LLMClassifierError(RuntimeError):
    """Raised when the LLM response cannot be turned into a valid `ScoreResult`."""


def classify(
    transcript: str,
    *,
    client: Any | None = None,
    use_cache: bool = True,
    cache_dir: Path | None = None,
    model: str | None = None,
) -> ScoreResult:
    """Classify `transcript` via Gemini JSON mode; returns a validated `ScoreResult`.

    Args:
        transcript: The call transcript to score. Must be non-empty.
        client: A `google-genai`-style client exposing `.models.generate_content(...)`.
            Injected for testability; defaults to a real `genai.Client()` built from config
            (only constructed lazily, on a cache miss).
        use_cache: Read/write the on-disk transcript-hash prediction cache (default True).
        cache_dir: Override the cache directory (defaults to `config.cache_dir`).
        model: Override the model id (defaults to `config.llm_model_quality`).
    """
    if not transcript or not transcript.strip():
        raise ValueError("transcript must not be empty")

    cfg = get_config()
    active_model = model or cfg.llm_model_quality
    active_cache_dir = cache_dir if cache_dir is not None else cfg.cache_dir

    if use_cache:
        cached = _read_cache(active_cache_dir, active_model, transcript)
        if cached is not None:
            return cached

    active_client = client if client is not None else _default_client(cfg)
    try:
        payload = generate_json(
            active_client,
            model=active_model,
            prompt=transcript,
            response_schema=_RESPONSE_SCHEMA,
            system_instruction=_SYSTEM_PROMPT,
            max_output_tokens=_MAX_TOKENS,
            thinking_budget=thinking_budget_for(active_model),
        )
    except LLMResponseError as exc:
        raise LLMClassifierError(str(exc)) from exc

    result = _build_score_result(payload, transcript)

    if use_cache:
        _write_cache(active_cache_dir, active_model, transcript, result)
    return result


def _default_client(cfg) -> Any:  # pragma: no cover - real network client, not exercised in tests
    from google import genai

    return genai.Client(api_key=cfg.gemini_api_key)


def _build_score_result(payload: dict[str, Any], transcript: str) -> ScoreResult:
    try:
        risk = float(payload["risk"])
    except (KeyError, TypeError, ValueError) as exc:
        raise LLMClassifierError(f"LLM response missing/invalid 'risk': {payload!r}") from exc
    risk = max(0.0, min(1.0, risk))

    raw_confidence = payload.get("confidence")
    if raw_confidence is not None:
        try:
            raw_confidence = max(0.0, min(1.0, float(raw_confidence)))
        except (TypeError, ValueError):
            raw_confidence = None

    tags = _build_tags(payload.get("tactic_tags") or [])
    spans = spans_from_phrases(payload.get("trigger_phrases") or [], transcript)

    return ScoreResult(
        risk=risk,
        tags=tags,
        attributions=spans,
        backend="llm",
        raw_confidence=raw_confidence,
    )


def _build_tags(raw_tags: list[Any]) -> tuple[TacticTag, ...]:
    tags: list[TacticTag] = []
    for item in raw_tags:
        if not isinstance(item, dict):
            continue
        tag_id = item.get("id")
        if not tag_id or not str(tag_id).strip():
            continue
        weight = item.get("weight", 1.0)
        try:
            weight = max(0.0, min(1.0, float(weight)))
        except (TypeError, ValueError):
            weight = 1.0
        tags.append(TacticTag(id=str(tag_id), weight=weight))
    return tuple(tags)


def _cache_key(model: str, transcript: str) -> str:
    digest = hashlib.sha256(f"{_CACHE_VERSION}::{model}::{transcript}".encode("utf-8")).hexdigest()
    return digest


def _cache_path(cache_dir: Path, model: str, transcript: str) -> Path:
    return cache_dir / f"{_cache_key(model, transcript)}.json"


def _read_cache(cache_dir: Path, model: str, transcript: str) -> ScoreResult | None:
    path = _cache_path(cache_dir, model, transcript)
    if not path.exists():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return ScoreResult.model_validate(raw)
    except (json.JSONDecodeError, ValueError):
        # Corrupt/incompatible cache entry -- treat as a miss rather than crash the demo.
        return None


def _write_cache(cache_dir: Path, model: str, transcript: str, result: ScoreResult) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = _cache_path(cache_dir, model, transcript)
    path.write_text(json.dumps(result.model_dump(mode="json"), ensure_ascii=False, indent=2), encoding="utf-8")
