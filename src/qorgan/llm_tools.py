"""Small shared helpers for Gemini structured-JSON generation calls.

Used by both `classifier/llm_classifier.py` (scoring) and `data/generate.py` (synthetic
corpus generation) so the "call Gemini for a JSON payload and parse it" logic lives in
exactly one place.

The `client` is always injected by callers (a `google-genai`-style object exposing
`client.models.generate_content(...)`), so importing these modules never requires a
`GEMINI_API_KEY` and tests can pass a fake client with no network.
"""

from __future__ import annotations

import json
from typing import Any


class LLMResponseError(RuntimeError):
    """Raised when a Gemini response cannot be parsed into the expected JSON object."""


def generate_json(
    client: Any,
    *,
    model: str,
    prompt: str,
    response_schema: dict[str, Any] | None = None,
    system_instruction: str | None = None,
    max_output_tokens: int = 1024,
) -> dict[str, Any]:
    """Call Gemini for structured JSON output and return the parsed object.

    Uses `response_mime_type="application/json"` (plus an optional `response_schema`) so the
    model is constrained to emit a single JSON object. Raises `LLMResponseError` if the
    response carries no parseable JSON object.
    """
    config: dict[str, Any] = {
        "response_mime_type": "application/json",
        "max_output_tokens": max_output_tokens,
    }
    if system_instruction:
        config["system_instruction"] = system_instruction
    if response_schema is not None:
        config["response_schema"] = response_schema

    response = client.models.generate_content(model=model, contents=prompt, config=config)
    return parse_json_response(response)


def parse_json_response(response: Any) -> dict[str, Any]:
    """Extract a JSON object from a Gemini response (prefers `.parsed`, falls back to `.text`)."""
    parsed = getattr(response, "parsed", None)
    if isinstance(parsed, dict):
        return parsed

    text = getattr(response, "text", None)
    if not text or not str(text).strip():
        raise LLMResponseError("Gemini response contained no JSON payload (empty .text/.parsed)")
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, TypeError) as exc:
        raise LLMResponseError(f"Gemini response was not valid JSON: {text!r}") from exc
    if not isinstance(data, dict):
        raise LLMResponseError(f"Gemini JSON payload was not an object: {data!r}")
    return data
