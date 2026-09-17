"""Request hygiene shared by every route (PLAN_2026-09 C5 review findings):

- validation errors never echo the offending value back (FastAPI's default 422 body
  carries `input`, which for a transcript field is the raw, possibly unscrubbed text);
- request bodies are capped by `Content-Length`, and a body that hides its length behind
  chunked transfer encoding is refused, so an oversized payload is rejected before it is
  buffered or validated.

Both are pure functions/ASGI wrappers registered in `qorgan.api`.
"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable, MutableMapping
from typing import Any

from fastapi import Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

# Generous for the largest legitimate body (a 20 000-char transcript + 50 phrases ≈ 100 KB).
MAX_BODY_BYTES = 256 * 1024
_ECHOED_KEYS = ("input", "url")
_METHODS_WITH_BODIES = frozenset({"POST", "PUT", "PATCH"})

Scope = MutableMapping[str, Any]
Receive = Callable[[], Awaitable[MutableMapping[str, Any]]]
Send = Callable[[MutableMapping[str, Any]], Awaitable[None]]
ASGIApp = Callable[[Scope, Receive, Send], Awaitable[None]]


async def validation_error_handler(_request: Request, exc: RequestValidationError) -> JSONResponse:
    """422 with `loc` / `msg` / `type` per error and nothing the client sent."""
    errors = [{k: v for k, v in error.items() if k not in _ECHOED_KEYS} for error in exc.errors()]
    return JSONResponse(status_code=422, content={"detail": _jsonable(errors)})


def _jsonable(value: Any) -> Any:
    return json.loads(json.dumps(value, default=str))


class BodySizeLimitMiddleware:
    """Reject bodies over `max_bytes` (413) and bodies of undeclared length (411)."""

    def __init__(self, app: ASGIApp, *, max_bytes: int = MAX_BODY_BYTES) -> None:
        if max_bytes <= 0:
            raise ValueError("max_bytes must be > 0")
        self._app = app
        self._max_bytes = max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope.get("type") != "http" or scope.get("method") not in _METHODS_WITH_BODIES:
            await self._app(scope, receive, send)
            return
        headers = {k.decode("latin-1").lower(): v.decode("latin-1") for k, v in scope.get("headers", [])}
        length = headers.get("content-length")
        if length is None:
            if "chunked" in headers.get("transfer-encoding", "").lower():
                await _reply(send, 411, "request body must declare Content-Length")
                return
        elif not length.isdigit() or int(length) > self._max_bytes:
            await _reply(send, 413, f"request body exceeds {self._max_bytes} bytes")
            return
        await self._app(scope, receive, send)


async def _reply(send: Send, status: int, detail: str) -> None:
    body = json.dumps({"detail": detail}).encode("utf-8")
    await send({
        "type": "http.response.start",
        "status": status,
        "headers": [(b"content-type", b"application/json"), (b"content-length", str(len(body)).encode("ascii"))],
    })
    await send({"type": "http.response.body", "body": body})
