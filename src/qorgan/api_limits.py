"""Request hygiene shared by every route (PLAN_2026-09 C5 review findings):

- validation errors never echo the offending value back (FastAPI's default 422 body
  carries `input`, which for a transcript field is the raw, possibly unscrubbed text);
- request bodies are capped by `Content-Length`, and a body that hides its length behind
  chunked transfer encoding is refused, so an oversized payload is rejected before it is
  buffered or validated.

- the live page is served **cross-origin isolated** (COOP + COEP) so the on-device speech
  recogniser can use SharedArrayBuffer (PLAN B9, ADR D25); other pages are untouched.

All are pure functions/ASGI wrappers registered in `qorgan.api`.
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
        # Whatever else is declared: a chunked body's real size is not what Content-Length says.
        if "chunked" in headers.get("transfer-encoding", "").lower():
            await _reply(send, 411, "request body must declare Content-Length")
            return
        if length is not None and (not length.isdigit() or int(length) > self._max_bytes):
            await _reply(send, 413, f"request body exceeds {self._max_bytes} bytes")
            return
        await self._app(scope, receive, send)


# The document that needs SharedArrayBuffer (the Vosklet recogniser) — every cross-origin
# subresource on it must be CORS-loaded (`crossorigin` on script/link tags) — plus the
# scripts it starts workers from: a dedicated worker inherits the document's embedder
# policy and fails to load (an ErrorEvent with no message) unless its own script response
# carries the same COEP header. That includes the self-hosted runtimes under /vendor/:
# embed-worker.js imports transformers.js from there and onnxruntime-web starts its own workers
# from the .mjs there (ADR D61: WebKit refused the import, Chromium blocked the .mjs).
CROSS_ORIGIN_ISOLATED_PATHS = frozenset({"/live.html"})
CROSS_ORIGIN_ISOLATED_PREFIXES = ("/core/", "/vendor/")
_ISOLATION_HEADERS = (
    (b"cross-origin-opener-policy", b"same-origin"),
    (b"cross-origin-embedder-policy", b"require-corp"),
    # Always revalidate: a worker script cached *before* these headers existed fails the
    # COEP check (net::ERR_BLOCKED_BY_RESPONSE) until the cached headers are refreshed.
    (b"cache-control", b"no-cache"),
)


# Sent on every response (QA 2026-09-29). The analyst console must never be framed
# (clickjacking): X-Frame-Options for old browsers, and a CSP header carrying ONLY
# `frame-ancestors` (it restricts nothing else, and frame-ancestors is ignored in a <meta> CSP).
# No referrer leaves the site; the microphone stays allowed (the live page's on-device ASR).
# HSTS belongs on the TLS terminator in front of this server, not here (docs/DEPLOY.md).
SECURITY_HEADERS: tuple[tuple[bytes, bytes], ...] = (
    (b"x-content-type-options", b"nosniff"),
    (b"referrer-policy", b"no-referrer"),
    (b"x-frame-options", b"DENY"),
    (b"content-security-policy", b"frame-ancestors 'none'"),
    (b"permissions-policy", b"camera=(), geolocation=(), payment=(), usb=()"),
)


class SecurityHeadersMiddleware:
    """Add `SECURITY_HEADERS` to every HTTP response (replacing any same-named header)."""

    def __init__(self, app: ASGIApp) -> None:
        self._app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope.get("type") != "http":
            await self._app(scope, receive, send)
            return
        names = {name for name, _ in SECURITY_HEADERS}

        async def send_with_headers(message: MutableMapping[str, Any]) -> None:
            if message.get("type") == "http.response.start":
                headers = [(k, v) for k, v in message.get("headers", []) if k.lower() not in names]
                message = {**message, "headers": [*headers, *SECURITY_HEADERS]}
            await send(message)

        await self._app(scope, receive, send_with_headers)


# Static site files the browser must revalidate on every load (a 304 via ETag when unchanged).
# Without a Cache-Control header browsers cache JS/CSS heuristically, so after a deploy the fresh
# index.html ran against a stale i18n.js / styles.css (raw i18n keys, unstyled controls).
# /api/ sets its own policy; /models/ are large, versioned files the service worker caches.
_UNREVALIDATED_PREFIXES = ("/api/", "/models/")
_REVALIDATED_MODEL_FILES = frozenset({"/models/weights.json"})  # the shipped heads: they follow the code
_REVALIDATE = (b"cache-control", b"no-cache")


class RevalidateShellMiddleware:
    """Add `Cache-Control: no-cache` to site responses that set no cache policy themselves."""

    def __init__(self, app: ASGIApp) -> None:
        self._app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        path = str(scope.get("path", ""))
        if scope.get("type") != "http" or (path.startswith(_UNREVALIDATED_PREFIXES) and path not in _REVALIDATED_MODEL_FILES):
            await self._app(scope, receive, send)
            return

        async def send_with_policy(message: MutableMapping[str, Any]) -> None:
            if message.get("type") == "http.response.start":
                headers = list(message.get("headers", []))
                if not any(k.lower() == b"cache-control" for k, _ in headers):
                    message = {**message, "headers": [*headers, _REVALIDATE]}
            await send(message)

        await self._app(scope, receive, send_with_policy)


class CrossOriginIsolationMiddleware:
    """Add COOP/COEP to responses for `paths` (exact) and `prefixes`, and nothing else."""

    def __init__(
        self,
        app: ASGIApp,
        *,
        paths: frozenset[str] = CROSS_ORIGIN_ISOLATED_PATHS,
        prefixes: tuple[str, ...] = CROSS_ORIGIN_ISOLATED_PREFIXES,
    ) -> None:
        self._app = app
        self._paths = paths
        self._prefixes = prefixes

    def _isolated(self, path: str) -> bool:
        return path in self._paths or path.startswith(self._prefixes)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope.get("type") != "http" or not self._isolated(str(scope.get("path", ""))):
            await self._app(scope, receive, send)
            return

        async def send_with_headers(message: MutableMapping[str, Any]) -> None:
            if message.get("type") == "http.response.start":
                headers = [(k, v) for k, v in message.get("headers", []) if k.lower() not in {h for h, _ in _ISOLATION_HEADERS}]
                message = {**message, "headers": [*headers, *_ISOLATION_HEADERS]}
            await send(message)

        await self._app(scope, receive, send_with_headers)


async def _reply(send: Send, status: int, detail: str) -> None:
    body = json.dumps({"detail": detail}).encode("utf-8")
    await send({
        "type": "http.response.start",
        "status": status,
        "headers": [(b"content-type", b"application/json"), (b"content-length", str(len(body)).encode("ascii"))],
    })
    await send({"type": "http.response.body", "body": body})
