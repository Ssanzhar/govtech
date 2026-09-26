"""Authentication, roles and access auditing for the analyst console (`/api/admin`).

Every admin route depends on `require_analyst` (enforced by `tests/test_architecture.py`,
invariant 5): the analyst is identified **only** by the credential in `X-Analyst-Key`
(`QORGAN_ANALYST_KEYS`), never by a caller-supplied name. The console fails closed: with no
analyst credentials, or no audit-chain key to account for what analysts do, every route
answers 503. Revealing a full transcript additionally depends on `require_investigator`.

What is audited (content-free, into the chained log): a presented-but-wrong key, a refused
role, a refused open (rate limit), a session start, and every state-changing or revealing
action (`case.open`, `org.*`, `reports.ingest`). A request with *no* key carries no identity
and writes nothing. The presented secret is never recorded, not even hashed.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime

from fastapi import Depends, HTTPException, Request, Response, Security
from fastapi.security import APIKeyHeader

from qorgan.analysts import Analyst, AnalystRegistry
from qorgan.api_ratelimit import SlidingWindowLimiter
from qorgan.audit import (
    AUDIT_FILENAME,
    AccessPurpose,
    AuditEntry,
    AuditIntegrityError,
    MissingAuditKeyError,
    append_audit,
)
from qorgan.config import get_config

ANALYST_KEY_HEADER = "X-Analyst-Key"
UNAUTHENTICATED_ACTOR = "unauthenticated"
_NO_STORE = {"Cache-Control": "no-store"}

# Wrong keys per client address per minute: brute force is hopeless against >= 16-char
# secrets anyway; this mainly keeps a key-guessing loop from flooding the audit log.
_FAILED_AUTH_LIMITER = SlidingWindowLimiter(max_requests=10, window_seconds=60.0)
# Requests per analyst per minute: generous for a human at a console, not for a scraper.
_ANALYST_LIMITER = SlidingWindowLimiter(max_requests=120, window_seconds=60.0)
# Full-transcript opens per investigator per hour: reading calls one by one is an
# investigation; reading them in bulk is surveillance, and it is refused and audited.
_OPEN_LIMITER = SlidingWindowLimiter(max_requests=30, window_seconds=3600.0)

_LOGGER = logging.getLogger(__name__)

_key_scheme = APIKeyHeader(
    name=ANALYST_KEY_HEADER,
    scheme_name="AnalystKey",
    auto_error=False,
    description="Analyst secret issued out of band (server env `QORGAN_ANALYST_KEYS`, `id:secret:role`).",
)


def reset_limiters() -> None:
    """Test seam: empty every in-process window."""
    for limiter in (_FAILED_AUTH_LIMITER, _ANALYST_LIMITER, _OPEN_LIMITER):
        limiter.reset()


def require_analyst(request: Request, response: Response, api_key: str | None = Security(_key_scheme)) -> Analyst:
    """The authenticated analyst, or 503 (console not configured) / 401 / 429."""
    cfg = get_config()
    if not cfg.analyst_credentials:
        raise HTTPException(
            status_code=503,
            detail="analyst console is closed: no analyst credentials are configured on this server (QORGAN_ANALYST_KEYS)",
        )
    if cfg.audit_chain_key is None:
        raise HTTPException(
            status_code=503,
            detail="analyst console is closed: no audit-chain key is configured (QORGAN_AUDIT_CHAIN_KEY), "
            "so console actions could not be accounted for",
        )
    analyst = AnalystRegistry(cfg.analyst_credentials).authenticate(api_key)
    if analyst is None:
        if api_key:
            if not _FAILED_AUTH_LIMITER.allow(_client(request)):
                raise HTTPException(status_code=429, detail="too many failed sign-in attempts; try again in a minute")
            record_analyst_action(
                UNAUTHENTICATED_ACTOR, "auth.denied", subject=_route(request), outcome="denied:invalid_key"
            )
        raise HTTPException(
            status_code=401, detail="missing or invalid analyst key", headers={"WWW-Authenticate": "ApiKey"}
        )
    if not _ANALYST_LIMITER.allow(analyst.id):
        raise HTTPException(status_code=429, detail="request budget for this analyst exceeded; try again in a minute")
    response.headers.update(_NO_STORE)
    return analyst


def require_investigator(request: Request, analyst: Analyst = Depends(require_analyst)) -> Analyst:
    """The authenticated analyst if they hold the investigator role; a refusal is audited."""
    if not analyst.has_role("investigator"):
        record_analyst_action(analyst.id, "access.denied", subject=_route(request), outcome="denied:needs_investigator")
        raise HTTPException(
            status_code=403,
            detail="opening a full transcript needs the investigator role; your role is analyst",
        )
    return analyst


def charge_case_open(analyst: Analyst, subject: str) -> None:
    """Charge one full-transcript open to the investigator's hourly budget (429 + audited)."""
    if not _OPEN_LIMITER.allow(analyst.id):
        record_analyst_action(analyst.id, "access.denied", subject=subject, outcome="denied:open_rate_limited")
        raise HTTPException(
            status_code=429, detail="hourly limit of opened transcripts reached; the refusal was logged"
        )


def record_analyst_action(
    actor_id: str,
    action: str,
    *,
    subject: str | None = None,
    outcome: str = "ok",
    purpose: AccessPurpose | None = None,
) -> None:
    """Append one content-free, chained audit line -- or refuse the action (503).

    Raises `pydantic.ValidationError` if a field carries content (the caller maps it to 422
    *before* anything happened). Any failure to write the line is a 503: an action that
    cannot be accounted for does not happen, and nothing is released.
    """
    cfg = get_config()
    entry = AuditEntry(
        timestamp=datetime.now(UTC), actor_kind="analyst", actor_id=actor_id,
        action=action, subject=subject, outcome=outcome, purpose=purpose,
    )
    key = cfg.audit_chain_key.get_secret_value() if cfg.audit_chain_key is not None else None
    try:
        append_audit(entry, cfg.data_dir / "processed" / AUDIT_FILENAME, key=key)
    except (OSError, AuditIntegrityError, MissingAuditKeyError) as exc:
        _LOGGER.error("audit append failed (%s); refusing %s", type(exc).__name__, action)
        raise HTTPException(
            status_code=503, detail="the audit log is unavailable, so the action was refused and nothing was released"
        ) from exc


def _route(request: Request) -> str:
    """`route:<METHOD> <path template>` -- the template, never the caller's raw path."""
    route = request.scope.get("route")
    template = getattr(route, "path", None) or "unknown"
    return f"route:{request.method} {template}"


def _client(request: Request) -> str:
    return request.client.host if request.client else "unknown"
