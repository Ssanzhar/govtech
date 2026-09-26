"""The cloud second opinion -- the `llm` backend (Gemini) -- is a cross-border transfer.

It sends the call text to Google, outside Kazakhstan (docs/LEGAL_ASSESSMENT.md §2). So it is
OFF unless the operator switches it on (`QORGAN_CLOUD_TIER=on`); even then a request must carry
the explicit consent of the person who asked for it (ADR D11); request-time scoring never
writes the prediction cache; and analyst routes never use it. A refused request says so and
sends nothing -- it is never silently degraded, because the caller asked for something specific.
"""

from __future__ import annotations

from fastapi import HTTPException

from qorgan.config import Config

CLOUD_BACKEND = "llm"


def is_cloud(backend: str | None, cfg: Config) -> bool:
    """Would this request (explicit backend, else the configured default) reach the cloud?"""
    return (backend or cfg.classifier_backend) == CLOUD_BACKEND


def require_cloud_consent(cfg: Config, *, consent: bool) -> None:
    """403 when the tier is switched off; 422 without the requester's explicit consent."""
    if not cfg.cloud_tier_enabled:
        raise HTTPException(
            status_code=403,
            detail="the cloud second opinion is switched off on this server (QORGAN_CLOUD_TIER); nothing was sent",
        )
    if not consent:
        raise HTTPException(
            status_code=422,
            detail="the cloud second opinion sends the text to Google, outside Kazakhstan; "
            "it needs cloud_consent=true from the person who asked for it",
        )


def refuse_cloud_for_analysts(backend: str | None, cfg: Config) -> None:
    """Analyst routes score on this server only: citizens, not analysts, may choose the cloud."""
    if is_cloud(backend, cfg):
        raise HTTPException(
            status_code=422,
            detail="analyst routes score on this server only; the cloud second opinion is a citizen's choice",
        )
