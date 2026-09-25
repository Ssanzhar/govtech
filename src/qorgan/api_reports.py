"""Consented reports API -- the single ingress into Level 2 (ADR D13), stored minimised
(ADR D14), deletable by receipt (PLAN_2026-09 B5/C3).

`POST /api/reports` is the explicit user action. The client sends the reviewed draft; the
server reduces it (`reports.store.prepare_report`: PII-scrubbed transcript, caller number ->
HMAC digest + coarse prefix) and answers with exactly what it stored plus a receipt.
`DELETE /api/reports/{receipt}` forgets the report everywhere it reached, including an
incident already folded into the analysis (`analytics.intake.forget_report`).
"""

from __future__ import annotations

import re
from datetime import UTC, datetime
from typing import Literal

from fastapi import APIRouter, HTTPException, Request, Response
from pydantic import BaseModel, ConfigDict, Field, field_validator

from qorgan.analytics.intake import forget_report, report_incident_id
from qorgan.api_ratelimit import SlidingWindowLimiter
from qorgan.config import get_config
from qorgan.privacy.numbers import MissingHmacKeyError
from qorgan.reports.model import RECEIPT_ID_PATTERN
from qorgan.reports.store import REPORTS_FILENAME, append_report, prepare_report
from qorgan.taxonomy import get_taxonomy

router = APIRouter(prefix="/api/reports", tags=["reports"])

_MAX_TRANSCRIPT_CHARS = 20_000
_MAX_PHONE_CHARS = 32
_MAX_PHRASES = 50
_MAX_PHRASE_CHARS = 300
_MAX_TACTICS = 32
# Per client address: a citizen files a handful of reports, not hundreds.
_LIMITER = SlidingWindowLimiter(max_requests=20, window_seconds=60.0)
_RECEIPT_RE = re.compile(RECEIPT_ID_PATTERN)


class ReportSubmission(BaseModel):
    """A reviewed, consented draft as the client sends it (raw number allowed *in transit*
    over TLS; it is hashed on receipt and never persisted)."""

    model_config = ConfigDict(frozen=True)

    transcript: str = Field(min_length=1, max_length=_MAX_TRANSCRIPT_CHARS)
    phone_number: str | None = Field(default=None, max_length=_MAX_PHONE_CHARS)
    flagged_phrases: tuple[str, ...] = Field(default=(), max_length=_MAX_PHRASES)
    tactic_ids: tuple[str, ...] = Field(default=(), max_length=_MAX_TACTICS)
    risk_score: float = Field(ge=0.0, le=100.0)
    consent: Literal[True]
    timestamp: datetime | None = None

    @field_validator("transcript")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("transcript must not be blank")
        return value

    @field_validator("flagged_phrases")
    @classmethod
    def _phrases_bounded(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if any(len(p) > _MAX_PHRASE_CHARS for p in value):
            raise ValueError(f"each flagged phrase must be <= {_MAX_PHRASE_CHARS} chars")
        return value

    @field_validator("tactic_ids")
    @classmethod
    def _known_tactics(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        known = set(get_taxonomy().tactic_ids())
        unknown = sorted(set(value) - known)
        if unknown:
            raise ValueError(f"unknown tactic ids: {unknown}")
        return tuple(dict.fromkeys(value))


class ReportReceipt(BaseModel):
    """Exactly what was stored -- shown back so the citizen sees what left their device."""

    receipt_id: str
    report_id: str
    status: str
    number_prefix: str | None
    stored_transcript: str
    flagged_phrases: list[str]
    tactic_ids: list[str]
    timestamp: datetime


@router.post("", response_model=ReportReceipt, status_code=201)
def submit(req: ReportSubmission, request: Request) -> ReportReceipt:
    if not _LIMITER.allow(_client_key(request)):
        raise HTTPException(status_code=429, detail="too many reports from this client; try again in a minute")
    cfg = get_config()
    now = datetime.now(UTC)
    try:
        stored = prepare_report(
            transcript=req.transcript,
            phone_number=req.phone_number,
            flagged_phrases=req.flagged_phrases,
            tactic_ids=req.tactic_ids,
            timestamp=req.timestamp or now,
            risk_score=req.risk_score,
            hmac_key=cfg.number_hmac_key,
            received_at=now,
        )
    except MissingHmacKeyError as exc:  # misconfigured server: refuse, never store raw
        raise HTTPException(
            status_code=503, detail="reports with a caller number are not accepted: server has no number-hashing key"
        ) from exc
    except ValueError as exc:  # unparseable number
        raise HTTPException(status_code=422, detail=f"caller number not understood: {exc}") from exc

    append_report(stored, cfg.data_dir / "processed" / REPORTS_FILENAME)
    return ReportReceipt(
        receipt_id=stored.receipt_id,
        report_id=report_incident_id(stored),
        status="stored",
        number_prefix=stored.number_prefix,
        stored_transcript=stored.transcript,
        flagged_phrases=list(stored.flagged_phrases),
        tactic_ids=list(stored.tactic_ids),
        timestamp=stored.timestamp,
    )


@router.delete("/{receipt_id}", status_code=204, response_class=Response)
def delete(receipt_id: str) -> Response:
    if not _RECEIPT_RE.fullmatch(receipt_id):
        raise HTTPException(status_code=422, detail="malformed receipt id")
    processed = get_config().data_dir / "processed"
    summary = forget_report(
        receipt_id,
        reports_path=processed / REPORTS_FILENAME,
        incidents_path=processed / "incidents.jsonl",
        organizations_path=processed / "organizations.jsonl",
        embeddings_path=processed / "incident_embeddings.npz",
    )
    if summary is None:
        raise HTTPException(status_code=404, detail="unknown receipt")
    return Response(status_code=204)


def _client_key(request: Request) -> str:
    return request.client.host if request.client else "unknown"
