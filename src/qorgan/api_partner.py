"""Partner intake API -- the second *consented* ingress into Level 2 (PLAN_2026-09 C5,
ADR D19), deliberately not a bulk-transcript feed.

`POST /api/v1/reports` takes ONE report per request from an authenticated partner (bank
fraud desk, telecom, hotline): preferably structured tactic hits, optionally a transcript
that the partner has already PII-scrubbed (verified here, refused otherwise -- never
scrubbed silently, never echoed back), always a machine-readable `consent_basis`. The
caller number is hashed on receipt like a citizen report. Every partner action leaves a
content-free audit line; each partner has a per-minute rate limit and a rolling daily
quota. `DELETE /api/v1/reports/{receipt}` works only on the partner's own reports.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta

from fastapi import APIRouter, Depends, HTTPException, Response, Security
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from qorgan.analytics.intake import forget_report, report_incident_id
from qorgan.api_ratelimit import SlidingWindowLimiter
from qorgan.audit import AUDIT_FILENAME, AuditEntry, append_audit
from qorgan.config import Config, get_config
from qorgan.data.scrub import scrub_text
from qorgan.partners import Partner, PartnerRegistry
from qorgan.privacy.numbers import MissingHmacKeyError
from qorgan.reports.model import CONSENT_BASIS_PATTERN, PARTNER_REFERENCE_PATTERN, RECEIPT_ID_PATTERN, StoredReport
from qorgan.reports.partner import find_partner_report, partner_reports_since
from qorgan.reports.store import REPORTS_FILENAME, append_report, load_reports, prepare_report
from qorgan.taxonomy import get_taxonomy

router = APIRouter(prefix="/api/v1", tags=["partner"])

API_KEY_HEADER = "X-API-Key"
_MAX_TRANSCRIPT_CHARS = 20_000
_MAX_TACTICS = 32
# `occurred_at` may not be in the future (beyond clock skew) nor older than retention.
_OCCURRED_AT_FUTURE_TOLERANCE = timedelta(minutes=5)
_MAX_PHONE_CHARS = 32
_MAX_PHRASES = 50
_MAX_PHRASE_CHARS = 300
# A partner report is a confirmed case unless the partner says otherwise.
_DEFAULT_PARTNER_RISK_SCORE = 100.0
# Per partner, per minute -- the quota (per day) is the real budget.
_LIMITER = SlidingWindowLimiter(max_requests=60, window_seconds=60.0)
_RECEIPT_RE = re.compile(RECEIPT_ID_PATTERN)
_QUOTA_LIMIT_HEADER = "X-Quota-Limit"
_QUOTA_REMAINING_HEADER = "X-Quota-Remaining"

_api_key_scheme = APIKeyHeader(
    name=API_KEY_HEADER,
    scheme_name="PartnerApiKey",
    auto_error=False,
    description="Partner secret issued out of band (server env `QORGAN_PARTNER_API_KEYS`).",
)


# --- auth ------------------------------------------------------------------------------------


def require_partner(api_key: str | None = Security(_api_key_scheme)) -> Partner:
    """Authenticate the request's partner and charge its per-minute rate limit."""
    partner = PartnerRegistry(get_config().partner_credentials).authenticate(api_key)
    if partner is None:
        raise HTTPException(
            status_code=401, detail="missing or invalid partner API key", headers={"WWW-Authenticate": "ApiKey"}
        )
    if not _LIMITER.allow(partner.id):
        raise HTTPException(status_code=429, detail="rate limit exceeded for this partner; try again in a minute")
    return partner


def record_partner_action(
    partner: Partner, action: str, *, subject: str | None, outcome: str, cfg: Config, now: datetime
) -> None:
    entry = AuditEntry(
        timestamp=now, actor_kind="partner", actor_id=partner.id, action=action, subject=subject, outcome=outcome
    )
    append_audit(entry, cfg.data_dir / "processed" / AUDIT_FILENAME)


# --- schemas ---------------------------------------------------------------------------------


class PartnerReportIn(BaseModel):
    """One consented report. Structured tactic hits are the preferred shape; a transcript
    is accepted only if the partner has already scrubbed it (checked on receipt)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    consent_basis: str = Field(
        pattern=CONSENT_BASIS_PATTERN,
        description="Machine-readable legal basis code from the partner agreement, e.g. `customer_consent`.",
    )
    tactic_ids: tuple[str, ...] = Field(default=(), max_length=_MAX_TACTICS, description="Structured tactic hits (taxonomy ids).")
    transcript: str | None = Field(
        default=None, max_length=_MAX_TRANSCRIPT_CHARS, description="Pre-scrubbed transcript (optional)."
    )
    flagged_phrases: tuple[str, ...] = Field(default=(), max_length=_MAX_PHRASES)
    phone_number: str | None = Field(
        default=None, max_length=_MAX_PHONE_CHARS, description="Caller number; hashed on receipt, never stored raw."
    )
    risk_score: float = Field(default=_DEFAULT_PARTNER_RISK_SCORE, ge=0.0, le=100.0)
    occurred_at: datetime | None = Field(default=None, description="When the call happened (bounded by retention; the quota uses receipt time).")
    partner_reference: str | None = Field(
        default=None, pattern=PARTNER_REFERENCE_PATTERN, description="Your case id; retries with the same id are no-ops."
    )

    @field_validator("flagged_phrases")
    @classmethod
    def _phrases_bounded(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if any(len(p) > _MAX_PHRASE_CHARS for p in value):
            raise ValueError(f"each flagged phrase must be <= {_MAX_PHRASE_CHARS} chars")
        return value

    @field_validator("tactic_ids")
    @classmethod
    def _known_tactics(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        unknown = sorted(set(value) - set(get_taxonomy().tactic_ids()))
        if unknown:
            raise ValueError(f"unknown tactic ids: {unknown}")
        return tuple(dict.fromkeys(value))  # de-duplicated, order kept

    @model_validator(mode="after")
    def _signals_or_transcript(self) -> "PartnerReportIn":
        if not self.tactic_ids and not (self.transcript or "").strip():
            raise ValueError("send structured tactic_ids (preferred) and/or a pre-scrubbed transcript")
        return self


class QuotaOut(BaseModel):
    limit: int
    used: int
    remaining: int
    window_hours: int


class PartnerReportOut(BaseModel):
    """Exactly what was stored, plus the partner's remaining budget."""

    receipt_id: str
    report_id: str
    status: str
    source: str
    partner_id: str
    partner_reference: str | None
    consent_basis: str
    number_prefix: str | None
    stored_transcript: str | None
    flagged_phrases: list[str]
    tactic_ids: list[str]
    timestamp: datetime
    quota: QuotaOut


# --- endpoints -------------------------------------------------------------------------------


@router.post(
    "/reports",
    response_model=PartnerReportOut,
    status_code=201,
    responses={200: {"description": "Duplicate `partner_reference`: the existing receipt, nothing stored."}},
)
def submit(req: PartnerReportIn, response: Response, partner: Partner = Depends(require_partner)) -> PartnerReportOut:
    cfg = get_config()
    now = datetime.now(UTC)
    reports_path = cfg.data_dir / "processed" / REPORTS_FILENAME

    if req.transcript is not None and scrub_text(req.transcript) != req.transcript:
        record_partner_action(partner, "report.submit", subject=None, outcome="rejected:unscrubbed_transcript", cfg=cfg, now=now)
        raise HTTPException(
            status_code=422,
            detail="transcript contains unscrubbed personal data (numbers, cards, IINs, emails); scrub it before sending",
        )

    if req.occurred_at is not None and not _occurred_at_in_window(req.occurred_at, cfg, now):
        record_partner_action(partner, "report.submit", subject=None, outcome="rejected:occurred_at_out_of_window", cfg=cfg, now=now)
        raise HTTPException(
            status_code=422,
            detail=f"occurred_at must be within the last {cfg.report_retention_days} days and not in the future",
        )

    reports = load_reports(reports_path)  # one scan per request: quota + idempotency
    quota = _quota(reports, partner, cfg, now)
    existing = find_partner_report(reports, partner_id=partner.id, reference=req.partner_reference)
    if existing is not None:
        record_partner_action(partner, "report.submit", subject=f"receipt:{existing.receipt_id}", outcome="duplicate", cfg=cfg, now=now)
        _set_quota_headers(response, quota)
        response.status_code = 200
        return _to_out(existing, status="duplicate", quota=quota)

    if quota.remaining <= 0:
        record_partner_action(partner, "report.submit", subject=None, outcome="rejected:quota", cfg=cfg, now=now)
        raise HTTPException(
            status_code=429,
            detail=f"daily quota of {quota.limit} reports exhausted for this partner",
            headers=_quota_headers(quota),
        )

    try:
        stored = prepare_report(
            transcript=req.transcript or "",
            phone_number=req.phone_number,
            flagged_phrases=req.flagged_phrases,
            tactic_ids=req.tactic_ids,
            timestamp=req.occurred_at or now,
            risk_score=req.risk_score,
            hmac_key=cfg.number_hmac_key,
            source="partner",
            consent_basis=req.consent_basis,
            partner_id=partner.id,
            partner_reference=req.partner_reference,
            received_at=now,
        )
    except MissingHmacKeyError as exc:  # misconfigured server: refuse, never store raw
        record_partner_action(partner, "report.submit", subject=None, outcome="rejected:no_hashing_key", cfg=cfg, now=now)
        raise HTTPException(
            status_code=503, detail="reports with a caller number are not accepted: server has no number-hashing key"
        ) from exc
    except ValueError as exc:  # unparseable number
        record_partner_action(partner, "report.submit", subject=None, outcome="rejected:bad_number", cfg=cfg, now=now)
        raise HTTPException(status_code=422, detail=f"caller number not understood: {exc}") from exc

    append_report(stored, reports_path)
    record_partner_action(partner, "report.submit", subject=f"receipt:{stored.receipt_id}", outcome="stored", cfg=cfg, now=now)
    charged = QuotaOut(limit=quota.limit, used=quota.used + 1, remaining=quota.remaining - 1, window_hours=quota.window_hours)
    _set_quota_headers(response, charged)
    return _to_out(stored, status="stored", quota=charged)


@router.delete("/reports/{receipt_id}", status_code=204, response_class=Response)
def delete(receipt_id: str, partner: Partner = Depends(require_partner)) -> Response:
    if not _RECEIPT_RE.fullmatch(receipt_id):
        raise HTTPException(status_code=422, detail="malformed receipt id")
    cfg = get_config()
    now = datetime.now(UTC)
    processed = cfg.data_dir / "processed"
    owned = any(
        r.receipt_id == receipt_id and r.partner_id == partner.id for r in load_reports(processed / REPORTS_FILENAME)
    )
    if not owned:  # someone else's receipt looks exactly like an unknown one
        record_partner_action(partner, "report.delete", subject=f"receipt:{receipt_id}", outcome="not_found", cfg=cfg, now=now)
        raise HTTPException(status_code=404, detail="unknown receipt")
    forget_report(
        receipt_id,
        reports_path=processed / REPORTS_FILENAME,
        incidents_path=processed / "incidents.jsonl",
        organizations_path=processed / "organizations.jsonl",
        embeddings_path=processed / "incident_embeddings.npz",
    )
    record_partner_action(partner, "report.delete", subject=f"receipt:{receipt_id}", outcome="deleted", cfg=cfg, now=now)
    return Response(status_code=204)


# --- helpers ---------------------------------------------------------------------------------


def _occurred_at_in_window(occurred_at: datetime, cfg: Config, now: datetime) -> bool:
    stamp = occurred_at if occurred_at.tzinfo is not None else occurred_at.replace(tzinfo=UTC)
    return now - timedelta(days=cfg.report_retention_days) <= stamp <= now + _OCCURRED_AT_FUTURE_TOLERANCE


def _quota(reports: Sequence[StoredReport], partner: Partner, cfg: Config, now: datetime) -> QuotaOut:
    """Budget used in the rolling window, counted by server receipt time (`received_at`) --
    a partner-supplied `occurred_at` cannot move a report out of the window."""
    since = now - timedelta(hours=cfg.partner_quota_window_hours)
    used = partner_reports_since(reports, partner_id=partner.id, since=since)
    return QuotaOut(
        limit=partner.daily_quota,
        used=used,
        remaining=max(partner.daily_quota - used, 0),
        window_hours=cfg.partner_quota_window_hours,
    )


def _quota_headers(quota: QuotaOut) -> dict[str, str]:
    return {_QUOTA_LIMIT_HEADER: str(quota.limit), _QUOTA_REMAINING_HEADER: str(quota.remaining)}


def _set_quota_headers(response: Response, quota: QuotaOut) -> None:
    for name, value in _quota_headers(quota).items():
        response.headers[name] = value


def _to_out(stored: StoredReport, *, status: str, quota: QuotaOut) -> PartnerReportOut:
    return PartnerReportOut(
        receipt_id=stored.receipt_id,
        report_id=report_incident_id(stored),
        status=status,
        source=stored.source,
        partner_id=stored.partner_id or "",
        partner_reference=stored.partner_reference,
        consent_basis=stored.consent_basis,
        number_prefix=stored.number_prefix,
        stored_transcript=stored.transcript or None,
        flagged_phrases=list(stored.flagged_phrases),
        tactic_ids=list(stored.tactic_ids),
        timestamp=stored.timestamp,
        quota=quota,
    )
