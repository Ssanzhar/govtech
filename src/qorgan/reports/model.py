"""`StoredReport` -- a consented report as it is persisted, plus its conversion to an
`Incident` (PLAN_2026-09 B5/C2).

The schema is the last line of defence: the transcript must already be PII-scrubbed
(`scrub_text` is a fixed point on it), and the caller number may only appear as a digest
plus a coarse display prefix. Anything else fails validation and is never written.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from qorgan.data.schema import Incident, Label, TacticTag, spans_from_phrases
from qorgan.data.scrub import scrub_text
from qorgan.privacy.numbers import is_number_hash

RECEIPT_ID_PATTERN = r"^[0-9a-f]{24}$"
_RISK_SCORE_MAX = 100.0
_NUMBER_PREFIX_PATTERN = r"^\+\d{1,3}( \d{3})? \*\*\*$"

ReportSource = Literal["citizen", "partner"]
CITIZEN_CONSENT_BASIS = "citizen_explicit_submit"


class StoredReport(BaseModel):
    """What the reports file holds. Frozen; edits happen on the draft, before this."""

    model_config = ConfigDict(frozen=True)

    receipt_id: str = Field(pattern=RECEIPT_ID_PATTERN)
    transcript: str
    number_hash: str | None = None
    number_prefix: str | None = Field(default=None, pattern=_NUMBER_PREFIX_PATTERN)
    flagged_phrases: tuple[str, ...] = ()
    tactic_ids: tuple[str, ...] = ()
    timestamp: datetime
    risk_score: float = Field(ge=0.0, le=_RISK_SCORE_MAX)
    source: ReportSource = "citizen"
    consent_basis: str = Field(default=CITIZEN_CONSENT_BASIS, min_length=1)

    @field_validator("transcript")
    @classmethod
    def _transcript_is_scrubbed_and_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("StoredReport.transcript must not be blank")
        if scrub_text(value) != value:
            raise ValueError("StoredReport.transcript still contains PII -- scrub before storing")
        return value

    @field_validator("number_hash")
    @classmethod
    def _number_is_hashed(cls, value: str | None) -> str | None:
        if value is not None and not is_number_hash(value):
            raise ValueError("StoredReport.number_hash must be a number digest, never a raw number")
        return value


def report_to_incident(
    report: StoredReport, *, incident_id: str, dialogue_id: str | None = None
) -> Incident:
    """Convert a stored report into a Level-2 `Incident`.

    Flagged phrases are re-grounded against the stored (scrubbed, possibly user-edited)
    transcript via verbatim search; anything no longer present is dropped, never fabricated.
    """
    spans = spans_from_phrases(report.flagged_phrases, report.transcript)
    label = Label(
        risk=report.risk_score / _RISK_SCORE_MAX,
        tactic_tags=tuple(TacticTag(id=tactic_id) for tactic_id in report.tactic_ids),
        trigger_spans=spans,
    )
    return Incident(
        id=incident_id,
        dialogue_id=dialogue_id or f"live-{incident_id}",
        transcript=report.transcript,
        label=label,
        number_hash=report.number_hash,
        number_prefix=report.number_prefix,
        timestamp=report.timestamp,
    )
