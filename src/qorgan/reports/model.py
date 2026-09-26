"""`StoredReport` -- a consented report as it is persisted, plus its conversion to an
`Incident` (PLAN_2026-09 B5/C2).

The schema is the last line of defence: the transcript must already be PII-scrubbed
(`scrub_text` is a fixed point on it), and the caller number may only appear as a digest
plus a coarse display prefix. Anything else fails validation and is never written.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from types import MappingProxyType
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from qorgan.data.schema import Incident, Label, TacticTag, spans_from_phrases
from qorgan.data.scrub import scrub_text
from qorgan.partners import PARTNER_ID_PATTERN
from qorgan.privacy.numbers import is_number_hash

RECEIPT_ID_PATTERN = r"^[0-9a-f]{24}$"
_RISK_SCORE_MAX = 100.0
_NUMBER_PREFIX_PATTERN = r"^\+\d{1,3}( \d{3})? \*\*\*$"
# A partner's own case id: short, opaque, safe to echo in receipts and audit lines.
PARTNER_REFERENCE_PATTERN = r"^[A-Za-z0-9._:-]{1,64}$"
# `consent_basis` is a code from the data-sharing agreement (e.g. `customer_consent`), not prose.
CONSENT_BASIS_PATTERN = r"^[a-z][a-z0-9_]{2,63}$"

ReportSource = Literal["citizen", "partner"]
CITIZEN_CONSENT_BASIS = "citizen_explicit_submit"

# The consent sentence a citizen ticked, by version (privacy iteration 2026-09-26; legal review
# M3, PD Law Art. 25(2)(5)). Each version maps to the SHA-256 of its exact wording in every
# page language (`site/i18n.js`, keys `report.consent*`, canonical JSON), so a stored
# `consent_version` proves what was agreed to. `tests/reports/test_consent_versions.py` fails
# if the page's wording changes without a new version here. Never edit an entry; add one, and
# drop an old one only after every report citing it has expired.
CONSENT_VERSION_PATTERN = r"^[a-z][a-z0-9-]{0,31}-v[0-9]{1,4}$"
CITIZEN_CONSENT_VERSIONS: Mapping[str, str] = MappingProxyType({
    "report-v1": "1e11422686c366970dcbb06d19fac06c51f269b3f6247caa261b20c759b26f67",
})


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
    # Server clock at storage time (never client-supplied); quotas and retention anchor on
    # it. Optional only so reports stored before it existed still load.
    received_at: datetime | None = None
    risk_score: float = Field(ge=0.0, le=_RISK_SCORE_MAX)
    source: ReportSource = "citizen"
    consent_basis: str = Field(default=CITIZEN_CONSENT_BASIS, min_length=1)
    # Which consent text the citizen ticked (`CITIZEN_CONSENT_VERSIONS`). Required by the
    # citizen route; None on partner reports (their basis is `consent_basis`) and on rows
    # stored before it existed.
    consent_version: str | None = Field(default=None, pattern=CONSENT_VERSION_PATTERN)
    # Partner provenance (PLAN_2026-09 C5): who sent it and their own case reference.
    partner_id: str | None = Field(default=None, pattern=PARTNER_ID_PATTERN)
    partner_reference: str | None = Field(default=None, pattern=PARTNER_REFERENCE_PATTERN)

    @field_validator("transcript")
    @classmethod
    def _transcript_is_scrubbed(cls, value: str) -> str:
        if scrub_text(value) != value:
            raise ValueError("StoredReport.transcript still contains PII -- scrub before storing")
        return value

    @model_validator(mode="after")
    def _shape_matches_source(self) -> "StoredReport":
        if (self.source == "partner") != (self.partner_id is not None):
            raise ValueError("StoredReport.partner_id is required for source='partner' and forbidden otherwise")
        if not self.transcript.strip():
            # Signals-only reports (structured tactic hits, no transcript) are a partner shape.
            if self.source != "partner":
                raise ValueError("StoredReport.transcript must not be blank")
            if not self.tactic_ids:
                raise ValueError("a report without a transcript must carry at least one tactic id")
        return self

    @property
    def is_signals_only(self) -> bool:
        return not self.transcript.strip()

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
