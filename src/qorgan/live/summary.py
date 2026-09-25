"""Post-call summary + consent-gated reporting (design spec §§07, 11).

`summarize()` renders the structured verdict shown when a call ends. `build_report()`
produces an editable *draft* — number, transcript, flagged phrases, tactics, timestamp,
score — and `submit_report()` appends it to the intake file only when explicitly called:
reporting is never automatic, consent and review happen in the UI before anything is
written. A submitted report converts to a Level-2 `Incident` via `report_to_incident()`,
feeding the existing cluster/novelty/rank pipeline.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator

import qorgan.explain as explain_pkg
from qorgan.config import get_config
from qorgan.reports.model import (  # noqa: F401  (report_to_incident re-exported for callers)
    CITIZEN_CONSENT_BASIS,
    ReportSource,
    StoredReport,
    report_to_incident,
)
from qorgan.reports.store import append_report, prepare_report
from qorgan.explain.recommend import recommend
from qorgan.explain.templates import load_templates
from qorgan.live.meter import Band, band
from qorgan.live.session import LiveSessionState
from qorgan.taxonomy import get_taxonomy

_TEMPLATES_DIR = Path(explain_pkg.__file__).resolve().parent
_MAX_RECOMMENDED_ACTIONS = 3
_DEFAULT_REPORTS_FILENAME = "citizen_reports.jsonl"


class CallSummary(BaseModel):
    """The structured post-call verdict: final score, band, localized tactic names,
    top recommended actions, and the human-decides note."""

    model_config = ConfigDict(frozen=True)

    final_score: float = Field(ge=0.0, le=100.0)
    band: Band
    tactic_names: tuple[str, ...] = ()
    recommended_actions: tuple[str, ...] = ()
    human_note: str


class ReportDraft(BaseModel):
    """An editable report draft (§11): every field is reviewable and replaceable in the
    UI before submission (`model_copy(update=...)` — the models stay immutable)."""

    model_config = ConfigDict(frozen=True)

    phone_number: str | None = None
    transcript: str
    flagged_phrases: tuple[str, ...] = ()
    tactic_ids: tuple[str, ...] = ()
    timestamp: datetime
    risk_score: float = Field(ge=0.0, le=100.0)

    @field_validator("transcript")
    @classmethod
    def _transcript_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("ReportDraft.transcript must not be blank")
        return value


def summarize(state: LiveSessionState) -> CallSummary:
    """Render the post-call summary for a finished (or in-progress) session."""
    taxonomy = get_taxonomy()
    names: list[str] = []
    for tag in state.tags:
        try:
            names.append(taxonomy.display_name(tag.id, state.locale))
        except KeyError:
            continue  # unknown/future tag id must never crash the summary

    recommendation = recommend(state.tags, state.locale)
    templates = load_templates(state.locale, _TEMPLATES_DIR)
    return CallSummary(
        final_score=state.meter.score,
        band=band(state.meter.score),
        tactic_names=tuple(names),
        recommended_actions=recommendation.advices[:_MAX_RECOMMENDED_ACTIONS],
        human_note=templates.human_note,
    )


def build_report(
    state: LiveSessionState,
    *,
    phone_number: str | None = None,
    timestamp: datetime | None = None,
) -> ReportDraft:
    """Assemble the editable report draft from everything the session observed."""
    return ReportDraft(
        phone_number=phone_number,
        transcript=state.transcript(),
        flagged_phrases=state.seen_evidence,
        tactic_ids=tuple(tag.id for tag in state.tags),
        timestamp=timestamp or datetime.now(tz=UTC),
        risk_score=state.meter.score,
    )


def submit_report(
    draft: ReportDraft,
    *,
    hmac_key: bytes | None,
    reports_path: Path | None = None,
    source: ReportSource = "citizen",
    consent_basis: str = CITIZEN_CONSENT_BASIS,
) -> StoredReport:
    """Persist the reviewed draft as a minimised `StoredReport` (one JSON line).

    Only ever called from an explicit user action -- there is no automatic path here. The
    transcript is PII-scrubbed and the caller number reduced to a digest + prefix before
    anything touches disk (`reports.store.prepare_report`); the raw draft is never written.
    Returns the stored report (its `receipt_id` is what the citizen keeps for deletion).
    Raises `MissingHmacKeyError` if a number was given but no key is configured.
    """
    path = reports_path or get_config().data_dir / "processed" / _DEFAULT_REPORTS_FILENAME
    stored = prepare_report(
        transcript=draft.transcript,
        phone_number=draft.phone_number,
        flagged_phrases=draft.flagged_phrases,
        tactic_ids=draft.tactic_ids,
        timestamp=draft.timestamp,
        risk_score=draft.risk_score,
        hmac_key=hmac_key,
        source=source,
        consent_basis=consent_basis,
    )
    append_report(stored, path)
    return stored
