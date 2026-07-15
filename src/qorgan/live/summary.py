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
from qorgan.data.schema import Incident, Label, TacticTag, spans_from_phrases
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


def report_to_incident(
    draft: ReportDraft, *, incident_id: str, dialogue_id: str | None = None
) -> Incident:
    """Convert a (possibly user-edited) draft into a Level-2 `Incident`.

    Flagged phrases are re-grounded against the (possibly edited) transcript via
    verbatim search; anything no longer present is dropped, never fabricated.
    """
    spans = spans_from_phrases(draft.flagged_phrases, draft.transcript)
    label = Label(
        risk=draft.risk_score / 100.0,
        tactic_tags=tuple(TacticTag(id=tactic_id) for tactic_id in draft.tactic_ids),
        trigger_spans=spans,
    )
    return Incident(
        id=incident_id,
        dialogue_id=dialogue_id or f"live-{incident_id}",
        transcript=draft.transcript,
        label=label,
        phone_number=draft.phone_number,
        timestamp=draft.timestamp,
    )


def submit_report(draft: ReportDraft, *, reports_path: Path | None = None) -> Path:
    """Append the reviewed draft to the intake file (one JSON line per report).

    Only ever called from an explicit user action — there is no automatic path here.
    Returns the path written, defaulting to `data/processed/citizen_reports.jsonl`.
    """
    path = reports_path or get_config().data_dir / "processed" / _DEFAULT_REPORTS_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(draft.model_dump_json() + "\n")
    return path
