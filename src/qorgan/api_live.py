"""Live-call replay HTTP API — a thin wrapper over `qorgan.live` (session/meter/summary),
the same pipeline `app/live_view.py` drives. One suspicion-meter session per HTTP session
id, held in the in-memory `qorgan.live.session_store.SessionStore`.

Backend resolution mirrors `qorgan.api.analyze`: an explicit unknown backend is a client
error (4xx); any other failure to run the configured/requested backend here (no weights,
no API key) degrades honestly to the deterministic `mock` backend, recorded on the
session so every turn scores with the same resolved backend.
"""

from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from qorgan.analytics.intake import report_incident_id
from qorgan.asr.stream import CommittedUtterance
from qorgan.classifier import predict
from qorgan.data.demo_transcripts import LIVE_DEMO_CALLS
from qorgan.live.session import LiveUpdate, advance, initial_session
from qorgan.live.session_store import SessionStore
from qorgan.live.summary import build_report, submit_report, summarize

router = APIRouter(prefix="/api/live", tags=["live"])

# One store per process — fine for the single-worker demo server this ships with.
_STORE = SessionStore()

_MAX_UTTERANCE_CHARS = 4_000
_MAX_PHONE_CHARS = 32
# Innocuous probe text used only to test-drive the requested backend at session start;
# never scored for real, never shown to a user.
_BACKEND_PROBE_TEXT = "Алло, здравствуйте."
# HTTP replay has no real ASR confidence signal (the client sends plain text), so every
# turn is treated as fully transcribed — same convention as `asr.stream.replay_transcript`.
_DEFAULT_UTTERANCE_CONFIDENCE = 1.0

# Human-readable scenario labels (RU-facing UI copy for the demo picker).
_SCENARIO_LABELS: dict[str, str] = {
    "live_scam_bank_ru": "Bank security scam (RU)",
    "live_hard_negative_bank_ru": "Real bank call — hard negative (RU)",
}
Locale = Literal["ru", "kk"]


class ScenarioOut(BaseModel):
    id: str
    label: str
    lines: list[str]


class ScenariosResponse(BaseModel):
    scenarios: list[ScenarioOut]


class SessionCreateRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    locale: Locale = "ru"
    backend: str | None = None


class SessionCreateResponse(BaseModel):
    session_id: str
    locale: str
    backend: str


class UtteranceRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    text: str = Field(min_length=1, max_length=_MAX_UTTERANCE_CHARS)


class TagWeightOut(BaseModel):
    id: str
    weight: float


class EvidenceOut(BaseModel):
    text: str


class UtteranceResponse(BaseModel):
    meter: float
    band: str
    latched: bool
    turn: int
    risk: float
    advice: list[str]
    note: str | None
    new_evidence: list[EvidenceOut]
    tactics: list[TagWeightOut]


class EndResponse(BaseModel):
    final_score: float
    band: str
    tactic_names: list[str]
    recommended_actions: list[str]
    human_note: str


class ReportRequest(BaseModel):
    """Consent-gated post-call report: submitting IS the explicit user action (§11)."""

    model_config = ConfigDict(frozen=True)

    phone_number: str | None = Field(default=None, max_length=_MAX_PHONE_CHARS)


class ReportResponse(BaseModel):
    report_id: str
    risk_score: float
    status: str


@router.get("/scenarios", response_model=ScenariosResponse)
def scenarios() -> ScenariosResponse:
    return ScenariosResponse(
        scenarios=[
            ScenarioOut(
                id=scenario_id,
                label=_SCENARIO_LABELS.get(scenario_id, scenario_id),
                lines=[line.strip() for line in script.splitlines() if line.strip()],
            )
            for scenario_id, script in LIVE_DEMO_CALLS.items()
        ]
    )


def resolve_backend(requested: str | None) -> str:
    """Probe the requested/configured backend once, degrading honestly to `mock`.

    An explicitly unknown backend stays a caller error (`UnknownBackendError`/`ValueError`
    propagate); any other failure (missing weights, missing API key) resolves to `mock`.
    Shared by the HTTP session-create and the microphone WebSocket.
    """
    try:
        predict.score(_BACKEND_PROBE_TEXT, backend=requested)
    except (predict.UnknownBackendError, ValueError):
        raise
    except Exception:  # configured backend unavailable here (weights/keys) — degrade honestly
        return "mock"
    return requested or _configured_backend()


@router.post("/session", response_model=SessionCreateResponse)
def create_session(req: SessionCreateRequest) -> SessionCreateResponse:
    try:
        resolved_backend = resolve_backend(req.backend)
    except (predict.UnknownBackendError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    try:
        state = initial_session(req.locale, backend=resolved_backend)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    session_id = _STORE.create(state, backend=resolved_backend, locale=req.locale)
    return SessionCreateResponse(session_id=session_id, locale=req.locale, backend=resolved_backend)


@router.post("/session/{session_id}/utterance", response_model=UtteranceResponse)
def post_utterance(session_id: str, req: UtteranceRequest) -> UtteranceResponse:
    entry = _STORE.get(session_id)
    if entry is None:
        raise HTTPException(status_code=404, detail="unknown session")
    if not req.text.strip():
        raise HTTPException(status_code=422, detail="text must not be blank")

    committed = CommittedUtterance(text=req.text, confidence=_DEFAULT_UTTERANCE_CONFIDENCE)
    new_state, update = advance(entry.state, committed)
    _STORE.update(session_id, new_state)

    return update_response(update)


def update_response(update: LiveUpdate) -> UtteranceResponse:
    """Project one meter update into the wire shape (shared with the microphone WS)."""
    return UtteranceResponse(
        meter=update.meter.score,
        band=update.band,
        latched=update.meter.latched,
        turn=update.meter.turn_index,
        risk=update.result.risk,
        advice=list(update.recommendation.advices),
        note=update.recommendation.note,
        new_evidence=[EvidenceOut(text=span.text) for span in update.new_evidence],
        tactics=[TagWeightOut(id=tag.id, weight=tag.weight) for tag in update.result.tags],
    )


@router.post("/session/{session_id}/end", response_model=EndResponse)
def end_session(session_id: str) -> EndResponse:
    entry = _STORE.end(session_id)
    if entry is None:
        raise HTTPException(status_code=404, detail="unknown session")

    summary = summarize(entry.state)
    return EndResponse(
        final_score=summary.final_score,
        band=summary.band,
        tactic_names=list(summary.tactic_names),
        recommended_actions=list(summary.recommended_actions),
        human_note=summary.human_note,
    )


@router.post("/session/{session_id}/report", response_model=ReportResponse)
def report_session(session_id: str, req: ReportRequest) -> ReportResponse:
    """Submit the finished call as a citizen report (→ analyst-dashboard intake).

    Works on an ended session (the normal flow) or ends a still-live one first. The
    stash entry is consumed, so each session can be reported exactly once.
    """
    _STORE.end(session_id)  # a still-live session ends now; no-op if already ended
    entry = _STORE.take_ended(session_id)
    if entry is None:
        raise HTTPException(status_code=404, detail="unknown or already-reported session")

    phone = (req.phone_number or "").strip() or None
    try:
        draft = build_report(entry.state, phone_number=phone)
    except ValueError as exc:  # blank transcript — nothing was said yet
        raise HTTPException(status_code=422, detail="nothing to report yet") from exc

    submit_report(draft)
    return ReportResponse(
        report_id=report_incident_id(draft), risk_score=draft.risk_score, status="submitted"
    )


def _configured_backend() -> str:
    from qorgan.config import get_config

    return get_config().classifier_backend
