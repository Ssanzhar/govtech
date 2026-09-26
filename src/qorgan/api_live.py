"""Read-only facts for the live page: demo scenarios and what this install can do.

The server-side live-call session API (`/api/live/session/*`) is retired (2026-09-26; planned
as PLAN_2026-09 B6 once scoring moved to the device). It held every utterance of a call in
server memory with no expiry, and its `/report` route stored a citizen report with no review
and no consent -- a second citizen ingress the architecture says does not exist. The citizen
page scores on the device (`site/core/`); the Streamlit dev harness drives `qorgan.live`
in-process. Nothing here accepts call content.
"""

from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel, ConfigDict

from qorgan.data.demo_transcripts import LIVE_DEMO_CALLS

router = APIRouter(prefix="/api/live", tags=["live"])

# Human-readable scenario labels for the demo picker.
_SCENARIO_LABELS: dict[str, str] = {
    "live_scam_bank_ru": "Bank security scam (RU)",
    "live_scam_bank_kk": "Bank security scam (KK)",
    "live_hard_negative_bank_ru": "Real bank call — hard negative (RU)",
}


class ScenarioOut(BaseModel):
    id: str
    label: str
    lines: list[str]


class ScenariosResponse(BaseModel):
    scenarios: list[ScenarioOut]


class CapabilitiesResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    microphone: bool
    reason: str


# Raw call audio never reaches this server (PLAN_2026-09 §2 invariant 1; ADR D12).
_MICROPHONE_REASON = (
    "this server does not accept audio -- microphone mode runs speech recognition on your "
    "device (live.html), and nothing is sent unless you send a report"
)


@router.get("/capabilities", response_model=CapabilitiesResponse)
def capabilities() -> CapabilitiesResponse:
    """What this server offers the live page. Audio is never one of them."""
    return CapabilitiesResponse(microphone=False, reason=_MICROPHONE_REASON)


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
