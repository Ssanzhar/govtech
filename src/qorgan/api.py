"""Thin HTTP API around the classifier + static host for the landing page.

Run with `python -m qorgan.api` (or `uvicorn qorgan.api:app`), then open
http://localhost:8000/ — the landing page in `site/` is served at the root and its
"Try" widget calls `POST /api/analyze`.

The endpoint delegates to the one classifier contract (`classifier/predict.score`)
and the grounded explainer (`explain/explainer.explain`). If the configured backend
cannot run here (no trained weights pulled, no API key), the request falls back to
the deterministic `mock` backend and says so in the response — the demo must never
500 just because weights are gitignored.
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

from qorgan.api_admin import router as admin_router
from qorgan.api_admin_stats import router as admin_stats_router
from qorgan.api_limits import BodySizeLimitMiddleware, CrossOriginIsolationMiddleware, validation_error_handler
from qorgan.api_live import router as live_router
from qorgan.api_partner import router as partner_router
from qorgan.api_partner_export import router as partner_export_router
from qorgan.api_reports import router as reports_router
from qorgan.classifier import predict
from qorgan.cloud_tier import is_cloud, require_cloud_consent
from qorgan.config import get_config
from qorgan.explain.explainer import ExplainerError, explain
from qorgan.reports.purge import PurgeSchedule

_SITE_DIR = Path(os.environ.get("QORGAN_SITE_DIR", Path(__file__).resolve().parents[2] / "site"))
_MAX_TRANSCRIPT_CHARS = 20_000

@asynccontextmanager
async def _lifespan(_: FastAPI) -> AsyncIterator[None]:
    """Retention is the server's own job: purge at startup and every
    `QORGAN_REPORT_PURGE_INTERVAL_HOURS` (0 = leave it to cron). Single worker (see Dockerfile)."""
    cfg = get_config()
    schedule = PurgeSchedule(cfg) if cfg.report_purge_interval_hours > 0 else None
    if schedule is not None:
        schedule.start()
    try:
        yield
    finally:
        if schedule is not None:
            schedule.stop()


app = FastAPI(
    title="qorgan api",
    description="Scam-pattern verdicts with grounded evidence. A human always decides.",
    version="0.1.0",
    lifespan=_lifespan,
)
# Request hygiene for every route: bounded bodies, 422s that never echo the payload.
app.add_middleware(BodySizeLimitMiddleware)
app.add_middleware(CrossOriginIsolationMiddleware)  # live.html only: on-device ASR needs SharedArrayBuffer
app.add_exception_handler(RequestValidationError, validation_error_handler)


class AnalyzeRequest(BaseModel):
    """One transcript in, one explained verdict out."""

    model_config = ConfigDict(frozen=True)

    transcript: str = Field(min_length=1, max_length=_MAX_TRANSCRIPT_CHARS)
    locale: Literal["ru", "kk"] = "ru"
    backend: str | None = None
    # Only for the cloud second opinion (`llm`): the requester's explicit consent to sending the
    # text to Google, outside Kazakhstan (`qorgan.cloud_tier`). Ignored by local backends.
    cloud_consent: bool = False


class TagOut(BaseModel):
    id: str
    weight: float


class SpanOut(BaseModel):
    text: str
    start: int
    end: int


class ExplanationOut(BaseModel):
    reason: str
    confidence_label: str
    caveat: str
    human_note: str


class AnalyzeResponse(BaseModel):
    risk: float
    flagged: bool
    threshold: float
    backend: str
    fallback: bool
    tags: list[TagOut]
    spans: list[SpanOut]
    explanation: ExplanationOut


@app.get("/api/health")
def health() -> dict[str, object]:
    cfg = get_config()
    return {
        "status": "ok",
        "configured_backend": cfg.classifier_backend,
        "threshold": cfg.risk_threshold,
    }


@app.post("/api/analyze", response_model=AnalyzeResponse)
def analyze(req: AnalyzeRequest) -> AnalyzeResponse:
    if not req.transcript.strip():
        raise HTTPException(status_code=422, detail="transcript must not be blank")

    cfg = get_config()
    if is_cloud(req.backend, cfg):
        require_cloud_consent(cfg, consent=req.cloud_consent)
    fallback = False
    try:  # never cache: this route persists nothing, on any backend
        result = predict.score(req.transcript, backend=req.backend, use_cache=False)
    except (predict.UnknownBackendError, ValueError):
        raise
    except Exception:  # configured backend unavailable here (weights/keys) — degrade honestly
        result = predict.score(req.transcript, backend="mock")
        fallback = True

    try:
        explanation = explain(result, req.transcript, req.locale)
    except ExplainerError as exc:
        raise HTTPException(status_code=500, detail=f"explainer failed: {exc}") from exc

    return AnalyzeResponse(
        risk=result.risk,
        flagged=result.risk >= cfg.risk_threshold,
        threshold=cfg.risk_threshold,
        backend=result.backend,
        fallback=fallback,
        tags=[TagOut(id=t.id, weight=t.weight) for t in result.tags],
        spans=[SpanOut(text=s.text, start=s.start, end=s.end) for s in result.attributions],
        explanation=ExplanationOut(
            reason=explanation.reason,
            confidence_label=explanation.confidence_label,
            caveat=explanation.caveat,
            human_note=explanation.human_note,
        ),
    )


app.include_router(admin_router)
app.include_router(admin_stats_router)
app.include_router(live_router)
app.include_router(reports_router)
app.include_router(partner_router)
app.include_router(partner_export_router)

if _SITE_DIR.is_dir():
    app.mount("/", StaticFiles(directory=_SITE_DIR, html=True), name="site")


def main() -> None:
    import uvicorn

    uvicorn.run(app, host=os.environ.get("QORGAN_API_HOST", "127.0.0.1"),
                port=int(os.environ.get("QORGAN_API_PORT", "8000")))


if __name__ == "__main__":
    main()
