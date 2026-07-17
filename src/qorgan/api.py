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
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

from qorgan.api_admin import router as admin_router
from qorgan.api_admin_stats import router as admin_stats_router
from qorgan.api_live import router as live_router
from qorgan.api_live_ws import router as live_ws_router
from qorgan.classifier import predict
from qorgan.config import get_config
from qorgan.explain.explainer import ExplainerError, explain

_SITE_DIR = Path(os.environ.get("QORGAN_SITE_DIR", Path(__file__).resolve().parents[2] / "site"))
_MAX_TRANSCRIPT_CHARS = 20_000

app = FastAPI(
    title="qorgan api",
    description="Scam-pattern verdicts with grounded evidence. A human always decides.",
    version="0.1.0",
)


class AnalyzeRequest(BaseModel):
    """One transcript in, one explained verdict out."""

    model_config = ConfigDict(frozen=True)

    transcript: str = Field(min_length=1, max_length=_MAX_TRANSCRIPT_CHARS)
    locale: Literal["ru", "kk"] = "ru"
    backend: str | None = None


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
    fallback = False
    try:
        result = predict.score(req.transcript, backend=req.backend)
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
app.include_router(live_ws_router)

if _SITE_DIR.is_dir():
    app.mount("/", StaticFiles(directory=_SITE_DIR, html=True), name="site")


def main() -> None:
    import uvicorn

    uvicorn.run(app, host=os.environ.get("QORGAN_API_HOST", "127.0.0.1"),
                port=int(os.environ.get("QORGAN_API_PORT", "8000")))


if __name__ == "__main__":
    main()
