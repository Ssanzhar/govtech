"""Offline batch ASR wrapper (faster-whisper) with a manual-transcript passthrough
fallback (gap G10).

Used only to transcribe a few demo clips -- never streaming, never fine-tuned (per
`docs/SCOPE.md`). `authored_heldout` anchors (Day 2) can be supplied as pre-transcribed text
via `manual_transcript` when no audio pipeline is available yet, so the eval split isn't
blocked on ASR landing first.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from qorgan.config import get_config


def transcribe(
    audio_path: str | Path | None = None,
    *,
    manual_transcript: str | None = None,
    model: Any | None = None,
) -> str:
    """Return a transcript string.

    If `manual_transcript` is given, it is returned as-is (no ASR call at all) -- this is
    the fallback path for demo clips / authored_heldout anchors that already have a trusted
    transcript. Otherwise, `audio_path` is transcribed offline via faster-whisper.

    Args:
        audio_path: Path to an audio file. Required unless `manual_transcript` is given.
        manual_transcript: A pre-existing transcript to pass through unchanged. Takes
            priority over `audio_path` if both are given (no model is loaded/called).
        model: An injected faster-whisper `WhisperModel`-like object exposing
            `.transcribe(path) -> (segments, info)`. Defaults to a lazily-loaded real
            model (never constructed in tests).
    """
    if manual_transcript is not None:
        if not manual_transcript.strip():
            raise ValueError("manual_transcript must not be blank")
        return manual_transcript

    if audio_path is None:
        raise ValueError("Either audio_path or manual_transcript must be provided")

    resolved_path = Path(audio_path)
    if not resolved_path.exists():
        raise FileNotFoundError(f"Audio file not found: {resolved_path}")

    whisper_model = model if model is not None else _load_model()
    segments, _info = whisper_model.transcribe(str(resolved_path))
    return " ".join(segment.text.strip() for segment in segments).strip()


def _load_model() -> Any:  # pragma: no cover - real model load, not exercised in tests
    from faster_whisper import WhisperModel

    cfg = get_config()
    cfg.model_dir.mkdir(parents=True, exist_ok=True)
    return WhisperModel(
        cfg.whisper_model_size,
        device="cpu",
        compute_type="int8",
        download_root=str(cfg.model_dir),
    )
