"""Microphone WebSocket for the live-call page — browser PCM16 chunks → dual KK/RU Vosk
streaming recognition → the same suspicion-meter session the replay HTTP API drives.

Protocol (server frames are JSON; client audio frames are binary):
  client → {"type": "start", "locale": "ru"|"kk", "backend": str|null}   (first frame)
  client → binary PCM16 mono chunks at the advertised sample_rate
  client → {"type": "stop"}                        (flush the decoders, get the summary)
  server → {"type": "ready", "session_id", "backend", "sample_rate"}
  server → {"type": "partial", "text", "language"}
  server → {"type": "utterance", "text", "language", "confidence", ...meter fields}
  server → {"type": "summary", "session_id", ...post-call summary fields}
  server → {"type": "error", "message"}

Kaldi decoding and classifier scoring are blocking CPU work, so recognition runs in a
worker thread; `AudioQueue` + `iter_queue` bridge the async receive loop to it (dropping
the *oldest* audio on backpressure, draining fully after stop so the last words are
kept). The finished session is parked in the shared ended-stash, so the normal
`POST /api/live/session/{id}/report` consent flow works for microphone calls too.
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import threading
from collections.abc import Callable, Mapping
from typing import Any, Literal

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, ConfigDict, ValidationError

from qorgan.api_live import _STORE, resolve_backend, update_response
from qorgan.asr.capture import AudioQueue, iter_queue
from qorgan.asr.vosk_stream import Partial, recognize_stream
from qorgan.classifier import predict
from qorgan.config import get_config
from qorgan.live.session import advance, initial_session
from qorgan.live.summary import summarize

router = APIRouter(prefix="/api/live", tags=["live"])

# Test seam: injects scripted fake recognizers. None → `recognize_stream` lazily loads
# the real KK+RU vosk models (auto-downloaded to ~/.cache/vosk on first use).
_RECOGNIZER_FACTORY: Callable[[], Mapping[str, Any]] | None = None
# Sentinel the worker thread posts when the recognition stream is fully flushed.
_STREAM_DONE = object()
_MISSING_VOSK_MESSAGE = (
    "microphone mode needs the [live] extra — install with: pip install -e '.[live]'"
)


class StartMessage(BaseModel):
    model_config = ConfigDict(frozen=True)

    type: Literal["start"]
    locale: Literal["ru", "kk"] = "ru"
    backend: str | None = None


def _vosk_available() -> bool:
    return importlib.util.find_spec("vosk") is not None


@router.get("/capabilities")
def capabilities() -> dict[str, bool]:
    """What the live page can offer on this install (the mic chip greys out without vosk)."""
    return {"microphone": _RECOGNIZER_FACTORY is not None or _vosk_available()}


@router.websocket("/ws")
async def microphone_ws(ws: WebSocket) -> None:
    await ws.accept()
    try:
        raw = await ws.receive_text()
    except WebSocketDisconnect:
        return
    try:
        start = StartMessage.model_validate_json(raw)
    except ValidationError as exc:
        await _refuse(ws, f"bad start message: {exc}")
        return

    if _RECOGNIZER_FACTORY is None and not _vosk_available():
        await _refuse(ws, _MISSING_VOSK_MESSAGE)
        return

    try:
        backend = await asyncio.to_thread(resolve_backend, start.backend)
    except (predict.UnknownBackendError, ValueError) as exc:
        await _refuse(ws, str(exc))
        return

    state = initial_session(start.locale, backend=backend)
    session_id = _STORE.create(state, backend=backend, locale=start.locale)
    await _send_safe(
        ws,
        {
            "type": "ready",
            "session_id": session_id,
            "backend": backend,
            "sample_rate": get_config().asr_sample_rate,
        },
    )

    audio = AudioQueue()
    stop = threading.Event()
    events: asyncio.Queue[Any] = asyncio.Queue()
    loop = asyncio.get_running_loop()

    def _post(item: Any) -> None:
        try:
            loop.call_soon_threadsafe(events.put_nowait, item)
        except RuntimeError:  # event loop already gone (client vanished mid-stream)
            pass

    def _recognize() -> None:
        recognizers = _RECOGNIZER_FACTORY() if _RECOGNIZER_FACTORY is not None else None
        try:
            for event in recognize_stream(iter_queue(audio, stop), recognizers=recognizers):
                _post(event)
        except Exception as exc:  # model load/decoding failure — surfaced to the client
            _post(exc)
        finally:
            _post(_STREAM_DONE)

    threading.Thread(target=_recognize, name="qorgan-mic-asr", daemon=True).start()
    receiver = asyncio.create_task(_receive_audio(ws, audio, stop))
    try:
        await _forward_events(ws, events, session_id)
    finally:
        stop.set()
        receiver.cancel()

    entry = _STORE.end(session_id)  # parks the session in the ended stash for /report
    if entry is not None:
        summary = summarize(entry.state)
        await _send_safe(
            ws,
            {
                "type": "summary",
                "session_id": session_id,
                "final_score": summary.final_score,
                "band": summary.band,
                "tactic_names": list(summary.tactic_names),
                "recommended_actions": list(summary.recommended_actions),
                "human_note": summary.human_note,
            },
        )
    await _close_safe(ws)


async def _receive_audio(ws: WebSocket, audio: AudioQueue, stop: threading.Event) -> None:
    """Pump binary frames into the audio queue until stop/disconnect; always sets stop
    so the recognizer thread flushes and terminates."""
    try:
        while True:
            message = await ws.receive()
            if message["type"] == "websocket.disconnect":
                return
            chunk = message.get("bytes")
            if chunk:
                audio.put(chunk)
                continue
            text = message.get("text")
            if text and _is_stop(text):
                return
    finally:
        stop.set()


def _is_stop(text: str) -> bool:
    try:
        return json.loads(text).get("type") == "stop"
    except (json.JSONDecodeError, AttributeError):
        return False


async def _forward_events(ws: WebSocket, events: asyncio.Queue[Any], session_id: str) -> None:
    """Relay recognizer events until the stream flushes; committed utterances advance
    the meter session exactly like the HTTP replay path."""
    while True:
        event = await events.get()
        if event is _STREAM_DONE:
            return
        if isinstance(event, Exception):
            await _send_safe(ws, {"type": "error", "message": f"recognition failed: {event}"})
            return
        if isinstance(event, Partial):
            await _send_safe(
                ws, {"type": "partial", "text": event.text, "language": event.language}
            )
            continue
        entry = _STORE.get(session_id)
        if entry is None:  # evicted under load — stop scoring, summary still possible
            return
        new_state, update = await asyncio.to_thread(advance, entry.state, event)
        _STORE.update(session_id, new_state)
        await _send_safe(
            ws,
            {
                "type": "utterance",
                "text": event.text,
                "language": event.language,
                "confidence": event.confidence,
                **update_response(update).model_dump(),
            },
        )


async def _refuse(ws: WebSocket, message: str) -> None:
    await _send_safe(ws, {"type": "error", "message": message})
    await _close_safe(ws)


async def _send_safe(ws: WebSocket, payload: Mapping[str, Any]) -> None:
    try:
        await ws.send_json(dict(payload))
    except (WebSocketDisconnect, RuntimeError):  # client already gone — deliberate no-op
        pass


async def _close_safe(ws: WebSocket) -> None:
    try:
        await ws.close()
    except (WebSocketDisconnect, RuntimeError):  # already closed by the peer
        pass
