"""Transport-agnostic live-audio plumbing (design spec §06).

Both capture vectors — browser WebRTC frames (resampled in the app layer) and the local
microphone — push PCM16 chunks into the same bounded `AudioQueue`; `iter_queue()` turns
it into the chunk iterator `vosk_stream.recognize_stream()` consumes. The queue drops the
*oldest* audio on overflow: a stalled consumer must never block the capture callback, and
for a live meter the newest audio is always the most valuable.

`sounddevice_chunks()` is the local-mic adapter (lazy import — the `[live]` extra).
"""

from __future__ import annotations

import queue
import threading
from collections.abc import Iterator

# ~0.25 s of 16 kHz PCM16 per chunk block; 64 chunks ≈ 16 s of buffered audio headroom.
_DEFAULT_MAX_CHUNKS = 64
_DEFAULT_POLL_TIMEOUT_S = 0.1
_BLOCK_SECONDS = 0.25


class AudioQueue:
    """Bounded, thread-safe FIFO of PCM chunks that drops the oldest on overflow."""

    def __init__(self, max_chunks: int = _DEFAULT_MAX_CHUNKS) -> None:
        self._queue: queue.Queue[bytes] = queue.Queue(maxsize=max_chunks)

    def put(self, chunk: bytes) -> None:
        """Enqueue `chunk`, evicting the oldest buffered chunk if full (never blocks)."""
        while True:
            try:
                self._queue.put_nowait(chunk)
                return
            except queue.Full:
                try:
                    self._queue.get_nowait()
                except queue.Empty:  # racing consumer emptied it -- retry the put
                    continue

    def get(self, *, timeout: float) -> bytes | None:
        """Dequeue one chunk, or `None` if nothing arrives within `timeout` seconds."""
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None


def iter_queue(
    audio_queue: AudioQueue,
    stop: threading.Event,
    *,
    poll_timeout: float = _DEFAULT_POLL_TIMEOUT_S,
) -> Iterator[bytes]:
    """Yield chunks until `stop` is set *and* the queue is drained.

    Buffered audio recorded before the stop click still reaches the recognizer, so the
    last spoken words are not cut off.
    """
    while True:
        chunk = audio_queue.get(timeout=poll_timeout)
        if chunk is not None:
            yield chunk
        elif stop.is_set():
            return


def sounddevice_chunks(stop: threading.Event) -> Iterator[bytes]:  # pragma: no cover
    """Local-microphone chunk source via sounddevice (the `[live]` extra).

    Captures directly at the configured ASR sample rate, mono PCM16 — no resampling
    needed. Only meaningful when the browser and the server are the same machine.
    """
    import sounddevice as sd

    from qorgan.config import get_config

    cfg = get_config()
    audio_queue = AudioQueue()
    block_size = int(cfg.asr_sample_rate * _BLOCK_SECONDS)

    def _on_audio(indata, _frames, _time, _status) -> None:
        audio_queue.put(bytes(indata))

    with sd.RawInputStream(
        samplerate=cfg.asr_sample_rate,
        blocksize=block_size,
        channels=1,
        dtype="int16",
        callback=_on_audio,
    ):
        yield from iter_queue(audio_queue, stop)
