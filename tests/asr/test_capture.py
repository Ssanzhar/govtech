"""Tests for `qorgan.asr.capture` — transport-agnostic audio plumbing (design spec §06).

The queue and iterator are the deterministic core shared by both capture vectors
(browser WebRTC and local sounddevice); the device adapters themselves are thin,
lazily-imported shims that never load in tests.
"""

import threading

from qorgan.asr.capture import AudioQueue, iter_queue


def test_queue_preserves_fifo_order():
    queue = AudioQueue(max_chunks=4)
    queue.put(b"a")
    queue.put(b"b")

    assert queue.get(timeout=0.01) == b"a"
    assert queue.get(timeout=0.01) == b"b"


def test_queue_drops_oldest_when_full():
    queue = AudioQueue(max_chunks=2)
    queue.put(b"a")
    queue.put(b"b")
    queue.put(b"c")  # overflows: 'a' is dropped, live audio must never block the producer

    assert queue.get(timeout=0.01) == b"b"
    assert queue.get(timeout=0.01) == b"c"


def test_get_returns_none_on_timeout():
    queue = AudioQueue(max_chunks=2)

    assert queue.get(timeout=0.01) is None


def test_iter_queue_yields_until_stopped():
    queue = AudioQueue(max_chunks=8)
    stop = threading.Event()
    queue.put(b"a")
    queue.put(b"b")

    chunks = []
    for chunk in iter_queue(queue, stop, poll_timeout=0.01):
        chunks.append(chunk)
        if len(chunks) == 2:
            stop.set()

    assert chunks == [b"a", b"b"]


def test_iter_queue_drains_remaining_chunks_after_stop():
    queue = AudioQueue(max_chunks=8)
    stop = threading.Event()
    queue.put(b"a")
    queue.put(b"b")
    stop.set()  # stop first -- whatever is already buffered must still come through

    assert list(iter_queue(queue, stop, poll_timeout=0.01)) == [b"a", b"b"]


def test_iter_queue_stops_promptly_when_empty_and_stopped():
    queue = AudioQueue(max_chunks=8)
    stop = threading.Event()
    stop.set()

    assert list(iter_queue(queue, stop, poll_timeout=0.01)) == []
