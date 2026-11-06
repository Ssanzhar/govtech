"""Minimal in-process sliding-window rate limiter for the write endpoints.

Good enough for a single-worker demo server (the same constraint that keeps live sessions
in-process). Keyed by client address; injectable clock for tests. Not a substitute for an
edge rate limiter in production.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import Callable
from typing import Any

from qorgan.config import get_config


def client_key(request: Any, *, hops: int | None = None) -> str:
    """The client's address, for per-client limits. Behind `hops` trusted proxies
    (`QORGAN_TRUSTED_PROXY_HOPS`) it is that many entries from the right of `X-Forwarded-For`: each proxy
    appends what it saw, so those entries cannot be forged, and anything further left is whatever the
    client sent. Without a declared proxy the header means nothing."""
    hops = get_config().trusted_proxy_hops if hops is None else hops
    forwarded = request.headers.get("x-forwarded-for", "") if hops > 0 else ""
    entries = [entry.strip() for entry in forwarded.split(",") if entry.strip()]
    if entries:
        return entries[-min(hops, len(entries))]
    return request.client.host if request.client else "unknown"


# Clients remembered before the idle ones are dropped; each costs a small deque.
_MAX_KEYS = 10_000


class SlidingWindowLimiter:
    def __init__(self, *, max_requests: int, window_seconds: float, clock: Callable[[], float] = time.monotonic, max_keys: int = _MAX_KEYS) -> None:
        if max_requests <= 0 or window_seconds <= 0:
            raise ValueError("max_requests and window_seconds must be > 0")
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._clock = clock
        self._max_keys = max_keys
        self._hits: dict[str, deque[float]] = {}
        self._lock = threading.Lock()

    def allow(self, key: str) -> bool:
        """Record a hit for `key` and return whether it is within the window budget."""
        now = self._clock()
        with self._lock:
            if key not in self._hits and len(self._hits) >= self._max_keys:
                self._forget_idle(now)
            hits = self._hits.setdefault(key, deque())
            while hits and now - hits[0] >= self.window_seconds:
                hits.popleft()
            if len(hits) >= self.max_requests:
                return False
            hits.append(now)
            return True

    def _forget_idle(self, now: float) -> None:
        """Every client whose window has passed: one new key per request must not grow memory for ever."""
        self._hits = {key: hits for key, hits in self._hits.items() if hits and now - hits[-1] < self.window_seconds}

    def reset(self) -> None:
        with self._lock:
            self._hits.clear()
