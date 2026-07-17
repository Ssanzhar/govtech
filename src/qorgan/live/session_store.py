"""In-memory session store for the live-call HTTP API — the one deliberately mutable
boundary in the live pipeline. Every `LiveSessionState` it wraps stays immutable
(`qorgan.live.session.advance` always returns a brand-new state); this store just tracks
"which id points at which state right now" for a running demo server.
"""

from __future__ import annotations

import uuid
from collections import OrderedDict
from dataclasses import dataclass

from qorgan.live.session import LiveSessionState

# Caps memory for a long-running demo server; the oldest (least-recently-touched)
# session is evicted first once the store is over capacity.
_MAX_SESSIONS = 200


@dataclass(frozen=True)
class SessionEntry:
    """One live session: its current immutable state plus the backend/locale fixed
    at creation (so every turn scores with the same resolved backend)."""

    state: LiveSessionState
    backend: str
    locale: str


class SessionStore:
    """`dict[str, SessionEntry]` with oldest-eviction above `max_sessions`.

    Not thread-safe by design — the demo server runs single-worker `uvicorn`. `create`
    and `update` both mark a session as most-recently-used for eviction purposes.
    """

    def __init__(self, *, max_sessions: int = _MAX_SESSIONS) -> None:
        self._max_sessions = max_sessions
        self._entries: OrderedDict[str, SessionEntry] = OrderedDict()
        # Ended sessions parked for the consent-gated report flow (one report per
        # session: `take_ended` consumes). Same capacity policy as the live map.
        self._ended: OrderedDict[str, SessionEntry] = OrderedDict()

    def create(self, state: LiveSessionState, *, backend: str, locale: str) -> str:
        """Store a fresh session and return its new id."""
        session_id = uuid.uuid4().hex
        self._entries[session_id] = SessionEntry(state=state, backend=backend, locale=locale)
        self._entries.move_to_end(session_id)
        while len(self._entries) > self._max_sessions:
            self._entries.popitem(last=False)
        return session_id

    def get(self, session_id: str) -> SessionEntry | None:
        """Return the entry for `session_id`, or `None` if it is unknown/expired."""
        return self._entries.get(session_id)

    def update(self, session_id: str, state: LiveSessionState) -> None:
        """Replace the stored state for `session_id`; a no-op for an unknown id."""
        entry = self._entries.get(session_id)
        if entry is None:
            return
        self._entries[session_id] = SessionEntry(
            state=state, backend=entry.backend, locale=entry.locale
        )
        self._entries.move_to_end(session_id)

    def end(self, session_id: str) -> SessionEntry | None:
        """Pop and return the entry for `session_id`, or `None` if already gone.

        The entry is parked in the ended stash so the post-call report flow can still
        reach the finished session's state via `take_ended`.
        """
        entry = self._entries.pop(session_id, None)
        if entry is not None:
            self._ended[session_id] = entry
            while len(self._ended) > self._max_sessions:
                self._ended.popitem(last=False)
        return entry

    def take_ended(self, session_id: str) -> SessionEntry | None:
        """Consume and return an *ended* session's entry, or `None` if unknown.

        Consuming enforces one report per session; live sessions are never visible here.
        """
        return self._ended.pop(session_id, None)
