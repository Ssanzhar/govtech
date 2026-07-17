"""TDD tests for `qorgan.live.session_store` — the one mutable boundary for the
live-call API (every `LiveSessionState` it wraps stays immutable)."""

from qorgan.live.meter import initial_state
from qorgan.live.session import LiveSessionState
from qorgan.live.session_store import SessionStore


def _state(locale: str = "ru", backend: str = "mock") -> LiveSessionState:
    return LiveSessionState(locale=locale, backend=backend, meter=initial_state())


def test_create_returns_unique_ids() -> None:
    store = SessionStore()

    first = store.create(_state(), backend="mock", locale="ru")
    second = store.create(_state(), backend="mock", locale="ru")

    assert first != second
    assert isinstance(first, str) and first


def test_get_returns_the_stored_entry() -> None:
    store = SessionStore()
    state = _state()

    session_id = store.create(state, backend="mock", locale="ru")
    entry = store.get(session_id)

    assert entry is not None
    assert entry.state == state
    assert entry.backend == "mock"
    assert entry.locale == "ru"


def test_get_unknown_id_returns_none() -> None:
    store = SessionStore()

    assert store.get("does-not-exist") is None


def test_update_replaces_state_and_keeps_backend_and_locale() -> None:
    store = SessionStore()
    session_id = store.create(_state(), backend="mock", locale="kk")
    new_state = _state(locale="kk")

    store.update(session_id, new_state)
    entry = store.get(session_id)

    assert entry is not None
    assert entry.state == new_state
    assert entry.backend == "mock"
    assert entry.locale == "kk"


def test_update_unknown_id_is_a_noop() -> None:
    store = SessionStore()

    store.update("does-not-exist", _state())  # must not raise

    assert store.get("does-not-exist") is None


def test_end_pops_the_session() -> None:
    store = SessionStore()
    session_id = store.create(_state(), backend="mock", locale="ru")

    entry = store.end(session_id)

    assert entry is not None
    assert store.get(session_id) is None


def test_end_twice_returns_none_the_second_time() -> None:
    store = SessionStore()
    session_id = store.create(_state(), backend="mock", locale="ru")
    store.end(session_id)

    assert store.end(session_id) is None


def test_oldest_session_evicted_when_over_capacity() -> None:
    store = SessionStore(max_sessions=2)
    first = store.create(_state(), backend="mock", locale="ru")
    second = store.create(_state(), backend="mock", locale="ru")
    third = store.create(_state(), backend="mock", locale="ru")

    assert store.get(first) is None
    assert store.get(second) is not None
    assert store.get(third) is not None


def test_updating_a_session_marks_it_as_recently_used_for_eviction() -> None:
    store = SessionStore(max_sessions=2)
    first = store.create(_state(), backend="mock", locale="ru")
    second = store.create(_state(), backend="mock", locale="ru")
    store.update(first, _state())  # touch `first` so it is no longer the oldest

    store.create(_state(), backend="mock", locale="ru")  # pushes the store over capacity

    assert store.get(first) is not None
    assert store.get(second) is None


# --- ended stash (post-call reporting) ---------------------------------------------------------


def test_end_stashes_the_entry_for_reporting() -> None:
    store = SessionStore()
    session_id = store.create(_state(), backend="mock", locale="ru")

    ended = store.end(session_id)
    taken = store.take_ended(session_id)

    assert ended is not None
    assert taken is not None
    assert taken.state == ended.state
    assert taken.backend == "mock"


def test_take_ended_consumes_the_stash() -> None:
    store = SessionStore()
    session_id = store.create(_state(), backend="mock", locale="ru")
    store.end(session_id)

    assert store.take_ended(session_id) is not None
    assert store.take_ended(session_id) is None  # one report per session


def test_take_ended_unknown_id_returns_none() -> None:
    store = SessionStore()

    assert store.take_ended("does-not-exist") is None


def test_take_ended_does_not_see_live_sessions() -> None:
    store = SessionStore()
    session_id = store.create(_state(), backend="mock", locale="ru")

    assert store.take_ended(session_id) is None
    assert store.get(session_id) is not None  # still live, untouched


def test_ended_stash_is_capped_oldest_first() -> None:
    store = SessionStore(max_sessions=2)
    first = store.create(_state(), backend="mock", locale="ru")
    second = store.create(_state(), backend="mock", locale="ru")
    store.end(first)
    store.end(second)
    third = store.create(_state(), backend="mock", locale="ru")
    store.end(third)  # pushes the ended stash over its (same) capacity

    assert store.take_ended(first) is None
    assert store.take_ended(second) is not None
    assert store.take_ended(third) is not None
