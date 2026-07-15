"""Live microphone modes for the Live-call tab (design spec §06).

Two capture vectors feed one pipeline:

- **browser** — streamlit-webrtc delivers real-time audio frames from the browser's mic
  permission dialog; a poller thread resamples them to 16 kHz mono PCM16.
- **local** — sounddevice captures the machine's microphone directly (same-machine
  demos only).

Both push PCM into `qorgan.asr.capture.AudioQueue`; a recognizer worker runs
`vosk_stream.recognize_stream` (KK ∥ RU voting) and emits `Partial` /
`CommittedUtterance` events onto a thread-safe queue. Only the `st.fragment` drain loop
running in the script thread touches `st.session_state` and advances the live session —
worker threads never call Streamlit APIs.

All heavy deps (vosk, streamlit-webrtc, av, sounddevice) are optional (`pip install -e
".[live]"`) and imported lazily; `missing_deps()` powers the graceful-degrade hint.
"""

from __future__ import annotations

import importlib.util
import queue
import threading

import streamlit as st

from qorgan.asr.capture import AudioQueue, iter_queue, sounddevice_chunks
from qorgan.asr.stream import CommittedUtterance
from qorgan.asr.vosk_stream import Partial, recognize_stream
from qorgan.config import get_config
from qorgan.live.session import advance, initial_session

_RUNTIME_KEY = "mic_runtime"
_SESSION_KEY = "mic_session"
_LINES_KEY = "mic_lines"
_PARTIAL_KEY = "mic_partial"
_FINAL_STATE_KEY = "live_final_state"  # shared with live_view's post-call flow

_POLL_EVERY = "0.4s"
_SENTINEL = None

_DEPS_BY_VECTOR = {
    "browser": ("vosk", "streamlit_webrtc", "av"),
    "local": ("vosk", "sounddevice"),
}

# Seam for tests: monkeypatched to simulate missing optional dependencies.
_find_spec = importlib.util.find_spec


def missing_deps(vector: str) -> tuple[str, ...]:
    """Names of the optional packages `vector` needs that are not importable."""
    return tuple(name for name in _DEPS_BY_VECTOR[vector] if _find_spec(name) is None)


def render_mic_mode(vector: str, locale: str, backend: str) -> None:
    """Render one microphone capture mode (\"browser\" or \"local\")."""
    missing = missing_deps(vector)
    if missing:
        st.info(
            "Live microphone needs the optional audio stack "
            f"(missing: {', '.join(missing)}). Install it with `pip install -e \".[live]\"`."
        )
        return

    st.caption(
        "Put the call on speakerphone near this device. First start downloads the "
        "KK+RU recognition models (~100 MB) to `~/.cache/vosk`."
        + (" Local capture has no echo cancellation." if vector == "local" else "")
    )
    if vector == "browser":
        _render_browser_controls(locale, backend)
    else:
        _render_local_controls(locale, backend)
    _live_panel()


# --- capture control ---------------------------------------------------------------------


def _render_browser_controls(locale: str, backend: str) -> None:
    from streamlit_webrtc import WebRtcMode, webrtc_streamer

    ctx = webrtc_streamer(
        key="qorgan_mic",
        mode=WebRtcMode.SENDONLY,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={"audio": True, "video": False},
        audio_receiver_size=1024,
        desired_playing_state=None,
    )
    runtime = st.session_state.get(_RUNTIME_KEY)
    if ctx.state.playing and runtime is None and ctx.audio_receiver:
        _start_runtime(locale, backend, lambda stop: _browser_worker(ctx.audio_receiver, stop))
    elif not ctx.state.playing and runtime is not None:
        runtime["stop"].set()  # component STOP clicked -> flush and finalize


def _render_local_controls(locale: str, backend: str) -> None:
    runtime = st.session_state.get(_RUNTIME_KEY)
    if runtime is None:
        if st.button("Start recording", type="primary", key="mic_start"):
            _start_runtime(locale, backend, _local_worker)
            st.rerun()
    elif st.button("Stop recording", key="mic_stop"):
        runtime["stop"].set()


def _start_runtime(locale: str, backend: str, worker_factory) -> None:
    """Spawn the capture/recognition threads and reset the live session state."""
    stop = threading.Event()
    events: queue.Queue = queue.Queue()
    thread = threading.Thread(target=worker_factory(stop), args=(events,), daemon=True)
    st.session_state[_RUNTIME_KEY] = {"stop": stop, "events": events, "thread": thread}
    st.session_state[_SESSION_KEY] = initial_session(locale, backend=backend)
    st.session_state[_LINES_KEY] = []
    st.session_state[_PARTIAL_KEY] = ""
    st.session_state.pop(_FINAL_STATE_KEY, None)
    thread.start()


def _local_worker(stop: threading.Event):
    """Worker body: local mic -> recognizer -> event queue."""

    def run(events: queue.Queue) -> None:
        try:
            for event in recognize_stream(sounddevice_chunks(stop)):
                events.put(event)
        except Exception as exc:  # surfaced by the drain loop; workers must not die silently
            events.put(exc)
        finally:
            events.put(_SENTINEL)

    return run


def _browser_worker(receiver, stop: threading.Event):
    """Worker body: WebRTC frames -> resample -> recognizer -> event queue."""

    def run(events: queue.Queue) -> None:
        import av

        cfg = get_config()
        audio_queue = AudioQueue()
        resampler = av.AudioResampler(format="s16", layout="mono", rate=cfg.asr_sample_rate)

        def poll_frames() -> None:
            while not stop.is_set():
                try:
                    frames = receiver.get_frames(timeout=1)
                except queue.Empty:
                    continue
                for frame in frames:
                    for resampled in resampler.resample(frame):
                        audio_queue.put(resampled.to_ndarray().tobytes())

        poller = threading.Thread(target=poll_frames, daemon=True)
        poller.start()
        try:
            for event in recognize_stream(iter_queue(audio_queue, stop)):
                events.put(event)
        except Exception as exc:
            events.put(exc)
        finally:
            events.put(_SENTINEL)

    return run


# --- live panel (script-thread only) --------------------------------------------------------


@st.fragment(run_every=_POLL_EVERY)
def _live_panel() -> None:
    """Drain worker events and render the live meter; the only session-state writer."""
    runtime = st.session_state.get(_RUNTIME_KEY)
    state = st.session_state.get(_SESSION_KEY)
    if runtime is None or state is None:
        return

    finished = False
    while True:
        try:
            event = runtime["events"].get_nowait()
        except queue.Empty:
            break
        if event is _SENTINEL:
            finished = True
            break
        if isinstance(event, Exception):
            st.session_state[_RUNTIME_KEY] = None
            st.error(f"Live capture failed: {event}")
            return
        if isinstance(event, Partial):
            st.session_state[_PARTIAL_KEY] = event.text
            continue
        state, update = advance(state, event)
        st.session_state[_SESSION_KEY] = state
        st.session_state[_LINES_KEY].append(event.text)
        st.session_state[_PARTIAL_KEY] = ""
        st.session_state["mic_last_update"] = update

    _render_live_state(state)

    if finished:
        st.session_state[_RUNTIME_KEY] = None
        if state.utterances:
            st.session_state[_FINAL_STATE_KEY] = state
        st.rerun(scope="app")  # hand off to the shared post-call summary + report flow


def _render_live_state(state) -> None:
    from app.live_view import _BAND_LABELS, _highlighted
    from qorgan.live.meter import band

    st.progress(state.meter.score / 100.0, text=f"Suspicion {state.meter.score:.0f} / 100")
    st.markdown(f"**{_BAND_LABELS[band(state.meter.score)]}**")
    lines = st.session_state.get(_LINES_KEY, [])
    if lines:
        st.markdown(_highlighted(lines, state.seen_evidence))
    partial = st.session_state.get(_PARTIAL_KEY, "")
    if partial:
        st.caption(f"… {partial}")
    last_update = st.session_state.get("mic_last_update")
    if last_update is not None and state.meter.latched and last_update.recommendation.advices:
        st.error(last_update.recommendation.advices[0])
