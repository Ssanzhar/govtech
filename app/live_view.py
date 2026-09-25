"""Live-call tab (design spec §03): a replayed call drives the climbing suspicion meter,
grounded evidence highlights, and tactic-specific recommendations, ending in a post-call
summary and a consent-gated, editable report flow.

The replay is the web prototype's honest stand-in for streaming capture: one committed
utterance per line, advanced through the same `qorgan.live` pipeline a mobile client
would use.
"""

from __future__ import annotations

import re
import time

import streamlit as st

from qorgan.asr.stream import replay_transcript
from qorgan.data.demo_transcripts import LIVE_DEMO_CALLS
from qorgan.live.meter import Band
from qorgan.live.session import LiveSessionState, advance, initial_session
from qorgan.config import get_config
from qorgan.privacy.numbers import MissingHmacKeyError
from qorgan.live.summary import ReportDraft, build_report, submit_report, summarize

# Pause between replayed turns — long enough to watch the meter move, short enough for
# the smoke tests to stay fast.
_REPLAY_DELAY_S = 0.35
_STATE_KEY = "live_final_state"

_BAND_LABELS: dict[Band, str] = {
    "low": ":green[LOW] — no obvious scam indicators",
    "medium": ":orange[MEDIUM] — caution, warning signs detected",
    "high": ":orange[HIGH] — strong warning",
    "critical": ":red[CRITICAL] — immediate warning",
}


def render_live_tab(backend: str) -> None:
    """The whole Live-call tab: input mode, live analysis, post-call summary, report."""
    st.caption(
        "One pipeline behind every input: streaming utterances -> rolling window -> risk -> "
        "suspicion meter (0-100, hysteresis) -> grounded advice."
    )
    locale = st.radio(
        "Advice language / Kenes tili", ["ru", "kk"], horizontal=True, key="live_locale"
    )
    mode = st.radio(
        "Input",
        ["Replay script", "Microphone — browser", "Microphone — local"],
        horizontal=True,
        key="live_mode",
        help="Replay needs no extra setup; the microphone modes need `pip install -e \".[live]\"`.",
    )
    if mode == "Replay script":
        _render_replay_mode(locale, backend)
    else:
        from app.mic_live import render_mic_mode

        render_mic_mode("browser" if "browser" in mode else "local", locale, backend)

    final_state = st.session_state.get(_STATE_KEY)
    if final_state is not None:
        _render_post_call(final_state)


def _render_replay_mode(locale: str, backend: str) -> None:
    scenario = st.selectbox(
        "Live call scenario (or 'custom' to paste your own)",
        ["custom", *LIVE_DEMO_CALLS],
        key="live_choice",
    )
    # Deliberately unkeyed (like the L1 text area): a keyed text_area pins its first
    # default forever, so switching scenarios would keep showing the old script.
    default_script = LIVE_DEMO_CALLS.get(scenario, "")
    script = st.text_area(
        "Call script — one utterance per line", value=default_script, height=160
    )

    if st.button("Start live analysis", type="primary", key="live_start") and script.strip():
        _replay(_as_turns(script), locale, backend)


def _as_turns(script: str) -> str:
    """Normalize pasted text into one-utterance-per-line form.

    Multi-line input is taken as-is; a single-paragraph transcript is split at sentence
    boundaries so the replay still shows a turn-by-turn build-up.
    """
    lines = [line.strip() for line in script.splitlines() if line.strip()]
    if len(lines) > 1:
        return "\n".join(lines)
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", lines[0]) if s.strip()]
    return "\n".join(sentences)


def _replay(script: str, locale: str, backend: str) -> None:
    """Advance the live session one utterance at a time, updating the meter in place."""
    state = initial_session(locale, backend=backend)
    meter_slot = st.empty()
    band_slot = st.empty()
    transcript_slot = st.empty()
    warning_slot = st.empty()

    shown_lines: list[str] = []
    for utterance in replay_transcript(script):
        state, update = advance(state, utterance)
        shown_lines.append(utterance.text)

        meter_slot.progress(
            update.meter.score / 100.0, text=f"Suspicion {update.meter.score:.0f} / 100"
        )
        band_slot.markdown(f"**{_BAND_LABELS[update.band]}**")
        transcript_slot.markdown(_highlighted(shown_lines, state.seen_evidence))
        if update.meter.latched and update.recommendation.advices:
            warning_slot.error(update.recommendation.advices[0])
        time.sleep(_REPLAY_DELAY_S)

    st.session_state[_STATE_KEY] = state


def _highlighted(lines: list[str], evidence: tuple[str, ...]) -> str:
    """Render the transcript so far, highlighting attributed evidence phrases verbatim."""
    rendered: list[str] = []
    for line in lines:
        for phrase in evidence:
            if phrase in line:
                line = line.replace(phrase, f"**:red[{phrase}]**", 1)
        rendered.append(line)
    return "  \n".join(rendered)


def _render_post_call(state: LiveSessionState) -> None:
    """Structured post-call summary (§07) + the consent-gated report flow (§11)."""
    summary = summarize(state)

    st.divider()
    st.subheader("Post-call summary")
    st.metric("Final suspicion score", f"{summary.final_score:.0f} / 100")
    st.markdown(f"**{_BAND_LABELS[summary.band]}**")
    if summary.tactic_names:
        st.markdown("**Detected tactics:** " + " · ".join(f"`{n}`" for n in summary.tactic_names))
    if summary.recommended_actions:
        st.markdown("**Recommended actions:**")
        for action in summary.recommended_actions:
            st.markdown(f"- {action}")
    if state.seen_evidence:
        st.markdown("**Evidence:** " + " ".join(f"«{p}»" for p in state.seen_evidence))
    st.caption(summary.human_note)

    st.subheader("Report this call")
    st.caption("Never automatic: you review and edit every field before anything is sent.")
    if not st.checkbox("I consent to reporting this call", key="live_consent"):
        return

    phone = st.text_input("Caller number (optional)", key="live_phone")
    edited_transcript = st.text_area(
        "Transcript (editable)", value=state.transcript(), key="live_report_transcript"
    )
    edited_phrases = st.text_area(
        "Flagged phrases — one per line (editable)",
        value="\n".join(state.seen_evidence),
        key="live_report_phrases",
    )
    if st.button("Submit report", key="live_submit"):
        base = build_report(state, phone_number=phone.strip() or None)
        try:
            # Re-validate the user's edits by constructing a fresh draft (model_copy
            # would skip validation; a blank transcript must fail loudly here).
            draft = ReportDraft(
                **{
                    **base.model_dump(),
                    "transcript": edited_transcript,
                    "flagged_phrases": tuple(
                        line.strip() for line in edited_phrases.splitlines() if line.strip()
                    ),
                }
            )
            stored = submit_report(draft, hmac_key=get_config().number_hmac_key)
        except (ValueError, MissingHmacKeyError) as exc:  # blank transcript, bad number, no key
            st.error(f"Could not submit the report: {exc}")
            return
        st.success(
            f"Report stored (receipt `{stored.receipt_id}`; number kept only as `{stored.number_prefix or '—'}`, "
            "transcript PII-scrubbed). Open the **Level 2 — Analyst view** tab and click "
            "*Ingest into analysis* to see it in the intelligence picture."
        )
