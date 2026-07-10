"""Qorgan L1 demo -- Streamlit app.

Paste or pick a transcript, get a risk score + grounded, localized explanation. Runs with
zero setup: if `QORGAN_CLASSIFIER_BACKEND=llm` (the default) but no `GEMINI_API_KEY`
is configured, the app transparently falls back to the deterministic `mock` backend so
the demo never depends on a live API call.

Day 1 scope: single L1 tab, static risk meter, no RU/KK toggle polish (D4-5), no L2 tab
yet (D5-6).
"""

from __future__ import annotations

import streamlit as st

from qorgan.classifier.predict import score
from qorgan.config import get_config
from qorgan.data.demo_transcripts import DEMO_TRANSCRIPTS
from qorgan.explain.explainer import explain

st.set_page_config(page_title="Qorgan -- scam-call risk", page_icon=":shield:", layout="centered")


def _effective_backend() -> str:
    """Resolve which backend to actually use: falls back to `mock` when the configured
    `llm` backend has no API key, so the app never requires a live call to run."""
    cfg = get_config()
    if cfg.classifier_backend == "llm" and not cfg.gemini_api_key:
        return "mock"
    return cfg.classifier_backend


def _render_risk_meter(risk: float, threshold: float) -> None:
    st.metric("Risk score", f"{risk:.0%}")
    st.progress(min(max(risk, 0.0), 1.0))
    if risk >= threshold:
        st.error("High risk -- likely scam pattern detected.")
    else:
        st.success("Low risk.")


def _render_highlighted_transcript(transcript: str, highlights) -> None:
    if not highlights:
        st.write(transcript)
        return
    pieces: list[str] = []
    cursor = 0
    for span in sorted(highlights, key=lambda s: s.start):
        pieces.append(transcript[cursor : span.start])
        pieces.append(f"**:red[{transcript[span.start : span.end]}]**")
        cursor = span.end
    pieces.append(transcript[cursor:])
    st.markdown("".join(pieces))


def main() -> None:
    st.title("Qorgan -- scam-call risk detector")
    st.caption("Decision-support only. A human always makes the final call.")

    backend = _effective_backend()
    if backend == "mock":
        st.info("No GEMINI_API_KEY configured -- running in offline demo (`mock`) mode.")

    demo_names = list(DEMO_TRANSCRIPTS.keys())
    choice = st.selectbox(
        "Pick a demo transcript (or 'custom' to paste your own)",
        ["custom", *demo_names],
    )
    default_text = "" if choice == "custom" else DEMO_TRANSCRIPTS[choice]
    transcript = st.text_area("Transcript", value=default_text, height=200)

    locale = st.radio("Explanation language / Tusindirme tili", ["ru", "kk"], horizontal=True)

    if st.button("Score call", type="primary") and transcript.strip():
        cfg = get_config()
        result = score(transcript, backend=backend)
        explanation = explain(result, transcript, locale)

        _render_risk_meter(result.risk, cfg.risk_threshold)
        _render_highlighted_transcript(transcript, explanation.highlights)

        if explanation.tags:
            st.write("**Tactic tags:**", ", ".join(tag.id for tag in explanation.tags))

        st.write("**Reason:**", explanation.reason)
        st.caption(explanation.caveat)
        st.caption(explanation.human_note)


if __name__ == "__main__":
    main()
