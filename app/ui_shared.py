"""Streamlit rendering helpers shared across tabs (L1 call-check + analyst view)."""

from __future__ import annotations

import streamlit as st


def render_highlighted_transcript(transcript: str, highlights) -> None:
    """Render `transcript` with attributed spans bolded in red (grounded highlights only)."""
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
