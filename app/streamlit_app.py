"""Qorgan L1 demo -- Streamlit app.

Paste or pick a transcript, get a risk score + grounded, localized (RU/KK) explanation:
highlighted trigger phrases, localized tactic chips, a calibrated/labeled confidence band,
a "where this can be wrong" caveat, and a human-decides note. Optionally replay the call
turn-by-turn to watch the risk meter climb (with alert hysteresis).

Runs with zero setup: if `QORGAN_CLASSIFIER_BACKEND=llm` (the default) but no
`GEMINI_API_KEY` is configured, the app transparently falls back to the deterministic
`mock` backend so the demo never depends on a live API call.
"""

from __future__ import annotations

import streamlit as st

from qorgan.classifier.predict import score
from qorgan.config import get_config
from qorgan.data.demo_transcripts import DEMO_TRANSCRIPTS
from qorgan.data.schema import UTTERANCE_JOIN
from qorgan.explain.explainer import explain
from qorgan.explain.windowing import apply_hysteresis, windows
from qorgan.taxonomy import get_taxonomy

st.set_page_config(page_title="Qorgan -- scam-call risk", page_icon=":shield:", layout="centered")


def _effective_backend() -> str:
    """Resolve which backend to actually use, degrading to `mock` when the configured
    backend's model/key isn't available -- so the app runs zero-setup on a fresh clone.

    - `llm` with no API key -> `mock`
    - `linear`/`xlmr` with no exported model -> `mock`
    """
    cfg = get_config()
    if cfg.classifier_backend == "llm" and not cfg.gemini_api_key:
        return "mock"
    if cfg.classifier_backend == "linear" and not (cfg.linear_model_dir / "metadata.json").exists():
        return "mock"
    if cfg.classifier_backend == "xlmr" and not (cfg.xlmr_model_dir / "metadata.json").exists():
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


def _render_tags(tags, locale: str) -> None:
    """Localized tactic chips -- display name plus the raw id (analyst transparency)."""
    if not tags:
        return
    taxonomy = get_taxonomy()
    chips = []
    for tag in tags:
        try:
            name = taxonomy.display_name(tag.id, locale)
        except Exception:  # noqa: BLE001 - unknown/legacy id must not crash the UI
            name = tag.id
        chips.append(f"`{name}` ({tag.id}, {tag.weight:.0%})")
    st.markdown("**Tactic tags:** " + " · ".join(chips))


def _render_buildup(transcript: str, backend: str) -> None:
    """Replay the call turn-by-turn: cumulative risk + the turn the alert fires (hysteresis)."""
    utterances = [line for line in transcript.split(UTTERANCE_JOIN) if line.strip()]
    if len(utterances) < 2:
        st.caption("Risk build-up needs a multi-line transcript (one utterance per line).")
        return
    cfg = get_config()
    risks = [score(w.text, backend=backend).risk for w in windows(utterances)]
    alert_state = apply_hysteresis(
        risks, enter=cfg.risk_threshold_enter, exit=cfg.risk_threshold_exit
    )
    st.line_chart({"risk": risks})
    fired_turn = next((i + 1 for i, on in enumerate(alert_state) if on), None)
    if fired_turn is not None:
        st.caption(f"Alert fires from turn {fired_turn} (enter={cfg.risk_threshold_enter:.0%}, "
                   f"exit={cfg.risk_threshold_exit:.0%} hysteresis).")
    else:
        st.caption("Alert never fires across the call.")


def main() -> None:
    st.title("Qorgan -- scam-call risk detector")
    st.caption("Decision-support only. A human always makes the final call.")

    backend = _effective_backend()
    if backend == "mock":
        st.info("Running in offline demo (`mock`) mode -- no trained model or API key required.")

    demo_names = list(DEMO_TRANSCRIPTS.keys())
    choice = st.selectbox(
        "Pick a demo transcript (or 'custom' to paste your own)",
        ["custom", *demo_names],
    )
    default_text = "" if choice == "custom" else DEMO_TRANSCRIPTS[choice]
    transcript = st.text_area("Transcript", value=default_text, height=200)

    locale = st.radio("Explanation language / Tusindirme tili", ["ru", "kk"], horizontal=True)
    show_buildup = st.checkbox("Show risk build-up (re-scores each turn)")

    if st.button("Score call", type="primary") and transcript.strip():
        cfg = get_config()
        result = score(transcript, backend=backend)
        explanation = explain(result, transcript, locale)

        _render_risk_meter(result.risk, cfg.risk_threshold)
        _render_highlighted_transcript(transcript, explanation.highlights)
        _render_tags(explanation.tags, locale)

        st.write("**Reason:**", explanation.reason)
        if explanation.confidence_label:
            st.caption(explanation.confidence_label)
        st.caption(explanation.caveat)
        st.caption(explanation.human_note)

        if show_buildup:
            st.divider()
            st.subheader("Risk build-up")
            _render_buildup(transcript, backend)


if __name__ == "__main__":
    main()
