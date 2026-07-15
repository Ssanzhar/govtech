"""Qorgan demo -- Streamlit app (both levels).

Level 1 (centerpiece): paste/pick a transcript -> risk score + grounded, localized (RU/KK)
explanation. Level 2 (analyst): scam organizations clustered from ~500 incidents by phone
number, ranked by priority, with a novelty-flagged new scheme. Runs zero-setup: the L1
backend degrades to `mock` without a model/key; the L2 tab reads a precomputed analysis.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# PyArrow (pulled in lazily by st.dataframe) bundles the mimalloc allocator, which segfaults
# on macOS/ARM when its first allocation runs on a Streamlit ScriptRunner thread rather than
# the main thread, with torch's libomp already resident (mi_thread_init dereferences an
# uninitialised main heap). Force Arrow onto the system allocator before it is ever imported.
# Must be set before `import streamlit` (Streamlit imports pyarrow on first st.dataframe).
os.environ.setdefault("ARROW_DEFAULT_MEMORY_POOL", "system")

import streamlit as st

# `streamlit run app/streamlit_app.py` puts `app/` (the script dir) on sys.path but not
# the repo root, so the `app` package itself isn't importable; pytest/AppTest does the
# opposite. Pin the repo root so `from app.live_view import ...` works in both harnesses.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from app.live_view import render_live_tab
from qorgan.classifier.predict import score
from qorgan.config import get_config
from qorgan.data.demo_transcripts import DEMO_TRANSCRIPTS
from qorgan.data.schema import UTTERANCE_JOIN
from qorgan.explain.explainer import explain
from qorgan.explain.windowing import apply_hysteresis, windows
from qorgan.taxonomy import get_taxonomy

st.set_page_config(page_title="Qorgan -- scam-call risk", page_icon=":shield:", layout="centered")


# --- Level 1: call check -----------------------------------------------------------------


def _effective_backend() -> str:
    """Resolve the backend, degrading to `mock` when the model/key isn't available."""
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
    if not tags:
        return
    taxonomy = get_taxonomy()
    chips = []
    for tag in tags:
        try:
            name = taxonomy.display_name(tag.id, locale)
        except KeyError:  # unknown/legacy tag id -> fall back to raw id; other errors surface
            name = tag.id
        chips.append(f"`{name}` ({tag.id}, {tag.weight:.0%})")
    st.markdown("**Tactic tags:** " + " · ".join(chips))


def _render_buildup(transcript: str, backend: str) -> None:
    utterances = [line for line in transcript.split(UTTERANCE_JOIN) if line.strip()]
    if len(utterances) < 2:
        st.caption("Risk build-up needs a multi-line transcript (one utterance per line).")
        return
    cfg = get_config()
    risks = [score(w.text, backend=backend).risk for w in windows(utterances)]
    alert_state = apply_hysteresis(risks, enter=cfg.risk_threshold_enter, exit=cfg.risk_threshold_exit)
    st.line_chart({"risk": risks})
    fired = next((i + 1 for i, on in enumerate(alert_state) if on), None)
    st.caption(
        f"Alert fires from turn {fired} (hysteresis {cfg.risk_threshold_enter:.0%}/{cfg.risk_threshold_exit:.0%})."
        if fired is not None else "Alert never fires across the call."
    )


def _render_level1() -> None:
    backend = _effective_backend()
    if backend == "mock":
        st.info("Running in offline demo (`mock`) mode -- no trained model or API key required.")

    demo_names = list(DEMO_TRANSCRIPTS.keys())
    choice = st.selectbox("Pick a demo transcript (or 'custom' to paste your own)", ["custom", *demo_names])
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


# --- Level 2: analyst view ---------------------------------------------------------------


def _render_level2() -> None:
    from qorgan.analytics.pipeline import load_organizations_jsonl
    from qorgan.data.incident_seed import load_incidents_jsonl

    cfg = get_config()
    orgs_path = cfg.data_dir / "processed" / "organizations.jsonl"
    incidents_path = cfg.data_dir / "processed" / "incidents.jsonl"
    if not orgs_path.exists():
        st.info(
            "No Level-2 analysis yet. Run `python scripts/demo_seed.py` then "
            "`python -m qorgan.analytics.pipeline`."
        )
        return

    organizations = load_organizations_jsonl(orgs_path)
    if not organizations:
        st.info(
            "No organizations to display yet. The Level-2 analysis file is empty -- run "
            "`python scripts/demo_seed.py` then `python -m qorgan.analytics.pipeline`."
        )
        return

    incidents = {i.id: i for i in load_incidents_jsonl(incidents_path)} if incidents_path.exists() else {}

    novel = [o for o in organizations if o.is_novel]
    if novel:
        st.warning(
            f"NEW SCHEME detected: **{novel[0].id}** ({len(novel[0].members)} incidents) -- "
            f"{(novel[0].representative_script or '')[:90]}..."
        )

    st.subheader("Scam organizations -- priority queue")
    st.dataframe(
        [
            {
                "Organization": org.id,
                "Incidents": len(org.members),
                "Numbers": ", ".join(org.numbers) or "-",
                "Priority": round(org.priority, 2),
                "New scheme": "NEW" if org.is_novel else "",
            }
            for org in organizations
        ],
        width="stretch",
    )

    selected = st.selectbox("Drill into organization", [o.id for o in organizations])
    org = next(o for o in organizations if o.id == selected)
    st.write(
        f"**Linked numbers:** {', '.join(org.numbers) or '-'} · **Incidents:** {len(org.members)} "
        f"· **New scheme:** {'yes' if org.is_novel else 'no'} · **Priority:** {org.priority:.2f}"
    )
    st.write("**Representative script:**", org.representative_script)
    st.caption("Sample incidents:")
    for member_id in org.members[:6]:
        incident = incidents.get(member_id)
        if incident is not None:
            stamp = incident.timestamp.strftime("%Y-%m-%d %H:%M") if incident.timestamp else "-"
            st.caption(f"{stamp} · {incident.phone_number} · {incident.transcript[:110]}")


def main() -> None:
    st.title("Qorgan -- scam-call risk detector")
    st.caption("Decision-support only. A human always makes the final call.")
    level1, live, level2 = st.tabs(
        ["Level 1 -- Call check", "Live call", "Level 2 -- Analyst view"]
    )
    with level1:
        _render_level1()
    with live:
        render_live_tab(_effective_backend())
    with level2:
        _render_level2()


if __name__ == "__main__":
    main()
