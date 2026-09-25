"""Analyst dashboard (Level 2): KPI row, new-scheme callout, priority queue, drill-down.

Reads the precomputed analysis (`organizations.jsonl` + `incidents.jsonl`) and renders it
through the pure helpers in `qorgan.analytics.presentation` — organizations appear under
tactic-derived display names ("Выдаёт себя за банк + Просьба назвать код…"), never raw
cluster ids. Degrades to run instructions when no analysis exists yet.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from app.ui_shared import render_highlighted_transcript
from qorgan.analytics.intake import ingest_pending, pending_reports
from qorgan.analytics.pipeline import EMBEDDINGS_FILENAME, load_organizations_jsonl
from qorgan.analytics.presentation import (
    dashboard_kpis,
    last_activity,
    org_activity_by_day,
    org_display_name,
    org_tactic_profile,
    short_tactic_name,
)
from qorgan.config import get_config
from qorgan.data.incident_seed import load_incidents_jsonl

_MAX_MEMBER_ROWS = 8
_QUOTE_CHARS = 100
_EXCERPT_CHARS = 90


def render_analyst_tab() -> None:
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

    incidents = load_incidents_jsonl(incidents_path) if incidents_path.exists() else []
    by_id = {incident.id: incident for incident in incidents}
    locale = cfg.default_locale
    names = {org.id: org_display_name(org, by_id, locale=locale) for org in organizations}

    reports_path = cfg.data_dir / "processed" / "citizen_reports.jsonl"
    pending = pending_reports(reports_path, incidents)

    _render_kpis(organizations, incidents, len(pending))
    _render_ingest_result(names)
    _render_ingest_banner(pending, reports_path, incidents_path, orgs_path)
    _render_novel_callout(organizations, names)
    _render_priority_queue(organizations, names, by_id)
    _render_drilldown(organizations, names, by_id, locale)


def _render_kpis(organizations, incidents, pending_count: int) -> None:
    kpis = dashboard_kpis(organizations, incidents, pending_reports=pending_count)
    cols = st.columns(4)
    cols[0].metric("Incidents analyzed", kpis.incidents_total)
    cols[1].metric("Organizations", kpis.organizations_total)
    cols[2].metric("New schemes", kpis.novel_schemes)
    cols[3].metric("Pending citizen reports", kpis.pending_reports)


def _render_ingest_banner(pending, reports_path, incidents_path, orgs_path) -> None:
    if not pending:
        return
    st.info(f"**{len(pending)}** new citizen report(s) awaiting analysis.")
    if st.button("Ingest into analysis", type="primary", key="l2_ingest"):
        with st.spinner("Embedding new reports and refreshing the analysis…"):
            summary = ingest_pending(
                reports_path=reports_path,
                incidents_path=incidents_path,
                organizations_path=orgs_path,
                embeddings_path=orgs_path.with_name(EMBEDDINGS_FILENAME),
            )
        st.session_state["l2_ingest_summary"] = summary
        st.rerun()


def _render_ingest_result(names) -> None:
    """One-shot success banner after an ingest rerun, with human org names."""
    summary = st.session_state.pop("l2_ingest_summary", None)
    if summary is None or summary.ingested == 0:
        return
    lines = [f"Ingested **{summary.ingested}** citizen report(s):"]
    for placement in summary.placements:
        org_label = names.get(placement.org_id, placement.org_id)
        badge = " · 🆕 possible new scheme" if placement.org_is_novel else ""
        lines.append(f"- `{placement.incident_id}` → **{org_label}**{badge}")
    st.success("\n".join(lines))


def _render_novel_callout(organizations, names) -> None:
    novel = [org for org in organizations if org.is_novel]
    if not novel:
        return
    lines = []
    for org in novel:
        quote = (org.representative_script or "")[:_QUOTE_CHARS]
        lines.append(
            f"**NEW SCHEME:** {names[org.id]} ({len(org.members)} incidents) — «{quote}…»"
        )
    st.warning("\n\n".join(lines))


def _render_priority_queue(organizations, names, by_id) -> None:
    st.subheader("Priority queue")
    rows = []
    for org in organizations:
        latest = last_activity(org, by_id)
        rows.append(
            {
                "Organization": names[org.id],
                "Priority": org.priority,
                "Incidents": len(org.members),
                "Numbers": len(org.numbers),
                "Last activity": latest.date().isoformat() if latest else "—",
                "New": "🆕" if org.is_novel else "",
            }
        )
    st.dataframe(
        rows,
        width="stretch",
        hide_index=True,
        column_config={
            "Priority": st.column_config.ProgressColumn(
                "Priority", min_value=0.0, max_value=1.0, format="%.2f"
            ),
        },
    )


def _render_drilldown(organizations, names, by_id, locale: str) -> None:
    selected = st.selectbox(
        "Drill into organization",
        [org.id for org in organizations],
        format_func=lambda org_id: names[org_id],
        key="l2_org",
    )
    org = next(o for o in organizations if o.id == selected)

    with st.container(border=True):
        badge = " · 🆕 possible new scheme" if org.is_novel else ""
        st.markdown(
            f"#### {names[org.id]}\n"
            f"Priority **{org.priority:.2f}** · **{len(org.members)}** incidents{badge}"
        )
        if org.numbers:
            st.markdown("**Linked numbers:** " + " ".join(f"`{n}`" for n in org.numbers))

        left, right = st.columns(2)
        profile = org_tactic_profile(org, by_id)
        if profile:
            labels = {
                tactic_id: short_tactic_name(tactic_id, locale=locale) or tactic_id
                for tactic_id in profile
            }
            left.caption("Tactic profile")
            left.bar_chart(
                pd.DataFrame(
                    {"count": list(profile.values())},
                    index=[labels[t] for t in profile],
                ),
                horizontal=True,
            )
        activity = org_activity_by_day(org, by_id)
        if activity:
            right.caption("Activity by day")
            right.bar_chart(
                pd.DataFrame({"incidents": list(activity.values())}, index=list(activity))
            )

        if org.representative_script:
            with st.expander("Representative script"):
                _render_representative_script(org, by_id)

        st.caption("Sample incidents")
        st.dataframe(
            [
                {
                    "Date": incident.timestamp.strftime("%Y-%m-%d %H:%M")
                    if incident.timestamp
                    else "—",
                    "Number": incident.number_prefix or "—",
                    "Risk": f"{incident.label.risk:.0%}",
                    "Excerpt": incident.transcript[:_EXCERPT_CHARS],
                }
                for incident in (by_id[m] for m in org.members[:_MAX_MEMBER_ROWS] if m in by_id)
            ],
            width="stretch",
            hide_index=True,
        )


def _render_representative_script(org, by_id) -> None:
    """Show the representative transcript, with grounded highlights when a member
    incident carries trigger spans for exactly this text."""
    source = next(
        (
            by_id[m]
            for m in org.members
            if m in by_id and by_id[m].transcript == org.representative_script
        ),
        None,
    )
    if source is not None and source.label.trigger_spans:
        render_highlighted_transcript(source.transcript, source.label.trigger_spans)
    else:
        st.write(org.representative_script)
