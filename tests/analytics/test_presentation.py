"""TDD tests for `qorgan.analytics.presentation` — pure analyst-dashboard helpers.

No Streamlit here: display names, tactic profiles, activity series, and KPI totals are
plain functions over `Organization`/`Incident` models so the view layer stays thin.
"""

from datetime import UTC, datetime

import pytest

from qorgan.analytics.presentation import (
    DashboardKpis,
    dashboard_kpis,
    last_activity,
    org_activity_by_day,
    org_display_name,
    org_tactic_profile,
)
from qorgan.data.schema import Incident, Label, Organization, TacticTag


def _incident(iid, tags=(), ts=None, family=None):
    return Incident(
        id=iid,
        dialogue_id=iid,
        transcript="алло это банк",
        label=Label(risk=0.9, tactic_tags=tuple(TacticTag(id=t) for t in tags)),
        timestamp=ts,
        script_family=family,
    )


def _org(members, **kwargs):
    return Organization(id="org_0", members=tuple(members), **kwargs)


BANK_ORG = _org(("i1", "i2", "i3"))
BANK_INCIDENTS = {
    "i1": _incident("i1", tags=("impersonation_bank", "otp_request")),
    "i2": _incident("i2", tags=("impersonation_bank",)),
    "i3": _incident("i3", tags=("urgency",)),
}


# --- org_display_name ------------------------------------------------------------------------


def test_display_name_uses_dominant_tactics_not_cluster_id():
    name = org_display_name(BANK_ORG, BANK_INCIDENTS, locale="ru")

    assert "org_0" not in name
    assert "Выдаёт себя за банк" in name  # shortened from the full "…/ службу безопасности…"
    assert "/" not in name  # long taxonomy names are truncated at the slash


def test_display_name_is_localized():
    ru = org_display_name(BANK_ORG, BANK_INCIDENTS, locale="ru")
    kk = org_display_name(BANK_ORG, BANK_INCIDENTS, locale="kk")

    assert ru != kk


def test_display_name_falls_back_to_script_family_without_tags():
    org = _org(("i1", "i2"))
    incidents = {
        "i1": _incident("i1", family="bank_security"),
        "i2": _incident("i2", family="bank_security"),
    }

    assert org_display_name(org, incidents, locale="ru") == "bank_security"


def test_display_name_falls_back_to_org_id_as_last_resort():
    org = _org(("i1",))

    assert org_display_name(org, {"i1": _incident("i1")}, locale="ru") == "org_0"


def test_display_name_skips_unknown_tactic_ids():
    org = _org(("i1",))
    incidents = {"i1": _incident("i1", tags=("future_tactic_v9", "urgency"))}

    name = org_display_name(org, incidents, locale="ru")
    assert "future_tactic_v9" not in name
    assert name  # falls through to the known tactic


def test_display_name_ignores_members_missing_from_incident_map():
    org = _org(("i1", "ghost"))

    name = org_display_name(org, {"i1": _incident("i1", tags=("urgency",))}, locale="ru")
    assert name  # no KeyError, ghost member skipped


# --- org_tactic_profile ------------------------------------------------------------------------


def test_tactic_profile_counts_tags_across_members():
    profile = org_tactic_profile(BANK_ORG, BANK_INCIDENTS)

    assert profile["impersonation_bank"] == 2
    assert profile["otp_request"] == 1
    assert profile["urgency"] == 1


def test_tactic_profile_is_ordered_most_common_first():
    profile = org_tactic_profile(BANK_ORG, BANK_INCIDENTS)

    assert next(iter(profile)) == "impersonation_bank"


def test_tactic_profile_empty_when_no_tags():
    org = _org(("i1",))

    assert org_tactic_profile(org, {"i1": _incident("i1")}) == {}


# --- activity ---------------------------------------------------------------------------------


def test_activity_by_day_buckets_iso_dates_ascending():
    org = _org(("i1", "i2", "i3"))
    incidents = {
        "i1": _incident("i1", ts=datetime(2026, 7, 11, 9, 0, tzinfo=UTC)),
        "i2": _incident("i2", ts=datetime(2026, 7, 10, 15, 0, tzinfo=UTC)),
        "i3": _incident("i3", ts=datetime(2026, 7, 10, 8, 0, tzinfo=UTC)),
    }

    activity = org_activity_by_day(org, incidents)

    assert activity == {"2026-07-10": 2, "2026-07-11": 1}
    assert list(activity) == ["2026-07-10", "2026-07-11"]


def test_activity_skips_incidents_without_timestamp():
    org = _org(("i1", "i2"))
    incidents = {
        "i1": _incident("i1", ts=datetime(2026, 7, 10, tzinfo=UTC)),
        "i2": _incident("i2"),
    }

    assert org_activity_by_day(org, incidents) == {"2026-07-10": 1}


def test_last_activity_returns_latest_timestamp_or_none():
    org = _org(("i1", "i2"))
    incidents = {
        "i1": _incident("i1", ts=datetime(2026, 7, 10, tzinfo=UTC)),
        "i2": _incident("i2", ts=datetime(2026, 7, 12, tzinfo=UTC)),
    }

    assert last_activity(org, incidents) == datetime(2026, 7, 12, tzinfo=UTC)
    assert last_activity(_org(("i3",)), {"i3": _incident("i3")}) is None


# --- KPIs -------------------------------------------------------------------------------------


def test_dashboard_kpis_totals():
    orgs = [
        Organization(id="a", members=("i1",), is_novel=True),
        Organization(id="b", members=("i2", "i3")),
    ]
    incidents = [_incident("i1"), _incident("i2"), _incident("i3")]

    kpis = dashboard_kpis(orgs, incidents, pending_reports=2)

    assert kpis == DashboardKpis(
        incidents_total=3, organizations_total=2, novel_schemes=1, pending_reports=2
    )


def test_short_tactic_name_truncates_and_handles_unknown():
    from qorgan.analytics.presentation import short_tactic_name

    assert short_tactic_name("impersonation_bank", locale="ru") == "Выдаёт себя за банк"
    assert short_tactic_name("nope_v9", locale="ru") is None


def test_dashboard_kpis_is_frozen():
    kpis = dashboard_kpis([], [], pending_reports=0)
    with pytest.raises(Exception):
        kpis.incidents_total = 99  # type: ignore[misc]
