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


# --- names an analyst can read and tell apart ---------------------------------------------------


@pytest.mark.parametrize("locale", ["ru", "kk", "en"])
def test_every_active_tactic_has_a_clean_short_name(locale):
    """The live queue showed `Ложное "подтверждение личности` and `SMS`: a truncation at the first
    slash cut inside a quotation and, in Kazakh, off the verb. Short names are written, not cut."""
    from qorgan.analytics.presentation import short_tactic_name
    from qorgan.taxonomy import get_taxonomy

    for tactic_id in get_taxonomy().tactic_ids():
        name = short_tactic_name(tactic_id, locale=locale)
        assert name and "/" not in name and "(" not in name, (tactic_id, name)
        assert name.count('"') % 2 == 0 and name.count("«") == name.count("»"), (tactic_id, name)
        assert name != "SMS", tactic_id


def test_truncating_fallback_never_leaves_an_open_quote():
    from qorgan.analytics.presentation import _short_name

    assert _short_name('Ложное "подтверждение личности/операции"') == 'Ложное "подтверждение личности"'
    assert _short_name("Ложное «подтверждение личности/операции»") == "Ложное «подтверждение личности»"
    assert _short_name('Требование тайны ("никому")') == "Требование тайны"


def _named_org(org_id: str, members: tuple[str, ...]) -> Organization:
    return Organization(id=org_id, members=members)


def _clashing_orgs() -> tuple[list[Organization], dict[str, Incident]]:
    """Two operations with the same two dominant tactics; only the impersonated role differs."""
    shared = ("fear_threat", "verification_ploy")
    incidents = {
        **{f"a{i}": _incident(f"a{i}", tags=(*shared, "impersonation_bank")) for i in range(3)},
        **{f"b{i}": _incident(f"b{i}", tags=(*shared, "impersonation_gov_police")) for i in range(3)},
        "c0": _incident("c0", tags=("urgency",)),
    }
    orgs = [
        _named_org("org_0", ("a0", "a1", "a2")),
        _named_org("org_1", ("b0", "b1", "b2")),
        _named_org("org_2", ("c0",)),
    ]
    return orgs, incidents


@pytest.mark.parametrize("locale", ["ru", "kk", "en"])
def test_queue_names_are_unique_and_name_what_differs(locale):
    from qorgan.analytics.presentation import org_display_names, short_tactic_name

    orgs, incidents = _clashing_orgs()
    assert org_display_name(orgs[0], incidents, locale=locale) == org_display_name(orgs[1], incidents, locale=locale)

    names = org_display_names(orgs, incidents, locale=locale)

    assert len(set(names.values())) == 3
    assert short_tactic_name("impersonation_bank", locale=locale) in names["org_0"]
    assert short_tactic_name("impersonation_gov_police", locale=locale) in names["org_1"]
    assert names["org_2"] == org_display_name(orgs[2], incidents, locale=locale), "no clash, no suffix"


def test_identical_operations_fall_back_to_their_ids():
    from qorgan.analytics.presentation import org_display_names

    incidents = {"a": _incident("a", tags=("urgency",)), "b": _incident("b", tags=("urgency",))}
    names = org_display_names([_named_org("org_0", ("a",)), _named_org("org_1", ("b",))], incidents, locale="ru")

    assert names["org_0"] != names["org_1"]
    assert "org_0" in names["org_0"] and "org_1" in names["org_1"]
