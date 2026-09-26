"""TDD tests for `qorgan.analysts` -- the analyst credential registry behind the Level-2
console (`/api/admin`). Mirrors the partner registry: parsed once, fails fast on weak,
ambiguous or unknown-role entries, constant-time lookup, secrets never in reprs."""

from __future__ import annotations

import pytest

from qorgan.analysts import (
    MIN_SECRET_CHARS,
    ROLES,
    Analyst,
    AnalystCredential,
    AnalystRegistry,
    parse_analyst_credentials,
)

SECRET = "analyst-secret-with-entropy-0001"
SECRET_2 = "investigator-secret-entropy-0002"


def test_parse_entries_with_roles_and_whitespace():
    creds = parse_analyst_credentials(f" aigerim:{SECRET}:analyst , inv.bek:{SECRET_2}:investigator ")
    assert [(c.id, c.role) for c in creds] == [("aigerim", "analyst"), ("inv.bek", "investigator")]
    assert creds[0].secret.get_secret_value() == SECRET


def test_blank_registry_means_no_analysts():
    assert parse_analyst_credentials("") == ()
    assert parse_analyst_credentials("  ,  ") == ()


@pytest.mark.parametrize(
    "raw",
    [
        f"aigerim:{SECRET}",  # the role is mandatory: privilege is never implied
        "aigerim",  # no secret
        f"aigerim:{SECRET}:analyst:extra",  # too many fields
        f"aigerim:{SECRET}:admin",  # unknown role
        f"aigerim:{SECRET}:Analyst",  # roles are exact
        f"Aigerim:{SECRET}:analyst",  # id must be a lowercase machine id
        f"anonymous-analyst:{SECRET}:analyst",  # reserved: the old unauthenticated default
        f"unauthenticated:{SECRET}:analyst",  # reserved: failed attempts are logged under it
        f"a87012345678:{SECRET}:analyst",  # an id the audit scrubber would read as a phone number
        "aigerim:short:analyst",  # weak secret
        f"aigerim:{SECRET}:analyst,aigerim:{SECRET_2}:investigator",  # duplicate id
        f"aigerim:{SECRET}:analyst,bek:{SECRET}:investigator",  # duplicate secret
    ],
)
def test_malformed_entries_fail_fast(raw):
    with pytest.raises(ValueError):
        parse_analyst_credentials(raw)


def test_parse_errors_never_echo_the_secret():
    with pytest.raises(ValueError) as excinfo:
        parse_analyst_credentials(f"aigerim:{SECRET}:superuser")
    assert SECRET not in str(excinfo.value)


def test_secret_never_appears_in_repr():
    cred = AnalystCredential(id="aigerim", secret=SECRET, role="analyst")
    assert SECRET not in repr(cred) and SECRET not in str(cred)


def test_roles_are_ordered_least_privilege_first():
    assert ROLES == ("analyst", "investigator")
    analyst = Analyst(id="a1", role="analyst")
    investigator = Analyst(id="i1", role="investigator")
    assert analyst.has_role("analyst") and not analyst.has_role("investigator")
    assert investigator.has_role("analyst") and investigator.has_role("investigator")


def test_registry_authenticates_by_secret_and_rejects_everything_else():
    registry = AnalystRegistry(parse_analyst_credentials(f"aigerim:{SECRET}:analyst,bek:{SECRET_2}:investigator"))

    assert registry.authenticate(SECRET) == Analyst(id="aigerim", role="analyst")
    assert registry.authenticate(SECRET_2) == Analyst(id="bek", role="investigator")
    for wrong in (None, "", SECRET[:-1], SECRET + " ", SECRET.upper()):
        assert registry.authenticate(wrong) is None


def test_empty_registry_authenticates_nobody():
    registry = AnalystRegistry(())
    assert registry.is_empty
    assert registry.authenticate(SECRET) is None


def test_min_secret_length_is_meaningful():
    assert MIN_SECRET_CHARS >= 16
