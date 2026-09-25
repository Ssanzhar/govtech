"""TDD tests for `qorgan.partners` -- the partner credential registry behind the partner
intake API (PLAN_2026-09 C5). Parsed once at startup, fails fast on weak or duplicate keys,
constant-time lookup."""

from __future__ import annotations

import pytest

from qorgan.partners import (
    DEFAULT_PARTNER_DAILY_QUOTA,
    MIN_SECRET_CHARS,
    PartnerCredential,
    PartnerRegistry,
    parse_partner_credentials,
)

SECRET = "s3cr3t-with-enough-entropy-0001"


def test_parse_single_entry_with_default_quota():
    creds = parse_partner_credentials(f"bank_a:{SECRET}")
    assert len(creds) == 1
    assert creds[0].id == "bank_a"
    assert creds[0].secret.get_secret_value() == SECRET
    assert creds[0].daily_quota == DEFAULT_PARTNER_DAILY_QUOTA


def test_parse_entries_with_per_partner_quota_and_whitespace():
    creds = parse_partner_credentials(f" bank_a:{SECRET}:50 , telecom-b:{SECRET}x ")
    assert [c.id for c in creds] == ["bank_a", "telecom-b"]
    assert creds[0].daily_quota == 50
    assert creds[1].daily_quota == DEFAULT_PARTNER_DAILY_QUOTA


def test_blank_registry_means_no_partners():
    assert parse_partner_credentials("") == ()
    assert parse_partner_credentials("   ") == ()


@pytest.mark.parametrize(
    "raw",
    [
        "bank_a",  # no secret
        f"Bank:{SECRET}",  # id must be lowercase machine id
        f"bank_a:{SECRET}:zero:extra",  # too many fields
        f"bank_a:{SECRET}:0",  # quota must be positive
        f"bank_a:{SECRET}:ten",  # quota must be an int
        "bank_a:short",  # weak secret
        f"bank_a:{SECRET},bank_a:{SECRET}x",  # duplicate id
        f"bank_a:{SECRET},bank_b:{SECRET}",  # duplicate secret (lookup would be ambiguous)
    ],
)
def test_malformed_entries_fail_fast(raw):
    with pytest.raises(ValueError):
        parse_partner_credentials(raw)


def test_secret_never_appears_in_repr():
    cred = PartnerCredential(id="bank_a", secret=SECRET, daily_quota=10)
    assert SECRET not in repr(cred) and SECRET not in str(cred)


def test_registry_authenticates_by_secret_and_rejects_everything_else():
    registry = PartnerRegistry(parse_partner_credentials(f"bank_a:{SECRET}:5,telecom_b:{SECRET}x"))

    partner = registry.authenticate(SECRET)
    assert partner is not None and partner.id == "bank_a" and partner.daily_quota == 5
    assert registry.authenticate(f"{SECRET}x").id == "telecom_b"
    assert registry.authenticate(None) is None
    assert registry.authenticate("") is None
    assert registry.authenticate(SECRET[:-1]) is None
    assert registry.authenticate(SECRET + " ") is None


def test_empty_registry_authenticates_nobody():
    assert PartnerRegistry(()).authenticate(SECRET) is None
    assert PartnerRegistry(()).is_empty


def test_min_secret_length_is_meaningful():
    assert MIN_SECRET_CHARS >= 16
