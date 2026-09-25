"""TDD tests for `qorgan.privacy.numbers` -- phone numbers are stored only as salted HMACs.

Why (PLAN_2026-09 C2 / ADR D14): a caller number is personal data. The analyst layer only
needs to *link* reports that share a number, which a keyed hash does; it never needs the
number itself. The display prefix (`+7 700 ***`) is the most an analyst sees.
"""

import pytest

from qorgan.privacy.numbers import (
    NUMBER_HASH_PATTERN,
    MissingHmacKeyError,
    display_prefix,
    hash_phone_number,
    is_number_hash,
    normalize_phone_number,
)

KEY = b"unit-test-key-not-a-secret"

# --- normalize ---------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("+7 700 101 20 30", "77001012030"),
        ("8 (700) 101-20-30", "77001012030"),  # trunk 8 -> country 7
        ("7001012030", "77001012030"),  # bare 10-digit subscriber number
        ("77001012030", "77001012030"),
        ("+7-700-101-20-30", "77001012030"),
        ("+380 50 123 45 67", "380501234567"),  # non-KZ numbers pass through as digits
    ],
)
def test_normalize_canonicalises_kz_formats(raw, expected):
    assert normalize_phone_number(raw) == expected


@pytest.mark.parametrize("raw", ["", "   ", "abc", "12345", "+7 700"])
def test_normalize_rejects_non_numbers(raw):
    assert normalize_phone_number(raw) is None


# --- hash ------------------------------------------------------------------------------------


def test_hash_is_deterministic_and_format_insensitive():
    a = hash_phone_number("+7 700 101 20 30", key=KEY)
    b = hash_phone_number("8 700 101-20-30", key=KEY)
    assert a == b
    assert is_number_hash(a)


def test_hash_depends_on_key_and_number():
    base = hash_phone_number("+7 700 101 20 30", key=KEY)
    assert hash_phone_number("+7 700 101 20 30", key=b"other") != base
    assert hash_phone_number("+7 700 101 20 31", key=KEY) != base


def test_hash_never_contains_the_number():
    digest = hash_phone_number("+7 700 101 20 30", key=KEY)
    assert "7001012030" not in digest
    assert "1012030" not in digest


@pytest.mark.parametrize("key", [None, b""])
def test_hash_requires_a_key(key):
    with pytest.raises(MissingHmacKeyError):
        hash_phone_number("+7 700 101 20 30", key=key)


def test_hash_rejects_invalid_numbers():
    with pytest.raises(ValueError):
        hash_phone_number("not a number", key=KEY)


def test_is_number_hash_matches_pattern_only():
    assert not is_number_hash("+7 700 101 20 30")
    assert not is_number_hash("77001012030")
    assert not is_number_hash("")
    import re

    assert re.fullmatch(NUMBER_HASH_PATTERN, hash_phone_number("+7 700 101 20 30", key=KEY))


# --- display prefix --------------------------------------------------------------------------


def test_display_prefix_shows_country_and_operator_code_only():
    assert display_prefix("+7 700 101 20 30") == "+7 700 ***"
    assert display_prefix("8 (777) 555-12-34") == "+7 777 ***"


def test_display_prefix_for_non_kz_and_invalid():
    assert display_prefix("+380 50 123 45 67") == "+380 ***"
    assert display_prefix("garbage") is None
