"""TDD tests for `qorgan.data.scrub` -- deterministic PII redaction for transcripts.

Covers: email/card/IIN/phone redaction rules (applied most-specific-first per CLAUDE.md
SS6 -- explanations/corpus must never leak PII), placeholder constants, idempotency,
non-str input handling, and preservation of legitimate numeric content (money, percents,
OTP-shaped digit runs) so scrubbing never destroys real signal.
"""

import pytest

from qorgan.data.scrub import (
    CARD_PLACEHOLDER,
    EMAIL_PLACEHOLDER,
    IIN_PLACEHOLDER,
    PHONE_PLACEHOLDER,
    scrub_text,
)

# --- placeholders ----------------------------------------------------------------------


def test_placeholder_constants():
    assert PHONE_PLACEHOLDER == "[PHONE]"
    assert CARD_PLACEHOLDER == "[CARD]"
    assert IIN_PLACEHOLDER == "[IIN]"
    assert EMAIL_PLACEHOLDER == "[EMAIL]"


# --- email -------------------------------------------------------------------------------


def test_scrub_email_basic():
    assert scrub_text("Пишите на ivan.petrov@bank.kz срочно") == f"Пишите на {EMAIL_PLACEHOLDER} срочно"


def test_scrub_email_with_plus_and_dots():
    text = "contact: a.b+test@sub.example.co.uk please"
    assert scrub_text(text) == f"contact: {EMAIL_PLACEHOLDER} please"


# --- card --------------------------------------------------------------------------------


def test_scrub_card_solid_16_digits():
    assert scrub_text("Карта 4400123456789010 активна") == f"Карта {CARD_PLACEHOLDER} активна"


def test_scrub_card_grouped_with_spaces():
    assert scrub_text("Карта 4400 1234 5678 9010 активна") == f"Карта {CARD_PLACEHOLDER} активна"


def test_scrub_card_grouped_with_dashes():
    assert scrub_text("Карта 4400-1234-5678-9010 активна") == f"Карта {CARD_PLACEHOLDER} активна"


# --- IIN ---------------------------------------------------------------------------------


def test_scrub_iin_12_digits():
    assert scrub_text("ИИН 940101300123 подтвержден") == f"ИИН {IIN_PLACEHOLDER} подтвержден"


def test_eleven_digit_number_not_starting_7_or_8_is_untouched():
    text = "код 12345612345 тут"
    assert scrub_text(text) == text


def test_thirteen_digit_number_is_untouched():
    text = "число 9401013001234 тут"
    assert scrub_text(text) == text


# --- phone -------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw",
    [
        "+7 777 123 45 67",
        "8 (727) 123-45-67",
        "87071234567",
        "+77071234567",
    ],
)
def test_scrub_phone_variants(raw):
    text = f"Перезвоните на {raw} пожалуйста"
    result = scrub_text(text)
    assert PHONE_PLACEHOLDER in result
    assert raw not in result


def test_scrub_bare_11_digit_run_starting_7_or_8():
    assert scrub_text("номер 77071234567 записан") == f"номер {PHONE_PLACEHOLDER} записан"
    assert scrub_text("номер 87071234567 записан") == f"номер {PHONE_PLACEHOLDER} записан"


# --- preservation guarantees ---------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    ["Переведите 5000 тенге", "Сумма 15000 тенге", "Скидка 30%", "Назовите 6 цифр"],
)
def test_money_percent_and_small_numbers_preserved(text):
    assert scrub_text(text) == text


def test_six_digit_otp_not_scrubbed():
    text = "Код подтверждения 123456 никому не говорите"
    assert scrub_text(text) == text


# --- idempotency -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "Позвоните +7 777 123 45 67 или карта 4400 1234 5678 9010",
        "ИИН 940101300123 и email test@bank.kz",
        "Без PII здесь ничего нет",
        "",
        "   ",
    ],
)
def test_idempotent(text):
    once = scrub_text(text)
    twice = scrub_text(once)
    assert once == twice


# --- empty / whitespace ------------------------------------------------------------------------


def test_empty_string_returns_as_is():
    assert scrub_text("") == ""


def test_whitespace_only_returns_as_is():
    assert scrub_text("   ") == "   "


# --- multiple PII kinds in one string -----------------------------------------------------------


def test_multiple_pii_kinds_all_scrubbed():
    text = (
        "Свяжитесь по email ivan@bank.kz, номер +7 777 123 45 67, "
        "карта 4400 1234 5678 9010, ИИН 940101300123."
    )
    result = scrub_text(text)
    assert EMAIL_PLACEHOLDER in result
    assert PHONE_PLACEHOLDER in result
    assert CARD_PLACEHOLDER in result
    assert IIN_PLACEHOLDER in result
    assert "ivan@bank.kz" not in result
    assert "940101300123" not in result
    assert "4400" not in result


# --- mixed Kazakh/Russian text ---------------------------------------------------------------


def test_scrub_works_on_kazakh_russian_mixed_text():
    text = "Сәлеметсіз бе, картаңыздың нөмірі 4400 1234 5678 9010, рахмет"
    result = scrub_text(text)
    assert result == f"Сәлеметсіз бе, картаңыздың нөмірі {CARD_PLACEHOLDER}, рахмет"


# --- ordering: card/phone in the same string does not corrupt either placeholder -------------


def test_card_then_phone_no_corruption():
    text = "Карта 4400123456789010, позвоните +7 777 123 45 67"
    result = scrub_text(text)
    assert result == f"Карта {CARD_PLACEHOLDER}, позвоните {PHONE_PLACEHOLDER}"


def test_phone_then_card_no_corruption():
    text = "Позвоните +7 777 123 45 67, карта 4400123456789010"
    result = scrub_text(text)
    assert result == f"Позвоните {PHONE_PLACEHOLDER}, карта {CARD_PLACEHOLDER}"


# --- non-str input --------------------------------------------------------------------------


@pytest.mark.parametrize("bad", [None, 123, 3.14, ["text"], {"text"}])
def test_non_str_input_raises_value_error(bad):
    with pytest.raises(ValueError):
        scrub_text(bad)


# --- PII glued to Cyrillic / Kazakh words (ASR output, hand-edited reports) --------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("карта4400123456789010 без пробела", f"карта{CARD_PLACEHOLDER} без пробела"),
        ("ЖСН940101300123 бірге жазылған", f"ЖСН{IIN_PLACEHOLDER} бірге жазылған"),
        ("ИИН940101300123", f"ИИН{IIN_PLACEHOLDER}"),
        ("номер87071234567записан", f"номер{PHONE_PLACEHOLDER}записан"),
    ],
)
def test_pii_glued_to_cyrillic_is_scrubbed(text, expected):
    assert scrub_text(text) == expected


@pytest.mark.parametrize(
    "identifier",
    [
        "a1b2c3940101300123d4",  # 12 digits inside an ASCII token (hex-like digest)
        "f4400123456789010e",  # 16 digits inside an ASCII token
        "report_940101300123",
        "rcpt-4f2c9a1b77",
    ],
)
def test_ascii_identifiers_with_long_digit_runs_are_untouched(identifier):
    # audit / feedback / report validators use scrub_text as a "no PII" fixed point on
    # system identifiers; the Cyrillic fix must not start flagging ASCII tokens.
    assert scrub_text(identifier) == identifier
