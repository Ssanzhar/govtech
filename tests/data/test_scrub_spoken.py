"""ADR D66: numbers the on-device recogniser writes as WORDS (and mixed digit/word forms),
IBANs and digit runs split by spaces must not reach a stored report.

Rules pinned here:
- a run of number words (ru / kk / en, incl. ASR spellings), optionally mixed with digit groups,
  is masked when its digit-equivalent length reaches `SPOKEN_NUMBER_MIN_DIGITS` (7);
- shorter quantities are evidence and stay verbatim («два раза», «код из шести цифр»,
  «назовите шесть цифр», a dictated 6-digit OTP, like the digit rule's 6-digit OTP);
- the placeholder names the kind when the digit string has a known shape (16 = card,
  12 = IIN, 10/11 starting 7/8 = phone), otherwise `[NUMBER]`;
- only the number tokens are replaced: the request around them survives verbatim.
"""

import pytest

from qorgan.data.scrub import (
    CARD_PLACEHOLDER,
    IBAN_PLACEHOLDER,
    IIN_PLACEHOLDER,
    NUMBER_PLACEHOLDER,
    PHONE_PLACEHOLDER,
    SPOKEN_NUMBER_MIN_DIGITS,
    scrub_text,
)
from qorgan.data.scrub_numbers import spoken_digits


def test_threshold_constant():
    assert SPOKEN_NUMBER_MIN_DIGITS == 7
    assert NUMBER_PLACEHOLDER == "[NUMBER]"
    assert IBAN_PLACEHOLDER == "[IBAN]"


# --- digit-equivalent composition -------------------------------------------------------


@pytest.mark.parametrize(
    ("words", "digits"),
    [
        ("восемь семьсот ноль один", "8701"),
        ("сто ноль ноль", "100"),
        ("семьсот ноль", "700"),
        ("семьсот двадцать три", "723"),
        ("сто", "100"),
        ("сорок пять", "45"),
        ("десять", "10"),
        ("двенадцать", "12"),
        ("сто двенадцать", "112"),
        ("семьсот один", "701"),
        ("ноль семь", "07"),
        ("сегіз жүз жетпіс", "870"),
        ("он бір", "11"),
        ("жүз", "100"),
        ("жетпіс жеті", "77"),
        ("eight seven zero one", "8701"),
        ("seven hundred twenty three", "723"),
        ("nineteen", "19"),
        ("8 707 сто двадцать три", "8707123"),
        ("пятсот семсот", "500700"),  # ASR spellings
        ("сегиз жуз жетпис", "870"),  # Kazakh without the special letters
    ],
)
def test_spoken_digits_composes_dictation_groups(words, digits):
    assert spoken_digits(words) == digits


# --- phones ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "spoken",
    [
        "восемь семьсот ноль один двести тридцать четыре пятьдесят шесть семьдесят восемь",
        "восемь семь ноль семь один два три четыре пять шесть семь",
        "сегіз жеті жүз жетпіс бір екі жүз отыз төрт елу алты алпыс жеті",
        "eight seven zero seven one two three four five six seven",
        "8 707 сто двадцать три сорок пять шестьдесят семь",
        "плюс семь семьсот семь сто двадцать три сорок пять шестьдесят семь",
    ],
)
def test_spoken_phone_numbers_are_masked(spoken):
    result = scrub_text(f"Перезвоните мне: {spoken}, пожалуйста")
    assert PHONE_PLACEHOLDER in result
    assert result.startswith("Перезвоните мне: ")
    assert result.endswith(", пожалуйста")
    for word in ("семьсот", "двадцать", "жиырма", "seven", "сорок"):
        assert word not in result


def test_spoken_phone_keeps_surrounding_words_exactly():
    text = "Мой номер восемь семь ноль семь один два три четыре пять шесть семь запишите"
    assert scrub_text(text) == f"Мой номер {PHONE_PLACEHOLDER} запишите"


# --- cards, IINs, other long numbers -----------------------------------------------------


def test_spoken_card_number_is_masked():
    words = " ".join(["четыре", "четыре", "ноль", "ноль"] * 4)
    assert scrub_text(f"номер карты {words} верно") == f"номер карты {CARD_PLACEHOLDER} верно"


def test_spoken_iin_is_masked():
    text = "ЖСН тоғыз төрт нөл бір нөл бір үш нөл нөл бір екі үш"
    assert scrub_text(text) == f"ЖСН {IIN_PLACEHOLDER}"


def test_unshaped_long_spoken_number_is_a_generic_number():
    text = "счёт один два три четыре пять шесть семь восемь девять"
    assert scrub_text(text) == f"счёт {NUMBER_PLACEHOLDER}"


def test_seven_digit_city_number_is_masked():
    text = "звоните два семь два сорок пять шестьдесят семь"
    assert scrub_text(text) == f"звоните {NUMBER_PLACEHOLDER}"


def test_dictation_split_across_utterances_is_masked_line_by_line():
    text = "запишите восемь семьсот ноль один\nдвести тридцать четыре пятьдесят шесть"
    result = scrub_text(text)
    assert result.count("\n") == 1  # the utterance structure is kept
    assert "семьсот" not in result and "тридцать" not in result
    assert result.startswith("запишите ")


# --- short quantities and scam evidence survive ------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "я звонил два раза",
        "подождите пять минут",
        "назовите код из шести цифр",
        "Назовите шесть цифр",
        "алты санды атаңыз",
        "три цифры на обороте",
        "номер карты и три цифры",
        "tell me the six-digit code",
        "код один два три четыре пять шесть",  # a dictated 6-digit OTP, like the digit rule
        "переведите пятьсот тысяч тенге",
        "сто пятьдесят тысяч двести",  # amounts: a multiplier word breaks the run
        "в семь тридцать пять",
        "две тысячи двадцать шестой год",
        "он сказал семь раз",
    ],
)
def test_short_quantities_and_cues_are_untouched(text):
    assert scrub_text(text) == text


def test_scam_request_survives_only_the_code_is_masked():
    text = "Назовите код из СМС: один два три четыре пять шесть семь восемь"
    assert scrub_text(text) == f"Назовите код из СМС: {NUMBER_PLACEHOLDER}"


# --- IBAN and digit groups split by spaces -----------------------------------------------


@pytest.mark.parametrize(
    "iban",
    ["KZ86125KZT5004100100", "KZ86 125K ZT50 0410 0100", "kz86 125k zt50 0410 0100"],
)
def test_kz_iban_is_masked(iban):
    assert scrub_text(f"переведите на {iban} сегодня") == f"переведите на {IBAN_PLACEHOLDER} сегодня"


def test_iban_glued_to_ascii_token_is_untouched():
    assert scrub_text("xKZ86125KZT5004100100") == "xKZ86125KZT5004100100"


@pytest.mark.parametrize(
    ("raw", "placeholder"),
    [
        ("940101 300123", IIN_PLACEHOLDER),
        ("94 01 01 30 01 23", IIN_PLACEHOLDER),
        ("4400 12345678 9010", CARD_PLACEHOLDER),
        ("44001234 56789010", CARD_PLACEHOLDER),
        ("707 123 45 67", PHONE_PLACEHOLDER),
        ("8 70 71 23 45 67", PHONE_PLACEHOLDER),
    ],
)
def test_digit_groups_split_by_spaces_are_masked(raw, placeholder):
    assert scrub_text(f"номер {raw} записан") == f"номер {placeholder} записан"


@pytest.mark.parametrize(
    "text",
    [
        "сумма 150 000 000 000 тенге",  # thousands groups are an amount, not an IIN
        "переведите 1 500 000 тенге",
        "дата 2026-10-09 время 08:00",
        "коды 123 456",
    ],
)
def test_amounts_dates_and_short_groups_are_untouched(text):
    assert scrub_text(text) == text


# --- idempotency ------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "восемь семь ноль семь один два три четыре пять шесть семь",
        "KZ86 125K ZT50 0410 0100",
        "запишите восемь семьсот ноль один\nдвести тридцать четыре пятьдесят шесть",
        "номер 940101 300123",
    ],
)
def test_idempotent(text):
    once = scrub_text(text)
    assert scrub_text(once) == once


# --- dictation with hesitations (ADR D72) ------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "placeholder"),
    [
        ("940101... эээ... 300123", IIN_PLACEHOLDER),
        ("940101 ээ 300123", IIN_PLACEHOLDER),
        ("940101, әә, 300123", IIN_PLACEHOLDER),
        ("940101… 300123", IIN_PLACEHOLDER),
        ("940101... мм... эм... 300123", IIN_PLACEHOLDER),
        ("707... ээ... 123 45 67", PHONE_PLACEHOLDER),
        ("восемь семьсот ноль один... ээ... двести тридцать четыре пятьдесят шесть семьдесят восемь", PHONE_PLACEHOLDER),
    ],
)
def test_dictation_broken_by_hesitations_is_masked(raw, placeholder):
    assert scrub_text(f"номер {raw} записан") == f"номер {placeholder} записан"


@pytest.mark.parametrize(
    "text",
    [
        "код 123... ээ... 456",  # a 6-digit code stays evidence, hesitation or not
        "пять... ээ... шесть раз",
        "940101. 300123",  # a full stop is not a hesitation
        "940101 и 300123",  # a word is not a hesitation
        "940101 ээ мм эм 300123",  # at most two fillers in one gap
        "940101 а 300123",  # a lone «а» is a conjunction, not a filler
    ],
)
def test_hesitation_rule_does_not_join_unrelated_numbers(text):
    assert scrub_text(text) == text


def test_hesitation_masking_is_idempotent():
    once = scrub_text("ИИН 940101... эээ... 300123, верно")
    assert once == "ИИН [IIN], верно"
    assert scrub_text(once) == once
