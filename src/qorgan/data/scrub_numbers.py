"""Numbers dictated as words, and digit groups split by spaces (ADR D66, gap M7).

Voice mode writes «восемь семьсот ноль один …», «сегіз жүз жетпіс …», "eight seven zero …" or a
mix («8 707 сто двадцать три …»). A *run* is a sequence of number tokens (number words and
ASCII digit groups) separated only by short gaps of spaces, commas or dashes. Each run is
composed into the digit string a listener would write down (dictation groups: «семьсот двадцать
три» = 723, «семьсот ноль один» = 701, «сегіз жүз» = 800, «он бір» = 11) and then:

- a run that contains a number WORD is masked when that string reaches
  `SPOKEN_NUMBER_MIN_DIGITS` digits -- shorter quantities («два раза», «шесть цифр», a dictated
  6-digit code, like the digit rule's 6-digit OTP) are evidence and stay;
- a digits-only run (two or more groups joined by single spaces/hyphens) is masked only when it
  has a known PII shape (16 card, 12 IIN, 10/11 phone), which the regex rules miss when the
  grouping is unusual («940101 300123», «707 123 45 67»);
- a run followed by a currency word or sign is an amount and stays («150 000 000 000 тенге»);
- a run may continue over an utterance break (a dictation split by the endpointer), but it is
  replaced line by line so the utterance structure survives.

Multiplier words (тысяча, миллион, мың, thousand …) are not number tokens, so amounts said with
them never form a long run. `site/core/scrub-numbers.js` is a 1:1 port.
"""

from __future__ import annotations

from collections.abc import Sequence

from qorgan.data.scrub_tokens import (
    CARD_PLACEHOLDER,
    IIN_PLACEHOLDER,
    NUMBER_PLACEHOLDER,
    PHONE_PLACEHOLDER,
    Token,
    apply_replacements,
    tokenize,
)

SPOKEN_NUMBER_MIN_DIGITS = 7
_MAX_GAP_CHARS = 3
# Dictation gaps only: an em/en dash is prose punctuation («12345678901 — одиннадцать цифр»).
_GAP_CHARS = frozenset(" \t\r\n\u00a0,-")
_DIGIT_GROUP_GAPS = frozenset({" ", "-"})
_ASCII_WORD = frozenset("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_")
_CURRENCY_WORDS = frozenset(
    {"тенге", "теңге", "тг", "tenge", "kzt", "рублей", "рубля", "руб", "долларов", "доллара",
     "dollars", "usd", "евро", "euro", "сом"}
)
_CURRENCY_SIGNS = frozenset("₸$€₽")


def _words(table: dict[int, tuple[str, ...]]) -> dict[str, int]:
    return {word: value for value, words in table.items() for word in words}


# Values: units 1-9, teens 10-19 (complete two-digit words), tens digit 1-9, hundreds digit 1-9.
# ASR spellings and Kazakh written without its special letters are listed next to the norm.
_ZERO = frozenset({"ноль", "нуль", "нөл", "нол", "zero"})
_UNITS = _words(
    {
        1: ("один", "одна", "одно", "бір", "бир", "one"),
        2: ("два", "две", "екі", "еки", "two"),
        3: ("три", "үш", "уш", "three"),
        4: ("четыре", "төрт", "торт", "four"),
        5: ("пять", "бес", "five"),
        6: ("шесть", "алты", "six"),
        7: ("семь", "жеті", "жети", "seven"),
        8: ("восемь", "сегіз", "сегиз", "eight"),
        9: ("девять", "тоғыз", "тогыз", "nine"),
    }
)
_TEENS = _words(
    {
        10: ("десять", "ten"),
        11: ("одиннадцать", "одинадцать", "eleven"),
        12: ("двенадцать", "twelve"),
        13: ("тринадцать", "thirteen"),
        14: ("четырнадцать", "fourteen"),
        15: ("пятнадцать", "fifteen"),
        16: ("шестнадцать", "sixteen"),
        17: ("семнадцать", "seventeen"),
        18: ("восемнадцать", "eighteen"),
        19: ("девятнадцать", "nineteen"),
    }
)
# Kazakh «он» (10) takes a unit like the other tens: «он бір» = 11.
_TENS = _words(
    {
        1: ("он",),
        2: ("двадцать", "жиырма", "twenty"),
        3: ("тридцать", "отыз", "thirty"),
        4: ("сорок", "қырық", "кырык", "forty"),
        5: ("пятьдесят", "пятдесят", "елу", "fifty"),
        6: ("шестьдесят", "шестдесят", "алпыс", "sixty"),
        7: ("семьдесят", "семдесят", "жетпіс", "жетпис", "seventy"),
        8: ("восемьдесят", "восемдесят", "сексен", "eighty"),
        9: ("девяносто", "тоқсан", "токсан", "ninety"),
    }
)
_HUNDREDS = _words(
    {
        1: ("сто",),
        2: ("двести",),
        3: ("триста",),
        4: ("четыреста",),
        5: ("пятьсот", "пятсот"),
        6: ("шестьсот", "шестсот"),
        7: ("семьсот", "семсот"),
        8: ("восемьсот", "восемсот"),
        9: ("девятьсот", "девятсот"),
    }
)
# «сегіз жүз» / "eight hundred": the preceding unit becomes the hundreds digit.
_HUNDRED_MULTIPLIERS = frozenset({"жүз", "жуз", "hundred"})


def _number_kind(token: Token) -> tuple[str, int] | None:
    if token.kind == "digits":
        return ("digits", 0)
    word = token.folded
    if word in _ZERO:
        return ("zero", 0)
    for kind, table in (("unit", _UNITS), ("teen", _TEENS), ("tens", _TENS), ("hundreds", _HUNDREDS)):
        if word in table:
            return (kind, table[word])
    if word in _HUNDRED_MULTIPLIERS:
        return ("multiplier", 1)
    return None


# --- composition -----------------------------------------------------------------------------
# A pending dictation group is (form, hundreds, tens, units, teen); form is "H", "T" or "U".
_Group = tuple[str, int | None, int | None, int | None, int | None]


def _flush(group: _Group | None) -> str:
    if group is None:
        return ""
    form, hundreds, tens, units, teen = group
    if form == "U":
        return str(units)
    if form == "T":
        return f"{tens}{units if units is not None else 0}"
    if teen is not None:
        return f"{hundreds}{teen}"
    return f"{hundreds}{tens if tens is not None else 0}{units if units is not None else 0}"


def _step(group: _Group | None, kind: str, value: int, text: str) -> tuple[str, _Group | None]:
    """Fold one number token into the pending group: `(digits emitted, new pending group)`."""
    form = group[0] if group else None
    open_hundreds = form == "H" and group[2] is None and group[3] is None and group[4] is None
    if kind == "digits":
        return _flush(group) + text, None
    if kind == "zero":
        if open_hundreds:
            return "", ("H", group[1], 0, None, None)
        if form == "H" and group[2] == 0 and group[3] is None:
            return _flush(("H", group[1], 0, 0, None)), None
        return _flush(group) + "0", None
    if kind == "unit":
        if form == "H" and group[3] is None and group[4] is None:
            return _flush(("H", group[1], group[2], value, None)), None
        if form == "T" and group[3] is None:
            return _flush(("T", None, group[2], value, None)), None
        return _flush(group), ("U", None, None, value, None)
    if kind == "teen":
        if open_hundreds:
            return _flush(("H", group[1], None, None, value)), None
        return _flush(group) + str(value), None
    if kind == "tens":
        if open_hundreds:
            return "", ("H", group[1], value, None, None)
        return _flush(group), ("T", None, value, None, None)
    if kind == "hundreds":
        return _flush(group), ("H", value, None, None, None)
    # multiplier: «сегіз жүз» turns a lone unit into hundreds; a bare «жүз» is 100
    if form == "U":
        return "", ("H", group[3], None, None, None)
    return _flush(group), ("H", 1, None, None, None)


def compose_digits(tokens: Sequence[Token]) -> str:
    digits: list[str] = []
    group: _Group | None = None
    for token in tokens:
        kind, value = _number_kind(token) or ("", 0)
        emitted, group = _step(group, kind, value, token.text)
        digits.append(emitted)
    digits.append(_flush(group))
    return "".join(digits)


def spoken_digits(text: str) -> str:
    """The digit string of the number tokens in `text` (for tests and diagnostics)."""
    return compose_digits([t for t in tokenize(text) if _number_kind(t) is not None])


# --- runs ------------------------------------------------------------------------------------


def _joinable(gap: str) -> bool:
    return len(gap) <= _MAX_GAP_CHARS and all(character in _GAP_CHARS for character in gap)


def _runs(text: str, tokens: Sequence[Token]) -> list[list[int]]:
    """Index lists of maximal number-token runs."""
    runs: list[list[int]] = []
    current: list[int] = []
    for index, token in enumerate(tokens):
        if _number_kind(token) is None:
            if current:
                runs.append(current)
            current = []
            continue
        if current and not _joinable(text[tokens[current[-1]].end : token.start]):
            runs.append(current)
            current = []
        current.append(index)
    if current:
        runs.append(current)
    return runs


def _placeholder_for(digits: str) -> str:
    if len(digits) == 16:
        return CARD_PLACEHOLDER
    if len(digits) == 12:
        return IIN_PLACEHOLDER
    if (len(digits) == 11 and digits[0] in "78") or (len(digits) == 10 and digits[0] == "7"):
        return PHONE_PLACEHOLDER
    return NUMBER_PLACEHOLDER


def _followed_by_currency(text: str, end: int, next_token: Token | None) -> bool:
    rest = text[end:].lstrip(" \u00a0")
    if rest[:1] in _CURRENCY_SIGNS:
        return True
    if next_token is None or text[end : next_token.start].strip(" \u00a0"):
        return False
    return next_token.folded in _CURRENCY_WORDS


def _digit_groups_qualify(text: str, run: Sequence[Token], digits: str) -> bool:
    if len(run) < 2 or _placeholder_for(digits) == NUMBER_PLACEHOLDER:
        return False
    gaps = [text[a.end : b.start] for a, b in zip(run, run[1:])]
    if not all(gap in _DIGIT_GROUP_GAPS for gap in gaps):
        return False
    before = text[run[0].start - 1] if run[0].start > 0 else ""
    after = text[run[-1].end] if run[-1].end < len(text) else ""
    return before not in _ASCII_WORD and after not in _ASCII_WORD


def _should_mask(text: str, run: Sequence[Token], digits: str) -> bool:
    if any(token.kind == "letters" for token in run):
        return len(digits) >= SPOKEN_NUMBER_MIN_DIGITS
    return _digit_groups_qualify(text, run, digits)


def _line_spans(text: str, run: Sequence[Token]) -> list[tuple[int, int]]:
    """`(start, end)` of the run's pieces on each line: a run never swallows a line break."""
    spans: list[tuple[int, int]] = []
    start, end = run[0].start, run[0].end
    for previous, token in zip(run, run[1:]):
        if "\n" in text[previous.end : token.start]:
            spans.append((start, end))
            start = token.start
        end = token.end
    spans.append((start, end))
    return spans


def mask_number_runs(text: str) -> str:
    tokens = tokenize(text)
    replacements: list[tuple[int, int, str]] = []
    for indices in _runs(text, tokens):
        run = [tokens[i] for i in indices]
        next_token = tokens[indices[-1] + 1] if indices[-1] + 1 < len(tokens) else None
        digits = compose_digits(run)
        if not _should_mask(text, run, digits) or _followed_by_currency(text, run[-1].end, next_token):
            continue
        placeholder = _placeholder_for(digits)
        replacements.extend((start, end, placeholder) for start, end in _line_spans(text, run))
    return apply_replacements(text, replacements)
