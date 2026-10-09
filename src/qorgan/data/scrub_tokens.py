"""Tokens for the context-aware scrub rules (spoken numbers, self-introduced names; ADR D66).

A token is a maximal run of Unicode letters (category L*) or of ASCII digits; everything else
is a gap. Offsets are code-point indices: `site/core/scrub-tokens.js` walks `Array.from(text)`
so the two runtimes agree even on characters outside the BMP. Pure functions only.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from typing import Literal

TokenKind = Literal["letters", "digits"]

# Every placeholder the scrubber writes (re-exported by `qorgan.data.scrub`).
PHONE_PLACEHOLDER = "[PHONE]"
CARD_PLACEHOLDER = "[CARD]"
IIN_PLACEHOLDER = "[IIN]"
EMAIL_PLACEHOLDER = "[EMAIL]"
IBAN_PLACEHOLDER = "[IBAN]"
NUMBER_PLACEHOLDER = "[NUMBER]"
NAME_PLACEHOLDER = "[NAME]"


@dataclass(frozen=True)
class Token:
    text: str
    start: int
    end: int
    kind: TokenKind

    @property
    def folded(self) -> str:
        """Lower-cased with `ё` → `е`: the form every word list is written in."""
        return fold(self.text)


def fold(word: str) -> str:
    return word.lower().replace("ё", "е")


def _char_kind(character: str) -> TokenKind | None:
    if "0" <= character <= "9":
        return "digits"
    if unicodedata.category(character).startswith("L"):
        return "letters"
    return None


def tokenize(text: str) -> tuple[Token, ...]:
    tokens: list[Token] = []
    start, kind = 0, None
    for index, character in enumerate(text):
        current = _char_kind(character)
        if current != kind:
            if kind is not None:
                tokens.append(Token(text[start:index], start, index, kind))
            start, kind = index, current
    if kind is not None:
        tokens.append(Token(text[start:], start, len(text), kind))
    return tuple(tokens)


def is_capitalised(word: str) -> bool:
    first = word[:1]
    return first != first.lower() and first == first.upper()


def apply_replacements(text: str, replacements: list[tuple[int, int, str]]) -> str:
    """`text` with each non-overlapping `(start, end, placeholder)` substituted (sorted by start)."""
    if not replacements:
        return text
    pieces: list[str] = []
    cursor = 0
    for start, end, placeholder in sorted(replacements):
        pieces.append(text[cursor:start])
        pieces.append(placeholder)
        cursor = end
    pieces.append(text[cursor:])
    return "".join(pieces)
