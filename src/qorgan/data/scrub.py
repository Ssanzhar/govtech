"""Deterministic PII redaction for raw/synthetic transcripts before they ship in the
corpus (CLAUDE.md SS6 -- data provenance is graded, and nothing PII-bearing may leak).

`scrub_text` is pure, idempotent, and order-sensitive: rules are applied
most-specific-first (email -> IBAN -> card -> IIN -> phone -> number runs -> names) so a
longer digit run is never half-consumed by a looser downstream rule. Money amounts,
percentages, and short digit runs (e.g. a 6-digit OTP) are never PII under these rules and
are always preserved verbatim.

ADR D66 added the context-aware rules voice mode needs: numbers dictated as words or as
mixed digits + words, digit groups split by spaces (`scrub_numbers`), KZ IBANs, and
self-introduced names (`scrub_names`). Scam evidence is never masked: only the number or
name tokens are replaced, the request around them stays verbatim.
"""

from __future__ import annotations

import re

from qorgan.data.scrub_names import mask_names
from qorgan.data.scrub_numbers import SPOKEN_NUMBER_MIN_DIGITS, mask_number_runs
from qorgan.data.scrub_tokens import (
    CARD_PLACEHOLDER,
    EMAIL_PLACEHOLDER,
    IBAN_PLACEHOLDER,
    IIN_PLACEHOLDER,
    NAME_PLACEHOLDER,
    NUMBER_PLACEHOLDER,
    PHONE_PLACEHOLDER,
)

__all__ = [
    "CARD_PLACEHOLDER",
    "EMAIL_PLACEHOLDER",
    "IBAN_PLACEHOLDER",
    "IIN_PLACEHOLDER",
    "NAME_PLACEHOLDER",
    "NUMBER_PLACEHOLDER",
    "PHONE_PLACEHOLDER",
    "PLACEHOLDERS",
    "SCRUBBER_VERSION",
    "SPOKEN_NUMBER_MIN_DIGITS",
    "STORED_RECORD_CONTEXT",
    "is_stored_record",
    "scrub_text",
]

# Bumped whenever a rule change can alter already-scrubbed text. The corpus manifest records the
# version its text was scrubbed with (absent = 1); data on disk older than this is a pending
# rebuild, and `scripts/hf_upload.py` refuses to publish it until then (it is no fixed point).
# 1 = digits-only rules (D45); 2 = spoken numbers, split digit groups, IBAN, names (D66).
SCRUBBER_VERSION = 2

PLACEHOLDERS: tuple[str, ...] = (
    PHONE_PLACEHOLDER,
    CARD_PLACEHOLDER,
    IIN_PLACEHOLDER,
    EMAIL_PLACEHOLDER,
    IBAN_PLACEHOLDER,
    NUMBER_PLACEHOLDER,
    NAME_PLACEHOLDER,
)

# Order matters: most-specific (longest/most-constrained) pattern first, so a 16-digit
# card or 12-digit IIN is fully consumed before the looser 10/11-digit phone rule runs.
_EMAIL_PATTERN = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
# Card / IIN edges: not `\b` -- in Python's Unicode `re` a Cyrillic/Kazakh letter is a word
# char, so "карта4400123456789010" or "ЖСН940101300123" (ASR output, hand-edited reports)
# would slip through. Only an ASCII letter/underscore or any digit blocks the match, which
# keeps ASCII system identifiers (digests, receipt ids) a fixed point for the validators
# that use `scrub_text` as a "no PII" check.
_EDGE_BEFORE = r"(?<![A-Za-z_\d])"
_EDGE_AFTER = r"(?![A-Za-z_\d])"
# Validation context for records read back from disk (ADR D66). The content check
# `scrub_text(value) == value` guards every write; a stored record was checked by the scrubber
# of its day, and re-checking it with a stricter one would break the audit chain's verification
# and make the reports file unreadable. Loaders pass this context; only the content check reads it.
STORED_RECORD_CONTEXT: dict[str, bool] = {"qorgan_stored_record": True}


def is_stored_record(info: object) -> bool:
    """True inside a validator when the record is being read back from storage."""
    context = getattr(info, "context", None)
    return bool(context and context.get("qorgan_stored_record"))


# KZ IBAN: "KZ" + 2 check digits + 16 alphanumerics, optionally grouped by single spaces.
_IBAN_PATTERN = re.compile(r"(?<![A-Za-z0-9_])[Kk][Zz][0-9]{2}(?: ?[A-Za-z0-9]){16}(?![A-Za-z0-9_])")
_CARD_PATTERN = re.compile(
    rf"{_EDGE_BEFORE}(?:\d{{16}}|\d{{4}}[ -]\d{{4}}[ -]\d{{4}}[ -]\d{{4}}){_EDGE_AFTER}"
)
_IIN_PATTERN = re.compile(rf"{_EDGE_BEFORE}\d{{12}}{_EDGE_AFTER}")
# KZ phone: "+7"/"8"/bare "7" country/trunk prefix + 10 digits, with optional
# spaces/dashes/parens between groups. Lookaround guards avoid eating into a longer
# digit run (e.g. a stray leftover 12/16-digit sequence) from either end.
_PHONE_PATTERN = re.compile(
    r"(?<!\d)(?:\+7|8|7)[\s\-]?\(?\d{3}\)?[\s\-]?\d{3}[\s\-]?\d{2}[\s\-]?\d{2}(?!\d)"
)


def scrub_text(text: str) -> str:
    """Return a new string with all PII (email, IBAN, card, IIN, phone, spoken or split
    numbers, self-introduced names) replaced by placeholders. Pure and idempotent: `scrub_text(scrub_text(t)) == scrub_text(t)`.

    Raises `ValueError` if `text` is not a `str`.
    """
    if not isinstance(text, str):
        raise ValueError(f"scrub_text expects a str, got {type(text).__name__}")
    if not text.strip():
        return text

    scrubbed = _EMAIL_PATTERN.sub(EMAIL_PLACEHOLDER, text)
    scrubbed = _IBAN_PATTERN.sub(IBAN_PLACEHOLDER, scrubbed)
    scrubbed = _CARD_PATTERN.sub(CARD_PLACEHOLDER, scrubbed)
    scrubbed = _IIN_PATTERN.sub(IIN_PLACEHOLDER, scrubbed)
    scrubbed = _PHONE_PATTERN.sub(PHONE_PLACEHOLDER, scrubbed)
    scrubbed = mask_number_runs(scrubbed)
    return mask_names(scrubbed)
