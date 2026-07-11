"""Deterministic PII redaction for raw/synthetic transcripts before they ship in the
corpus (CLAUDE.md SS6 -- data provenance is graded, and nothing PII-bearing may leak).

`scrub_text` is pure, idempotent, and order-sensitive: rules are applied
most-specific-first (email -> card -> IIN -> phone) so a longer digit run is never
half-consumed by a looser downstream rule. Money amounts, percentages, and short digit
runs (e.g. a 6-digit OTP) are never PII under these rules and are always preserved
verbatim.
"""

from __future__ import annotations

import re

PHONE_PLACEHOLDER = "[PHONE]"
CARD_PLACEHOLDER = "[CARD]"
IIN_PLACEHOLDER = "[IIN]"
EMAIL_PLACEHOLDER = "[EMAIL]"

# Order matters: most-specific (longest/most-constrained) pattern first, so a 16-digit
# card or 12-digit IIN is fully consumed before the looser 10/11-digit phone rule runs.
_EMAIL_PATTERN = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
_CARD_PATTERN = re.compile(r"\b\d{16}\b|\b\d{4}[ -]\d{4}[ -]\d{4}[ -]\d{4}\b")
_IIN_PATTERN = re.compile(r"\b\d{12}\b")
# KZ phone: "+7"/"8"/bare "7" country/trunk prefix + 10 digits, with optional
# spaces/dashes/parens between groups. Lookaround guards avoid eating into a longer
# digit run (e.g. a stray leftover 12/16-digit sequence) from either end.
_PHONE_PATTERN = re.compile(
    r"(?<!\d)(?:\+7|8|7)[\s\-]?\(?\d{3}\)?[\s\-]?\d{3}[\s\-]?\d{2}[\s\-]?\d{2}(?!\d)"
)


def scrub_text(text: str) -> str:
    """Return a new string with all PII (email, card, IIN, phone) replaced by
    placeholders. Pure and idempotent: `scrub_text(scrub_text(t)) == scrub_text(t)`.

    Raises `ValueError` if `text` is not a `str`.
    """
    if not isinstance(text, str):
        raise ValueError(f"scrub_text expects a str, got {type(text).__name__}")
    if not text.strip():
        return text

    scrubbed = _EMAIL_PATTERN.sub(EMAIL_PLACEHOLDER, text)
    scrubbed = _CARD_PATTERN.sub(CARD_PLACEHOLDER, scrubbed)
    scrubbed = _IIN_PATTERN.sub(IIN_PLACEHOLDER, scrubbed)
    scrubbed = _PHONE_PATTERN.sub(PHONE_PLACEHOLDER, scrubbed)
    return scrubbed
