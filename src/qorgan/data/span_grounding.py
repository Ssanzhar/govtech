"""Which flagged phrases / trigger spans may stand next to a scrubbed transcript (ADR D44 / D66).

Used for a report's flagged phrases (`reports.store.prepare_report`) and for a corpus dialogue's
gold trigger spans (`data.build_corpus.scrub_dialogue`, so `publish_guard.scrub_file` too).

A flagged phrase is evidence the model grounded in the call. When the scrubber masks PII inside
it (a name, a spoken number), dropping the phrase would throw the evidence away with the PII. So
a phrase is kept in the first of these forms that is verbatim in the stored text and still says
something besides placeholders:

1. the phrase itself (it carried no PII);
2. the phrase scrubbed on its own («Назовите код из СМС: [NUMBER]»);
3. the phrase scrubbed in its original context, for a rule that needs the words before it
   ("this is Major Ivanov from …" stores the span "Major [NAME] from …").

Anything else is dropped, never fabricated; duplicates collapse. `site/core/report.js`
(`groundFlaggedPhrases`) is a 1:1 port.
"""

from __future__ import annotations

from collections.abc import Sequence

from qorgan.data.scrub import PLACEHOLDERS, scrub_text

# How much of the original transcript before a phrase is re-scrubbed with it (form 3): enough
# for the longest trigger plus a title («my name is», «это майор»).
CONTEXT_CHARS = 40


def has_evidence(phrase: str) -> bool:
    """False for a phrase that is nothing but placeholders and punctuation."""
    for placeholder in PLACEHOLDERS:
        phrase = phrase.replace(placeholder, "")
    return any(character.isalpha() for character in phrase)


def _scrubbed_in_context(phrase: str, original: str) -> str | None:
    at = original.find(phrase)
    if at < 0:
        return None
    context = original[max(0, at - CONTEXT_CHARS) : at]
    with_phrase, alone = scrub_text(context + phrase), scrub_text(context)
    return with_phrase[len(alone) :] if with_phrase.startswith(alone) else None


def ground_flagged_phrases(
    flagged_phrases: Sequence[str], scrubbed_transcript: str, original_transcript: str = ""
) -> tuple[str, ...]:
    kept: list[str] = []
    for phrase in flagged_phrases:
        candidate = _grounded_form(phrase, scrubbed_transcript, original_transcript) if phrase else None
        if candidate is not None and candidate not in kept:
            kept.append(candidate)
    return tuple(kept)


def _grounded_form(phrase: str, scrubbed_transcript: str, original_transcript: str) -> str | None:
    if phrase in scrubbed_transcript:
        return phrase
    for candidate in (scrub_text(phrase), _scrubbed_in_context(phrase, original_transcript)):
        if candidate and candidate in scrubbed_transcript and has_evidence(candidate):
            return candidate
    return None
