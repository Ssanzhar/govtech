"""ASR-style text (PLAN_2026-09 A10): make clean corpus text look like the on-device Vosk
recogniser's output so the classifier can be evaluated -- and, if needed, trained -- on what
it will actually see. Measured on synthesized speech (ADR D25) the recogniser emits lowercase
words, no punctuation, numerals spelled out, and occasionally nothing usable for a Latin token
(«SMS» → «самая с»). Everything here is deterministic and idempotent; what it cannot model is
misrecognition itself, so the numbers it produces are a floor on ASR damage, not the damage.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence

from num2words import num2words

from qorgan.data.schema import Dialogue, Label, Utterance

# num2words language codes per dialogue language; code-switched text reads numbers in Russian.
_NUMBER_LANGUAGE = {"ru": "ru", "kk": "kz", "mixed": "ru"}
_DEFAULT_NUMBER_LANGUAGE = "ru"
# "1 250 000" -- digit groups joined by single spaces (thousands separators) read as one number.
_GROUPED_DIGITS = re.compile(r"\d{1,3}(?:[  ]\d{3})+")
_DIGITS = re.compile(r"\d+")
_LATIN_TOKEN = re.compile(r"[A-Za-z]+")
# Letters (any script) and digits survive; every other character is a word boundary.
_NON_WORD = re.compile(r"[^\w]+|_+", re.UNICODE)
_WHITESPACE = re.compile(r"\s+")
ASR_STYLE_ID_SUFFIX = "-asr"


def asr_style(text: str, language: str, *, drop_latin: bool = False) -> str:
    """`text` as the recogniser would emit it: numerals → words in `language`, Latin tokens
    lowercased (or removed with `drop_latin`, the worst case), punctuation → spaces, lowercase,
    single spaces."""
    number_language = _NUMBER_LANGUAGE.get(language, _DEFAULT_NUMBER_LANGUAGE)
    spelled = _GROUPED_DIGITS.sub(lambda m: _spell(m.group(0).replace(" ", "").replace(" ", ""), number_language), text)
    spelled = _DIGITS.sub(lambda m: _spell(m.group(0), number_language), spelled)
    if drop_latin:
        spelled = _LATIN_TOKEN.sub(" ", spelled)
    return _WHITESPACE.sub(" ", _NON_WORD.sub(" ", spelled)).strip().lower()


def _spell(digits: str, number_language: str) -> str:
    return f" {num2words(int(digits), lang=number_language)} "


def asr_style_dialogue(dialogue: Dialogue, *, drop_latin: bool = False) -> Dialogue:
    """A new dialogue whose utterances are ASR-styled; the label keeps risk and tactics but
    loses its trigger spans (they are no longer verbatim in the styled text). An utterance
    that styles to nothing (punctuation only, or a lone Latin token under `drop_latin`) is
    dropped -- a recogniser emits nothing for it. Raises `ValueError` if none remain."""
    styled = [(u.speaker, asr_style(u.text, dialogue.language, drop_latin=drop_latin)) for u in dialogue.utterances]
    utterances = tuple(Utterance(speaker=speaker, text=text) for speaker, text in styled if text)
    if not utterances:
        raise ValueError(f"dialogue {dialogue.id!r} has no utterance left after ASR styling")
    label = Label(risk=dialogue.label.risk, tactic_tags=dialogue.label.tactic_tags, is_hard_negative=dialogue.label.is_hard_negative)
    return dialogue.model_copy(update={"utterances": utterances, "label": label})


def asr_style_augment(
    dialogues: Sequence[Dialogue], *, fraction: float, seed: int, drop_latin: bool = False
) -> tuple[Dialogue, ...]:
    """ASR-styled copies (`<id>-asr`) of a seeded, id-stable subset of `dialogues` -- the
    train-time augmentation that teaches the heads the recogniser's register (PLAN A10).
    `fraction` ∈ [0, 1] of the rows are copied; the choice depends only on (seed, id), so a
    corpus rebuild reproduces it and adding rows never reshuffles the rest. Rows that style
    to nothing are skipped."""
    if not 0.0 <= fraction <= 1.0:
        raise ValueError(f"fraction must be in [0, 1], got {fraction}")
    chosen = []
    for dialogue in dialogues:
        if _unit_hash(f"{seed}:{dialogue.id}") >= fraction:
            continue
        try:
            styled = asr_style_dialogue(dialogue, drop_latin=drop_latin)
        except ValueError:
            continue
        chosen.append(styled.model_copy(update={"id": f"{dialogue.id}{ASR_STYLE_ID_SUFFIX}"}))
    return tuple(chosen)


def _unit_hash(key: str) -> float:
    """Deterministic value in [0, 1) from `key` (sha256), independent of Python's hash seed."""
    return int(hashlib.sha256(key.encode("utf-8")).hexdigest(), 16) / 2**256

