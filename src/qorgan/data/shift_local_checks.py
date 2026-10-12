"""Automated acceptance for generated `shift_local` calls (ADR D65).

A call written by the local generator is accepted or rejected by these checks ONLY:
format, turn count, both speakers present, turn length, prompt echo, language / script,
degeneration (looping) and duplicates. Nothing here looks at a classifier, a lexicon or a
score -- selecting held-out rows by what our model thinks of them would bias the set.

All checks are pure and deterministic; thresholds are module constants so the manifest can
record them.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

from qorgan.data.build_corpus import normalize_for_dedup

REJECT_REASONS: tuple[str, ...] = (
    "format", "turn_count", "speakers", "turn_length", "prompt_echo", "language", "degenerate", "duplicate",
)

SPEAKER_TAGS = {"caller": "caller", "callee": "callee"}
MIN_TURNS = 8
MAX_TURNS = 22
MIN_TURNS_PER_SPEAKER = 3
MAX_TURN_CHARS = 700
MAX_PREAMBLE_LINES = 2          # "Here is the dialogue:" before the first tag is tolerated
MAX_STRAY_LINES = 2             # untagged lines after the dialogue starts
ECHO_NGRAM = 8                  # copying this many consecutive prompt words is an echo
MAX_REPEATS_OF_ONE_TURN = 2     # the same utterance three times is a generation loop
NEAR_DUP_JACCARD = 0.7          # char 5-gram Jaccard against an already accepted call
NEAR_DUP_N = 5

# Script thresholds (share of alphabetic characters).
MIN_SCRIPT_SHARE = 0.9          # Cyrillic for kk/ru/mixed, Latin for en
KK_MIN_SPECIFIC_SHARE = 0.04    # Kazakh-only letters among Cyrillic letters (kk text ~8-10 %)
KK_MAX_RU_MARKER_TURNS = 0.34   # share of turns carrying a Russian function word
RU_MAX_SPECIFIC_SHARE = 0.01    # a Kazakh name or two is fine, Kazakh sentences are not
MIXED_MIN_SPECIFIC_SHARE = 0.01
MIXED_MIN_KK_TURNS = 2
MIXED_MIN_RU_TURNS = 2

_KK_SPECIFIC = frozenset("әғқңөұүһіӘҒҚҢӨҰҮҺІ")
# Generic Russian function words that Kazakh does not use (no Kazakh homographs); a
# grammar-level marker, not a scam cue.
_RU_MARKERS = frozenset({
    "что", "это", "вы", "вам", "вас", "пожалуйста", "здравствуйте", "сейчас", "нужно", "можно",
    "если", "очень", "просто", "конечно", "хорошо", "спасибо", "надо", "потому", "только", "когда",
    "где", "как", "уже", "ещё", "еще", "мне", "меня", "тоже", "или", "но",
})
_META = re.compile(r"(?i)\b(caller|callee|transcript|brief|scenario)\b|here is the|\bвот диалог")
_TAG_LINE = re.compile(r"^[\s*#>\-]*(caller|callee)[\s*]*[:：][\s*]*(.*)$", re.IGNORECASE)
_WORD = re.compile(r"\w+", re.UNICODE)


@dataclass(frozen=True)
class ScriptStats:
    letters: int
    cyrillic: int
    latin: int
    kazakh_specific: int


@dataclass(frozen=True)
class Verdict:
    reason: str | None
    utterances: tuple[tuple[str, str], ...]

    @property
    def accepted(self) -> bool:
        return self.reason is None

    @property
    def transcript(self) -> str:
        return "\n".join(text for _, text in self.utterances)


def parse_turns(raw: str) -> tuple[tuple[tuple[str, str], ...], int]:
    """`(turns, stray_line_count)`. A short preamble before the first tag is not counted;
    any untagged line after it is (an over-long preamble counts as stray too)."""
    turns: list[tuple[str, str]] = []
    stray = 0
    preamble = 0
    for line in raw.splitlines():
        if not line.strip():
            continue
        match = _TAG_LINE.match(line)
        if match:
            text = match.group(2).replace("**", "").strip()
            if text:
                turns.append((SPEAKER_TAGS[match.group(1).lower()], text))
            continue
        if turns:
            stray += 1
        else:
            preamble += 1
    return tuple(turns), stray + max(0, preamble - MAX_PREAMBLE_LINES)


def script_stats(text: str) -> ScriptStats:
    letters = cyrillic = latin = kazakh = 0
    for char in text:
        if not char.isalpha():
            continue
        letters += 1
        if "Ѐ" <= char <= "ӿ":
            cyrillic += 1
            kazakh += char in _KK_SPECIFIC
        elif char.isascii() or "À" <= char <= "ɏ":
            latin += 1
    return ScriptStats(letters=letters, cyrillic=cyrillic, latin=latin, kazakh_specific=kazakh)


def _has_ru_marker(text: str) -> bool:
    return any(word in _RU_MARKERS for word in _WORD.findall(text.lower()))


def _has_kk_letter(text: str) -> bool:
    return any(char in _KK_SPECIFIC for char in text)


def language_ok(texts: Sequence[str], language: str) -> bool:
    stats = script_stats(" ".join(texts))
    if stats.letters == 0:
        return False
    if language == "en":
        return stats.latin / stats.letters >= MIN_SCRIPT_SHARE
    if stats.cyrillic / stats.letters < MIN_SCRIPT_SHARE:
        return False
    kk_share = stats.kazakh_specific / stats.cyrillic
    ru_turns = sum(_has_ru_marker(t) for t in texts)
    kk_turns = sum(_has_kk_letter(t) for t in texts)
    if language == "ru":
        return kk_share <= RU_MAX_SPECIFIC_SHARE
    if language == "kk":
        return kk_share >= KK_MIN_SPECIFIC_SHARE and ru_turns / len(texts) <= KK_MAX_RU_MARKER_TURNS
    if language == "mixed":
        return kk_share >= MIXED_MIN_SPECIFIC_SHARE and kk_turns >= MIXED_MIN_KK_TURNS and ru_turns >= MIXED_MIN_RU_TURNS
    raise ValueError(f"unsupported language {language!r}")


def _ngrams(words: Sequence[str], n: int) -> set[tuple[str, ...]]:
    return {tuple(words[i : i + n]) for i in range(len(words) - n + 1)}


def echoes_prompt(texts: Sequence[str], prompt: str) -> bool:
    if any(_META.search(t) for t in texts):
        return True
    prompt_grams = _ngrams(_WORD.findall(prompt.lower()), ECHO_NGRAM)
    return any(_ngrams(_WORD.findall(t.lower()), ECHO_NGRAM) & prompt_grams for t in texts)


def is_degenerate(texts: Sequence[str]) -> bool:
    counts = Counter(normalize_for_dedup(t) for t in texts)
    return max(counts.values()) > MAX_REPEATS_OF_ONE_TURN


def char_ngram_jaccard(a: str, b: str, n: int = NEAR_DUP_N) -> float:
    grams_a = {a[i : i + n] for i in range(len(a) - n + 1)}
    grams_b = {b[i : i + n] for i in range(len(b) - n + 1)}
    if not grams_a or not grams_b:
        return 1.0 if a == b else 0.0
    return len(grams_a & grams_b) / len(grams_a | grams_b)


def is_duplicate(transcript: str, accepted: Sequence[str]) -> bool:
    key = normalize_for_dedup(transcript)
    return any(
        key == normalize_for_dedup(other) or char_ngram_jaccard(key, normalize_for_dedup(other)) >= NEAR_DUP_JACCARD
        for other in accepted
    )


def accept(raw: str, *, language: str, prompt: str, accepted: Sequence[str]) -> Verdict:
    """Run every check in a fixed order; the first failure names the rejection reason."""
    turns, stray = parse_turns(raw)
    texts = [text for _, text in turns]
    speakers = Counter(speaker for speaker, _ in turns)
    checks = (  # lazy: a later check may assume an earlier one passed
        ("format", lambda: not turns or stray > MAX_STRAY_LINES),
        ("turn_count", lambda: not MIN_TURNS <= len(turns) <= MAX_TURNS),
        ("speakers", lambda: min(speakers.get("caller", 0), speakers.get("callee", 0)) < MIN_TURNS_PER_SPEAKER),
        ("turn_length", lambda: any(len(t) > MAX_TURN_CHARS for t in texts)),
        ("prompt_echo", lambda: echoes_prompt(texts, prompt)),
        ("language", lambda: not language_ok(texts, language)),
        ("degenerate", lambda: is_degenerate(texts)),
        ("duplicate", lambda: is_duplicate("\n".join(texts), accepted)),
    )
    for reason, failed in checks:
        if failed():
            return Verdict(reason=reason, utterances=turns)
    return Verdict(reason=None, utterances=turns)
