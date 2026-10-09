"""Self-introduced personal names (ADR D66, gap M7), masked conservatively.

Only the name is replaced; the introduction and the claimed organisation stay, because «это X
из службы безопасности банка» is the impersonation evidence.

- **Strong** triggers announce a name unambiguously («меня зовут», «моё имя», «моя фамилия»,
  «атым», «есімім», "my name is"). The next word is taken in any case -- the recogniser writes
  lower case -- plus up to two more words that are capitalised or carry a surname ending
  («Алексей Петрович Иванов», "elena petrova").
- **Weak** triggers («это», "this is", "I'm", "I am") only count with a capitalised name that is
  followed by «из / с / со / от» or "from / calling / with": one word, or two when the second
  carries a surname ending («Канат Сапаров») -- two capitalised words are usually an
  organisation ("Kazakhstan Post calling"). A rank or title before the name stays («майор»,
  "Major"): it is the impersonation evidence.
- A stop list keeps pronouns, particles, organisation words and brands out of every name slot.

Addresses are out of scope (ADR D66). `site/core/scrub-names.js` is a 1:1 port.
"""

from __future__ import annotations

from collections.abc import Sequence

from qorgan.data.scrub_tokens import NAME_PLACEHOLDER, Token, apply_replacements, is_capitalised, tokenize

_STRONG_TRIGGERS: tuple[tuple[str, ...], ...] = (
    ("меня", "зовут"),
    ("мое", "имя"),
    ("моя", "фамилия"),
    ("атым",),
    ("есімім",),
    ("есимим",),
    ("my", "name", "is"),
)
_WEAK_TRIGGERS: tuple[tuple[str, ...], ...] = (("это",), ("this", "is"), ("i", "m"), ("i", "am"))
_WEAK_FOLLOWERS = frozenset({"из", "с", "со", "от", "from", "calling", "with"})
_STOP_WORDS = frozenset(
    {
        # ru
        "не", "я", "мы", "вы", "он", "она", "они", "это", "из", "с", "со", "от", "и", "а", "но",
        "ваш", "ваша", "вам", "вас", "так", "как", "что", "сотрудник", "сотрудница", "служба",
        "службы", "банк", "банка", "отдел", "полиция", "оператор", "специалист", "менеджер",
        "нацбанк", "казпочта", "налоговая", "компания",
        # kk
        "ба", "бе", "ма", "ме", "па", "пе", "мен", "сіз", "біз", "банк", "банктен", "қызметкер",
        "қызметкері", "қызметі",
        # en
        "the", "a", "an", "from", "calling", "with", "and", "i", "not", "your", "bank", "service",
        "security", "officer", "agent", "here", "just", "is", "post", "kazakhstan", "казахстан",
        # brands callers impersonate: evidence, never a personal name
        "kaspi", "каспи", "halyk", "халык", "халық", "homebank", "kazpost", "казпост", "kcell",
        "кселл", "beeline", "билайн", "tele", "теле", "egov", "егов", "jusan", "жусан", "forte",
        "форте", "bereke", "береке", "freedom", "фридом", "kazakhtelecom", "казахтелеком",
    }
)
# Kept in front of a weak-trigger name: «это майор [NAME] из …», "this is Major [NAME] from …".
_TITLES = frozenset(
    {
        "майор", "капитан", "лейтенант", "старший", "полковник", "подполковник", "сержант",
        "следователь", "инспектор", "господин", "госпожа", "major", "captain", "lieutenant",
        "colonel", "sergeant", "inspector", "detective", "mr", "mrs", "ms", "dr",
    }
)
_SURNAME_ENDINGS = (
    "ов", "ова", "ев", "ева", "ина", "ын", "ына", "ский", "ская", "цкий", "цкая", "енко", "ұлы",
    "қызы", "улы", "кызы", "ov", "ova", "ev", "eva", "ina", "sky", "skaya", "enko", "uly", "kyzy",
)
_MAX_EXTRA_NAME_WORDS = 2
_MIN_NAME_CHARS = 2
_MIN_SURNAME_CHARS = 4
_TRIGGER_GAP_CHARS = frozenset(" \t\u00a0'\u2019")
_INTRO_GAP_CHARS = frozenset(" \t\u00a0:\u2014\u2013-")
_MAX_INTRO_GAP = 4
_NAME_GAP_CHARS = frozenset(" \t\u00a0")
_MAX_NAME_GAP = 2


def _gap(text: str, left: Token, right: Token) -> str:
    return text[left.end : right.start]


def _gap_ok(gap: str, allowed: frozenset[str], limit: int) -> bool:
    return 0 < len(gap) <= limit and all(character in allowed for character in gap)


def _matches_trigger(text: str, tokens: Sequence[Token], index: int, trigger: tuple[str, ...]) -> bool:
    if index + len(trigger) > len(tokens):
        return False
    window = tokens[index : index + len(trigger)]
    if any(token.folded != word for token, word in zip(window, trigger)):
        return False
    return all(_gap_ok(_gap(text, a, b), _TRIGGER_GAP_CHARS, _MAX_INTRO_GAP) for a, b in zip(window, window[1:]))


def _name_word(token: Token) -> bool:
    return token.kind == "letters" and len(token.text) >= _MIN_NAME_CHARS and token.folded not in _STOP_WORDS


def _looks_like_surname(token: Token) -> bool:
    return len(token.text) >= _MIN_SURNAME_CHARS and token.folded.endswith(_SURNAME_ENDINGS)


def _strong_name(text: str, tokens: Sequence[Token], start: int) -> int:
    """Number of name tokens at `start` after a strong trigger (0 = none)."""
    if start >= len(tokens) or not _name_word(tokens[start]):
        return 0
    count = 1
    while count <= _MAX_EXTRA_NAME_WORDS and start + count < len(tokens):
        token = tokens[start + count]
        if not _gap_ok(_gap(text, tokens[start + count - 1], token), _NAME_GAP_CHARS, _MAX_NAME_GAP):
            break
        if not (_name_word(token) and (is_capitalised(token.text) or _looks_like_surname(token))):
            break
        count += 1
    return count


def _weak_name(text: str, tokens: Sequence[Token], start: int) -> int:
    """Number of capitalised name tokens at `start` that a weak-trigger follower confirms."""
    for count in (2, 1):
        end = start + count
        if end >= len(tokens):
            continue
        names = tokens[start:end]
        if not all(_name_word(t) and is_capitalised(t.text) for t in names):
            continue
        if count == 2 and not _looks_like_surname(names[1]):
            continue
        gaps = [_gap(text, a, b) for a, b in zip(tokens[start - 1 : end], tokens[start : end + 1])]
        if not all(_gap_ok(gap, _NAME_GAP_CHARS, _MAX_NAME_GAP) for gap in gaps):
            continue
        if tokens[end].folded in _WEAK_FOLLOWERS:
            return count
    return 0


def _name_at(text: str, tokens: Sequence[Token], index: int) -> tuple[int, int]:
    """`(first name token, token count)` introduced at `index`, or `(0, 0)`."""
    for trigger in _STRONG_TRIGGERS:
        if not _matches_trigger(text, tokens, index, trigger):
            continue
        start = index + len(trigger)
        if start < len(tokens) and _gap_ok(_gap(text, tokens[start - 1], tokens[start]), _INTRO_GAP_CHARS, _MAX_INTRO_GAP):
            count = _strong_name(text, tokens, start)
            if count:
                return start, count
    for trigger in _WEAK_TRIGGERS:
        if _matches_trigger(text, tokens, index, trigger):
            start = index + len(trigger)
            if start + 1 < len(tokens) and tokens[start].folded in _TITLES:
                start += 1  # the title stays visible; the gap check below covers it
            count = _weak_name(text, tokens, start) if start < len(tokens) else 0
            if count:
                return start, count
    return 0, 0


def mask_names(text: str) -> str:
    tokens = tokenize(text)
    replacements: list[tuple[int, int, str]] = []
    index = 0
    while index < len(tokens):
        start, count = _name_at(text, tokens, index)
        if count:
            replacements.append((tokens[start].start, tokens[start + count - 1].end, NAME_PLACEHOLDER))
            index = start + count
        else:
            index += 1
    return apply_replacements(text, replacements)
