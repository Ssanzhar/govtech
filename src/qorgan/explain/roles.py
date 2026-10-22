"""Layer 2 of taxonomy v2 (ADR D63): who the caller claims to be.

A deterministic lexicon lookup (`data/lexicon/roles.yaml`) -- no learning, no model input, no
FPR exposure. The result is a verbatim span of the transcript plus a role id, rendered with a
reviewed template («Звонящий представился {role}: «{span}»»). `site/core/role.js` is a 1:1
port pinned by `tests_js/fixtures/roles.json` (`scripts/export_role_fixture.py`).

Selection rule (both runtimes): for each role, the FIRST of its cues that `find_cue` matches
gives that role's cue span; a span overlapped by a strictly longer span of another role is
dropped; the remaining spans are ordered by start (ties: file order). `detect_role` returns
the first, `detect_roles` all of them (the v1 -> v2 label migration uses the full list).

Self-introduction by organisation (phase 2): «это Асель из Kaspi», «Halyk банктен Ерлан»,
"this is John from Beeline". Each line (an utterance; the boundary is a wall, as for cues) is
tokenised into runs of letters/digits (lower-cased, ё -> е). An organisation (`orgs` of a role;
tokens, `*` = prefix, otherwise exact or exact + a Kazakh ablative affix; a separate ablative
token right after it, «Tele2-ден», belongs to it) is a claim only when an introduction marker
(`intro.markers`) sits within `ROLE_INTRO_WINDOW` tokens of it in the same sentence (no
`ROLE_SENTENCE_END` character between them); the nearest marker gives the span (ties: the one
before). A bare ablative is not a claim (D63 v2.1: «Kaspi-ден бір нәрсе келді»).
Channel veto (D63 v2.1, `intro.channel`): the organisation is not the caller's role when
  - the token right before it is a `before` word («через Kaspi», "on Kaspi"),
  - the token right after it (after any ablative token) is an `after` word ("Kaspi support",
    «Kaspi арқылы»), or the token right after it is a separate non-ablative case affix
    («Kaspi-ге», «Kaspi-дің»), or
  - a `blockers` negation sits between a marker BEFORE it and it («Это не Kaspi»); a negation
    between it and a marker after it only drops that marker («Kaspi, не беспокойтесь»).
At each position the longest organisation wins (ties: file order), so «Kaspi доставка» is a
courier, not the bank. Per role, the earlier of its cue span and its first introduction span is
its candidate (ties: the longer). Names are never matched: only the organisation decides.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict

from qorgan.classifier.cue_match import UTTERANCE_BOUNDARY, find_cue
from qorgan.config import ROLE_INTRO_WINDOW, ROLE_SENTENCE_END, get_config
from qorgan.data.schema import Span

_LOCALES = ("ru", "kk", "en")
_SUPPORTED_VERSION = 1


class RoleLexiconError(ValueError):
    """The role lexicon is missing, malformed, or names an unknown impersonation tactic."""


_PREFIX = "*"
_TOKEN_CHAR = re.compile(r"[0-9a-zа-яәғқңөұүһі]")  # cue_match's alphabet after the ё -> е fold


@dataclass(frozen=True)
class RoleDefinition:
    id: str
    imp_tactic: str | None
    names: dict[str, str]
    cues: tuple[str, ...]
    orgs: tuple[tuple[str, ...], ...] = ()


@dataclass(frozen=True)
class ChannelSpec:
    """Words that make an organisation a channel or a help desk, not the caller (D63 v2.1).
    `before`/`after`/`blockers` are single-token patterns (`*` = prefix); `affixes` are
    separate case-affix tokens («Kaspi-ге»)."""

    before: tuple[str, ...] = ()
    after: tuple[str, ...] = ()
    affixes: tuple[str, ...] = ()
    blockers: tuple[str, ...] = ()


@dataclass(frozen=True)
class IntroSpec:
    """Self-introduction markers (token tuples, `*` = prefix on that token), the Kazakh
    ablative affixes, the token window between marker and organisation, the sentence-end
    characters that wall them apart, and the channel veto."""

    markers: tuple[tuple[str, ...], ...] = ()
    affixes: tuple[str, ...] = ()
    window: int = ROLE_INTRO_WINDOW
    sentence_end: str = ROLE_SENTENCE_END
    channel: ChannelSpec = ChannelSpec()


@dataclass(frozen=True)
class RoleLexicon:
    version: int
    roles: tuple[RoleDefinition, ...]
    intro: IntroSpec = IntroSpec()

    def get(self, role_id: str) -> RoleDefinition:
        for role in self.roles:
            if role.id == role_id:
                return role
        raise KeyError(f"Unknown role id: {role_id!r}")


class RoleClaim(BaseModel):
    """One claimed role, grounded as a verbatim span of the transcript."""

    model_config = ConfigDict(frozen=True)

    role_id: str
    span: Span


def load_role_lexicon(path: Path | None = None) -> RoleLexicon:
    resolved = path or get_config().role_lexicon_path
    if not resolved.exists():
        raise RoleLexiconError(f"Role lexicon not found: {resolved}")
    try:
        raw = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise RoleLexiconError(f"Invalid YAML in {resolved}: {exc}") from exc
    if not isinstance(raw, dict) or raw.get("version") != _SUPPORTED_VERSION:
        raise RoleLexiconError(f"{resolved}: expected a mapping with version {_SUPPORTED_VERSION}")
    roles = tuple(_parse_role(entry, resolved) for entry in raw.get("roles") or ())
    if not roles:
        raise RoleLexiconError(f"{resolved}: no roles defined")
    ids = [r.id for r in roles]
    if len(ids) != len(set(ids)):
        raise RoleLexiconError(f"{resolved}: duplicate role ids")
    return RoleLexicon(version=_SUPPORTED_VERSION, roles=roles, intro=_parse_intro(raw.get("intro") or {}, resolved))


def phrase_tokens(phrase: str) -> tuple[str, ...]:
    """A lexicon phrase as tokens; a trailing `*` on a word marks a prefix token."""
    out = []
    for word in str(phrase).split():
        star = word.endswith(_PREFIX)
        tokens = [t for t, _, _ in tokenize(word.rstrip(_PREFIX))]
        if tokens:
            out.extend(tokens[:-1] + [tokens[-1] + (_PREFIX if star else "")])
    return tuple(out)


def _parse_intro(raw: object, path: Path) -> IntroSpec:
    if not isinstance(raw, dict):
        raise RoleLexiconError(f"{path}: `intro` must be a mapping")
    markers = tuple(t for t in (phrase_tokens(m) for m in raw.get("markers") or ()) if t)
    return IntroSpec(
        markers=markers,
        affixes=_affix_list(raw.get("affixes")),
        window=ROLE_INTRO_WINDOW,
        sentence_end=ROLE_SENTENCE_END,
        channel=_parse_channel(raw.get("channel") or {}, path),
    )


def _affix_list(raw: object) -> tuple[str, ...]:
    return tuple(a for a in (str(x).strip().lower() for x in raw or ()) if a)


def _word_patterns(raw: object, field: str, path: Path) -> tuple[str, ...]:
    """Single-token patterns (`*` = prefix); a multi-token entry is a lexicon error."""
    patterns = []
    for word in raw or ():
        tokens = phrase_tokens(word)
        if len(tokens) != 1:
            raise RoleLexiconError(f"{path}: intro.channel.{field} entry {word!r} must be one token")
        patterns.append(tokens[0])
    return tuple(patterns)


def _parse_channel(raw: object, path: Path) -> ChannelSpec:
    if not isinstance(raw, dict):
        raise RoleLexiconError(f"{path}: `intro.channel` must be a mapping")
    return ChannelSpec(
        before=_word_patterns(raw.get("before"), "before", path),
        after=_word_patterns(raw.get("after"), "after", path),
        affixes=_affix_list(raw.get("affixes")),
        blockers=_word_patterns(raw.get("blockers"), "blockers", path),
    )


def _parse_role(entry: object, path: Path) -> RoleDefinition:
    from qorgan.taxonomy import get_taxonomy

    if not isinstance(entry, dict):
        raise RoleLexiconError(f"{path}: every role must be a mapping")
    role_id = str(entry.get("id") or "").strip()
    names = entry.get("names") or {}
    cues = tuple(str(c).strip() for c in entry.get("cues") or () if str(c).strip())
    imp_tactic = entry.get("imp_tactic")
    if not role_id or not cues:
        raise RoleLexiconError(f"{path}: role {role_id!r} needs an id and at least one cue")
    if not isinstance(names, dict) or any(not str(names.get(loc) or "").strip() for loc in _LOCALES):
        raise RoleLexiconError(f"{path}: role {role_id!r} must be named in {_LOCALES}")
    if imp_tactic is not None and imp_tactic not in get_taxonomy().all_tactic_ids():
        raise RoleLexiconError(f"{path}: role {role_id!r} maps to unknown tactic {imp_tactic!r}")
    orgs = tuple(t for t in (phrase_tokens(o) for o in entry.get("orgs") or ()) if t)
    return RoleDefinition(
        id=role_id, imp_tactic=imp_tactic, names={loc: str(names[loc]) for loc in _LOCALES}, cues=cues, orgs=orgs
    )


_LEXICON_CACHE: dict[tuple[str, int], RoleLexicon] = {}


def get_role_lexicon() -> RoleLexicon:
    """The configured lexicon, cached until the file changes (same pattern as the taxonomy)."""
    path = get_config().role_lexicon_path
    try:
        key = (str(path.resolve()), path.stat().st_mtime_ns)
    except OSError as exc:
        raise RoleLexiconError(f"Role lexicon not found: {path}") from exc
    if key not in _LEXICON_CACHE:
        _LEXICON_CACHE[key] = load_role_lexicon(path)
    return _LEXICON_CACHE[key]


def _first_match(transcript: str, cues: tuple[str, ...]) -> Span | None:
    for cue in cues:
        found = find_cue(transcript, cue)
        if found is not None:
            start, end = found
            return Span(text=transcript[start:end], start=start, end=end)
    return None


def _overlaps(a: Span, b: Span) -> bool:
    return a.start < b.end and b.start < a.end


def tokenize(line: str) -> list[tuple[str, int, int]]:
    """`(token, start, end)` runs of letters/digits, folded per character (lower, ё -> е).
    A character whose lower-case form is not exactly one alphabet character is a separator."""
    out: list[tuple[str, int, int]] = []
    buf: list[str] = []
    start = -1
    for index, character in enumerate(line):
        folded = character.lower()
        folded = "е" if folded == "ё" else folded
        if len(folded) == 1 and _TOKEN_CHAR.match(folded):
            if start < 0:
                start = index
            buf.append(folded)
        elif start >= 0:
            out.append(("".join(buf), start, index))
            buf, start = [], -1
    if start >= 0:
        out.append(("".join(buf), start, len(line)))
    return out


def _token_match(token: str, pattern: str, affixes: tuple[str, ...]) -> str | None:
    """The remainder of `token` after `pattern` when it matches, else None. A `*` pattern is
    a prefix; a plain pattern matches exactly or with one ablative affix glued on."""
    if pattern.endswith(_PREFIX):
        stem = pattern[:-1]
        return token[len(stem):] if token.startswith(stem) else None
    if token == pattern:
        return ""
    if token.startswith(pattern) and token[len(pattern):] in affixes:
        return token[len(pattern):]
    return None


def _seq_match(tokens, at: int, pattern: tuple[str, ...], affixes: tuple[str, ...]) -> str | None:
    """Remainder of the LAST token when `pattern` matches `tokens[at:]`, else None."""
    if at + len(pattern) > len(tokens):
        return None
    remainder = ""
    for offset, part in enumerate(pattern):
        remainder = _token_match(tokens[at + offset][0], part, affixes)
        if remainder is None:
            return None
    return remainder


def _org_at(tokens, at: int, lexicon: RoleLexicon) -> tuple[int, int, str] | None:
    """`(role order, organisation length, remainder)` of the longest organisation at `at`."""
    best: tuple[int, int, str] | None = None
    for order, role in enumerate(lexicon.roles):
        for org in role.orgs:
            remainder = _seq_match(tokens, at, org, lexicon.intro.affixes)
            if remainder is not None and (best is None or len(org) > best[1]):
                best = (order, len(org), remainder)
    return best


def _marker_ranges(tokens, intro: IntroSpec) -> list[tuple[int, int]]:
    ranges = []
    for at in range(len(tokens)):
        for marker in intro.markers:
            if _seq_match(tokens, at, marker, ()) is not None:
                ranges.append((at, at + len(marker) - 1))
    return ranges


def _is_word(tokens, at: int, patterns: tuple[str, ...]) -> bool:
    """True when tokens[at] exists and matches one single-token pattern (`*` = prefix)."""
    return 0 <= at < len(tokens) and any(_token_match(tokens[at][0], p, ()) is not None for p in patterns)


def _org_end(tokens, last: int, remainder: str, intro: IntroSpec) -> int:
    """Last token of the organisation: a separate ablative right after it belongs to it."""
    if not remainder and last + 1 < len(tokens) and tokens[last + 1][0] in intro.affixes:
        return last + 1
    return last


def _is_channel(tokens, first: int, last: int, end: int, channel: ChannelSpec) -> bool:
    """The organisation at tokens[first..last] (ablative-extended to `end`) is a channel."""
    if _is_word(tokens, first - 1, channel.before) or _is_word(tokens, end + 1, channel.after):
        return True
    return end == last and last + 1 < len(tokens) and tokens[last + 1][0] in channel.affixes


def _blocked(tokens, lo: int, hi: int, channel: ChannelSpec) -> bool:
    """A negation among tokens[lo..hi]."""
    return any(_is_word(tokens, at, channel.blockers) for at in range(lo, hi + 1))


def _same_sentence(line: str, tokens, left: int, right: int, intro: IntroSpec) -> bool:
    """No sentence end between the end of tokens[left] and the start of tokens[right]."""
    gap = line[tokens[left][2] : tokens[right][1]]
    return not any(c in intro.sentence_end for c in gap)


def _intro_span(line: str, tokens, first: int, last: int, remainder: str, markers, intro: IntroSpec) -> tuple[int, int] | None:
    """Token range of the claim for an organisation at tokens[first..last], or None."""
    end = _org_end(tokens, last, remainder, intro)
    if _is_channel(tokens, first, last, end, intro.channel):
        return None
    best: tuple[int, int, int, int] | None = None  # (gap, after?, start, end)
    for m_start, m_end in markers:
        if m_end < first:
            gap, after, lo, hi = first - m_end - 1, 0, m_end, first
        elif m_start > end:
            gap, after, lo, hi = m_start - end - 1, 1, end, m_start
        else:
            continue
        if gap > intro.window or not _same_sentence(line, tokens, lo, hi, intro):
            continue
        if _blocked(tokens, lo + 1, hi - 1, intro.channel):
            if not after:
                return None  # «Это не Kaspi»: the organisation is denied
            continue  # «Kaspi, не беспокойтесь»: only this marker is dropped
        if best is None or (gap, after) < best[:2]:
            best = (gap, after, min(first, m_start), max(end, m_end))
    return None if best is None else (best[2], best[3])


def intro_claims(transcript: str, lexicon: RoleLexicon) -> dict[int, Span]:
    """{role order: earliest self-introduction span} over every line of the transcript."""
    found: dict[int, Span] = {}
    if not lexicon.intro.markers:
        return found
    offset = 0
    for line in transcript.split(UTTERANCE_BOUNDARY):
        tokens = tokenize(line)
        markers = _marker_ranges(tokens, lexicon.intro)
        at = 0
        while at < len(tokens):
            org = _org_at(tokens, at, lexicon)
            if org is None:
                at += 1
                continue
            order, length, remainder = org
            span = _intro_span(line, tokens, at, at + length - 1, remainder, markers, lexicon.intro)
            if span is not None and order not in found:
                start, end = tokens[span[0]][1] + offset, tokens[span[1]][2] + offset
                found[order] = Span(text=transcript[start:end], start=start, end=end)
            at += length
        offset += len(line) + len(UTTERANCE_BOUNDARY)
    return found


def _earlier(a: Span | None, b: Span | None) -> Span | None:
    if a is None or b is None:
        return a or b
    return a if (a.start, -len(a.text)) <= (b.start, -len(b.text)) else b


def detect_roles(transcript: str, lexicon: RoleLexicon) -> tuple[RoleClaim, ...]:
    """Every non-dominated role claim in `transcript`, ordered by start then file order."""
    if not transcript or not transcript.strip():
        return ()
    intros = intro_claims(transcript, lexicon)
    candidates: list[tuple[int, RoleClaim]] = []
    for order, role in enumerate(lexicon.roles):
        span = _earlier(_first_match(transcript, role.cues), intros.get(order))
        if span is not None:
            candidates.append((order, RoleClaim(role_id=role.id, span=span)))
    kept = [
        (order, claim)
        for order, claim in candidates
        if not any(
            _overlaps(claim.span, other.span) and len(other.span.text) > len(claim.span.text)
            for _, other in candidates
        )
    ]
    kept.sort(key=lambda pair: (pair[1].span.start, pair[0]))
    return tuple(claim for _, claim in kept)


def detect_role(transcript: str, lexicon: RoleLexicon) -> RoleClaim | None:
    """The claim the caller made first, or `None`."""
    claims = detect_roles(transcript, lexicon)
    return claims[0] if claims else None


def role_name(role_id: str, locale: str, lexicon: RoleLexicon) -> str:
    if locale not in _LOCALES:
        raise RoleLexiconError(f"Unsupported locale {locale!r}")
    return lexicon.get(role_id).names[locale]


def render_role_line(template: str, role: str, span: str) -> str:
    """Fill a reviewed `role_template` (placeholders `{role}`, `{span}`)."""
    return template.replace("{role}", role).replace("{span}", span)
