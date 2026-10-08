"""Layer 2 of taxonomy v2 (ADR D63): who the caller claims to be.

A deterministic lexicon lookup (`data/lexicon/roles.yaml`) -- no learning, no model input, no
FPR exposure. The result is a verbatim span of the transcript plus a role id, rendered with a
reviewed template («Звонящий представился {role}: «{span}»»). `site/core/role.js` is a 1:1
port pinned by `tests_js/fixtures/roles.json` (`scripts/export_role_fixture.py`).

Selection rule (both runtimes): for each role, the FIRST of its cues that `find_cue` matches
gives that role's span; a span overlapped by a strictly longer span of another role is
dropped; the remaining spans are ordered by start (ties: file order). `detect_role` returns
the first, `detect_roles` all of them (the v1 -> v2 label migration uses the full list).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict

from qorgan.classifier.cue_match import find_cue
from qorgan.config import get_config
from qorgan.data.schema import Span

_LOCALES = ("ru", "kk", "en")
_SUPPORTED_VERSION = 1


class RoleLexiconError(ValueError):
    """The role lexicon is missing, malformed, or names an unknown impersonation tactic."""


@dataclass(frozen=True)
class RoleDefinition:
    id: str
    imp_tactic: str | None
    names: dict[str, str]
    cues: tuple[str, ...]


@dataclass(frozen=True)
class RoleLexicon:
    version: int
    roles: tuple[RoleDefinition, ...]

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
    return RoleLexicon(version=_SUPPORTED_VERSION, roles=roles)


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
    return RoleDefinition(
        id=role_id, imp_tactic=imp_tactic, names={loc: str(names[loc]) for loc in _LOCALES}, cues=cues
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


def detect_roles(transcript: str, lexicon: RoleLexicon) -> tuple[RoleClaim, ...]:
    """Every non-dominated role claim in `transcript`, ordered by start then file order."""
    if not transcript or not transcript.strip():
        return ()
    candidates: list[tuple[int, RoleClaim]] = []
    for order, role in enumerate(lexicon.roles):
        span = _first_match(transcript, role.cues)
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
