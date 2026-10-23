"""Layer 3 of taxonomy v2 (ADR D63): the call's scheme, derived by rules -- never learned.

`data/taxonomy/schemes.yaml` lists ~20 schemes in priority order, each a small logical rule
over the call's v2 tactic tags and its Layer-2 role (`qorgan.explain.roles`). The first rule
that holds names the scheme; advice is keyed by scheme (`advice_{locale}.yaml::scheme_advice`).
The shipped model still emits v1 tags, so `migrate_tags_for_schemes` maps them through the
same deterministic migration the corpus uses before the rules run.

A rule node is exactly one of `all`, `any`, `tactic`, `any_tactic`, `role` (see the YAML
header). Unknown keys, unknown tactic ids and unknown role ids fail at load time.
"""

from __future__ import annotations

from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from qorgan.config import GEN_LABEL_SPACE, get_config

_LOCALES = ("ru", "kk", "en")
_NODE_KEYS = frozenset({"all", "any", "tactic", "any_tactic", "role"})
_ADVICE_DIR = Path(__file__).resolve().parent


class SchemeError(ValueError):
    """schemes.yaml is missing/malformed, or a rule names something that does not exist."""


@dataclass(frozen=True)
class GenerationSpec:
    core: tuple[str, ...]
    optional: tuple[str, ...]
    roles: tuple[str, ...]
    twin: str


@dataclass(frozen=True)
class Scheme:
    id: str
    names: dict[str, str]
    description: str
    rule: Mapping[str, Any]
    generation: GenerationSpec


@dataclass(frozen=True)
class SchemeSet:
    version: int
    schemes: tuple[Scheme, ...]

    def get(self, scheme_id: str) -> Scheme:
        for scheme in self.schemes:
            if scheme.id == scheme_id:
                return scheme
        raise KeyError(f"Unknown scheme id: {scheme_id!r}")

    def ids(self) -> tuple[str, ...]:
        return tuple(s.id for s in self.schemes)

    def within(self, tactic_ids: Collection[str]) -> "SchemeSet":
        """The schemes whose rule and generation spec use only `tactic_ids` (ADR D73: a v2 grid
        never draws a scheme that needs a v3 tactic), in the same priority order."""
        allowed = set(tactic_ids)

        def fits(scheme: Scheme) -> bool:
            tactics, _ = _referenced(scheme.rule)
            return (tactics | set(scheme.generation.core) | set(scheme.generation.optional)) <= allowed

        return SchemeSet(version=self.version, schemes=tuple(s for s in self.schemes if fits(s)))


def evaluate_rule(node: Mapping[str, Any], tags: Collection[str], role: str | None) -> bool:
    """Evaluate one rule node against a set of tactic ids and a role (or `None`)."""
    if not isinstance(node, Mapping) or len(node) != 1 or not set(node) <= _NODE_KEYS:
        raise SchemeError(f"a rule node must have exactly one of {sorted(_NODE_KEYS)}, got {node!r}")
    (key, value), = node.items()
    if key == "all":
        return all(evaluate_rule(child, tags, role) for child in value)
    if key == "any":
        return any(evaluate_rule(child, tags, role) for child in value)
    if key == "tactic":
        return value in tags
    if key == "any_tactic":
        return any(t in tags for t in value)
    return role is not None and role in value  # "role"


def _referenced(node: Mapping[str, Any]) -> tuple[set[str], set[str]]:
    """(tactic ids, role ids) a rule mentions."""
    (key, value), = node.items()
    if key in ("all", "any"):
        tactics: set[str] = set()
        roles: set[str] = set()
        for child in value:
            t, r = _referenced(child)
            tactics |= t
            roles |= r
        return tactics, roles
    if key == "tactic":
        return {value}, set()
    if key == "any_tactic":
        return set(value), set()
    return set(), set(value)


def load_schemes(path: Path | None = None) -> SchemeSet:
    from qorgan.explain.roles import get_role_lexicon
    from qorgan.taxonomy import get_taxonomy

    resolved = path or get_config().schemes_path
    if not resolved.exists():
        raise SchemeError(f"Schemes file not found: {resolved}")
    try:
        raw = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise SchemeError(f"Invalid YAML in {resolved}: {exc}") from exc
    if not isinstance(raw, dict) or raw.get("version") != 1 or not raw.get("schemes"):
        raise SchemeError(f"{resolved}: expected version 1 with a non-empty `schemes` list")
    taxonomy = get_taxonomy()
    space = set(taxonomy.label_space(GEN_LABEL_SPACE))
    roles = {r.id for r in get_role_lexicon().roles}
    schemes = tuple(_parse(entry, resolved, space, roles, set(taxonomy.negative_ids())) for entry in raw["schemes"])
    if len({s.id for s in schemes}) != len(schemes):
        raise SchemeError(f"{resolved}: duplicate scheme ids")
    return SchemeSet(version=1, schemes=schemes)


def _parse(entry: Mapping[str, Any], path: Path, v2: set[str], roles: set[str], negatives: set[str]) -> Scheme:
    scheme_id = str(entry.get("id") or "")
    names = entry.get("names") or {}
    if not scheme_id or any(not str(names.get(loc) or "").strip() for loc in _LOCALES):
        raise SchemeError(f"{path}: scheme {scheme_id!r} needs an id and a name in {_LOCALES}")
    rule = entry.get("rule")
    try:
        evaluate_rule(rule, set(), None)  # structural check
        tactics, rule_roles = _referenced(rule)
    except (SchemeError, TypeError, ValueError, AttributeError) as exc:
        raise SchemeError(f"{path}: scheme {scheme_id!r} has a malformed rule: {exc}") from exc
    gen = entry.get("generation") or {}
    spec = GenerationSpec(
        core=tuple(gen.get("core") or ()),
        optional=tuple(gen.get("optional") or ()),
        roles=tuple(gen.get("roles") or ()),
        twin=str(gen.get("twin") or ""),
    )
    unknown_tactics = (tactics | set(spec.core) | set(spec.optional)) - v2
    unknown_roles = (rule_roles | set(spec.roles)) - roles
    if unknown_tactics or unknown_roles or not spec.core or spec.twin not in negatives:
        raise SchemeError(
            f"{path}: scheme {scheme_id!r}: unknown tactics {sorted(unknown_tactics)}, unknown roles "
            f"{sorted(unknown_roles)}, empty core or unknown twin {spec.twin!r}"
        )
    return Scheme(
        id=scheme_id,
        names={loc: str(names[loc]) for loc in _LOCALES},
        description=str(entry.get("description") or ""),
        rule=rule,
        generation=spec,
    )


_CACHE: dict[tuple[str, int], SchemeSet] = {}


def get_schemes() -> SchemeSet:
    path = get_config().schemes_path
    key = (str(path.resolve()), path.stat().st_mtime_ns)
    if key not in _CACHE:
        _CACHE[key] = load_schemes(path)
    return _CACHE[key]


def migrate_tags_for_schemes(tag_ids: Iterable[str], role: str | None) -> tuple[str, ...]:
    """Map (possibly v1) tag ids to v2 using the call's role -- the corpus migration's rule."""
    from qorgan.data.migrate_v2 import migrate_tag_ids
    from qorgan.explain.roles import get_role_lexicon

    imps = [get_role_lexicon().get(role).imp_tactic] if role is not None else []
    return migrate_tag_ids(tag_ids, imps)


def derive_scheme(tag_ids: Sequence[str], role: str | None, schemes: SchemeSet | None = None) -> str | None:
    """The first scheme whose rule holds for these (v2) tags and role, else `None`."""
    active = schemes or get_schemes()
    tags = set(tag_ids)
    if not tags:
        return None
    for scheme in active.schemes:
        if evaluate_rule(scheme.rule, tags, role):
            return scheme.id
    return None


_ADVICE_CACHE: dict[str, dict[str, str]] = {}


def scheme_advice(scheme_id: str, locale: str) -> str:
    """Reviewed advice for a scheme (`advice_{locale}.yaml::scheme_advice`)."""
    if locale not in _LOCALES:
        raise SchemeError(f"Unsupported locale {locale!r}")
    if locale not in _ADVICE_CACHE:
        raw = yaml.safe_load((_ADVICE_DIR / f"advice_{locale}.yaml").read_text(encoding="utf-8")) or {}
        advice = raw.get("scheme_advice") or {}
        missing = set(get_schemes().ids()) - set(advice)
        if missing:
            raise SchemeError(f"advice_{locale}.yaml is missing scheme advice for {sorted(missing)}")
        _ADVICE_CACHE[locale] = {str(k): str(v) for k, v in advice.items()}
    return _ADVICE_CACHE[locale][scheme_id]
