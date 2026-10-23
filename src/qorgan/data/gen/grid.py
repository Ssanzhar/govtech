"""The generation grid: a deterministic manifest of cells, one per dialogue to write (ADR D63).

Exact quotas (largest remainder) for the axes that must hit a share -- scam vs negative,
scheme / negative class, register -- and seeded draws for the rest. Registers are spread
evenly inside every class by a smooth weighted round-robin, so a 4-dialogue scheme is not
four Kazakh calls by chance. A scam cell's tactics are its scheme's `core` plus 0-2 seeded
`optional` ones, kept only if the scheme rule still derives the same scheme (so the label the
generator is asked for is the label the rules would give).

Phase 2 (ADR D63) adds four constraints the pilot's writers tripped over or the style confound
requires: occupation x age coherence, optional `imp_*` tactics only when they agree with the
cell's role, `hook_only` only for cells whose tactics a hook can show, and twin-matched
look-alike negatives (`lookalike_of`) allocated in proportion to each scheme's scam count.
"""

from __future__ import annotations

import json
import random
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

Register = Literal["kk", "ru", "mixed", "en"]
Kind = Literal["scam", "legit"]
_OPTIONAL_RETRIES = 10


class GridError(ValueError):
    """The grid config is malformed or cannot produce the requested run."""


class Persona(BaseModel):
    model_config = ConfigDict(frozen=True)

    age_band: str
    region: str
    occupation: str


class GridCell(BaseModel):
    """One dialogue to write. `tactics` = the v2 tactics a scam MUST show (empty for negatives)."""

    model_config = ConfigDict(frozen=True)

    cell_id: str
    run: str
    kind: Kind
    scheme: str | None = None
    negative_class: str | None = None
    hard_negative: bool = False
    role: str | None = None
    language: Register
    tactics: tuple[str, ...] = ()
    persona: Persona
    outcome: str
    channel: str
    stage: str
    length_target: int = Field(ge=3)
    phrasing: Literal["direct", "indirect"] | None = None
    confusable: bool | None = None
    # Negatives only: the scheme whose pretext this legitimate call shares (none of its asks).
    lookalike_of: str | None = None
    # v2.1: an English brief of the concrete situation -- every negative of a class with
    # `negative_scenarios`, and the scam cells picked by a `scam_scenarios` entry.
    scenario: str | None = None
    # v2.1: an extra per-cell writer instruction (e.g. a line the dialogue must NOT contain), set
    # by `with_constraint` when a cell is regenerated; omitted from manifests and specs when None.
    constraint: str | None = None

    @model_validator(mode="after")
    def _kind_consistent(self) -> "GridCell":
        if self.kind == "scam" and (not self.scheme or self.negative_class or not self.tactics):
            raise ValueError(f"{self.cell_id}: a scam cell needs a scheme and tactics, no negative class")
        if self.kind == "legit" and (self.scheme or not self.negative_class or self.tactics):
            raise ValueError(f"{self.cell_id}: a negative cell needs a negative class and no tactics")
        if self.kind == "scam" and self.lookalike_of:
            raise ValueError(f"{self.cell_id}: only a negative can be a look-alike")
        return self


class RangeSpec(BaseModel):
    model_config = ConfigDict(frozen=True)
    min: int = Field(ge=0)
    max: int = Field(ge=0)


class PersonaSpec(BaseModel):
    model_config = ConfigDict(frozen=True)
    age_band: dict[str, float]
    region: tuple[str, ...]
    occupation: tuple[str, ...]


class Scenario(BaseModel):
    """One concrete situation for a negative class (v2.1). `lookalike_of` must be a scheme whose
    `twin` is that class; the brief is English (the writer never sees a ru/kk phrase)."""

    model_config = ConfigDict(frozen=True)
    brief: str = Field(min_length=1)
    lookalike_of: str | None = None


class ScamScenario(BaseModel):
    """A brief given to exactly `cells` scam cells of the listed `schemes` (v2.1). Cells are
    allocated over the schemes in proportion to their cell counts and picked with a separate
    seeded RNG, so a grid without scam scenarios draws exactly as before. English only.
    `stages` restricts the candidates (e.g. a move made "before the ask" needs a stage with an ask)."""

    model_config = ConfigDict(frozen=True)
    brief: str = Field(min_length=1)
    schemes: tuple[str, ...] = Field(min_length=1)
    cells: int = Field(ge=1)
    stages: tuple[str, ...] = ()  # only cells of these stages (empty = any stage)


class GridConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    version: int
    seed: int
    registers: dict[str, float]
    scam_share: float = Field(ge=0.0, le=1.0)  # 0 / 1 = a legit-only / scam-only run (v2.1, PLAN A2)
    scheme_weights: dict[str, float] = {}
    negative_weights: dict[str, float]
    negative_roles: dict[str, tuple[str | None, ...]]
    confusable_negative_share: float = Field(ge=0.0, le=1.0)
    indirect_scam_share: float = Field(ge=0.0, le=1.0)
    optional_tactics: RangeSpec
    length_targets: dict[int, float]
    outcomes: dict[str, float]
    channels: dict[str, float]
    stages: dict[str, float]
    persona: PersonaSpec
    twin_share: float = Field(default=0.0, ge=0.0, le=1.0)
    hook_compatible_tactics: tuple[str, ...] = ()
    occupation_age_bands: dict[str, tuple[str, ...]] = {}
    negative_scenarios: dict[str, tuple[Scenario, ...]] = {}
    scam_scenarios: tuple[ScamScenario, ...] = ()
    # The label space the run is written in (ADR D73). A grid draws only schemes expressible in it,
    # so every v2 grid (no key) still yields the manifest it always did.
    label_space: Literal["v2", "v3"] = "v2"

    @model_validator(mode="after")
    def _registers_known(self) -> "GridConfig":
        unknown = set(self.registers) - {"kk", "ru", "mixed", "en"}
        if unknown or not self.registers:
            raise ValueError(f"unknown registers {sorted(unknown)}")
        unknown_scenarios = {k for k, v in self.negative_scenarios.items() if k not in self.negative_weights or not v}
        if unknown_scenarios:
            raise ValueError(f"scenarios for classes not in negative_weights (or empty): {sorted(unknown_scenarios)}")
        missing_roles = set(self.negative_weights) - set(self.negative_roles)
        if missing_roles:
            raise ValueError(f"negative classes without a caller role: {sorted(missing_roles)}")
        unknown_occupations = set(self.occupation_age_bands) - set(self.persona.occupation)
        unknown_bands = {b for bands in self.occupation_age_bands.values() for b in bands} - set(self.persona.age_band)
        if unknown_occupations or unknown_bands:
            raise ValueError(f"occupation_age_bands names unknown occupations {sorted(unknown_occupations)} or bands {sorted(unknown_bands)}")
        for band in self.persona.age_band:
            if not any(band in self.occupation_age_bands.get(o, (band,)) for o in self.persona.occupation):
                raise ValueError(f"no occupation fits age band {band!r}")
        return self


def load_grid_config(path: Path) -> GridConfig:
    if not path.exists():
        raise GridError(f"grid config not found: {path}")
    try:
        return GridConfig.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
    except (yaml.YAMLError, ValueError) as exc:
        raise GridError(f"invalid grid config {path}: {exc}") from exc


def allocate(total: int, weights: Mapping[str, float]) -> dict[str, int]:
    """Largest-remainder apportionment of `total` over `weights` (ties: key order). Pure."""
    if total < 0 or not weights or any(w < 0 for w in weights.values()):
        raise GridError("allocate needs total >= 0 and non-negative weights")
    weight_sum = sum(weights.values())
    if weight_sum <= 0:
        raise GridError("weights must not all be zero")
    exact = {k: total * w / weight_sum for k, w in weights.items()}
    counts = {k: int(v) for k, v in exact.items()}
    order = sorted(weights, key=lambda k: (-(exact[k] - counts[k]), list(weights).index(k)))
    for key in order[: total - sum(counts.values())]:
        counts[key] += 1
    return counts


def smooth_round_robin(quotas: Mapping[str, int]) -> list[str]:
    """Interleave keys so each appears `quotas[k]` times as evenly as possible (nginx SWRR)."""
    total = sum(quotas.values())
    current = {k: 0 for k in quotas}
    out: list[str] = []
    for _ in range(total):
        for key, quota in quotas.items():
            current[key] += quota
        best = max(quotas, key=lambda k: (current[k], -list(quotas).index(k)))
        current[best] -= total
        out.append(best)
    return out


def _draw(rng: random.Random, weights: Mapping, ) -> object:
    keys = list(weights)
    return rng.choices(keys, weights=[weights[k] for k in keys], k=1)[0]


def _persona(rng: random.Random, cfg: GridConfig) -> Persona:
    """Age band first, then an occupation that fits it (phase-2 coherence fix)."""
    band = str(_draw(rng, cfg.persona.age_band))
    fitting = [o for o in cfg.persona.occupation if band in cfg.occupation_age_bands.get(o, (band,))]
    return Persona(age_band=band, region=rng.choice(cfg.persona.region), occupation=rng.choice(fitting))


def _stage(rng: random.Random, scheme, cfg: GridConfig) -> str:
    """A stage the scheme's core can show: `hook_only` needs every core tactic hook-compatible."""
    hookable = set(cfg.hook_compatible_tactics)
    allowed = {k: w for k, w in cfg.stages.items() if k != "hook_only" or set(scheme.generation.core) <= hookable}
    return str(_draw(rng, allowed))


def _optional_pool(scheme, role: str | None, stage: str, cfg: GridConfig, roles) -> list[str]:
    """Optional tactics that fit the cell: an `imp_*` only if it is the role's own impersonation
    tactic (phase-2 fix: no `imp_regulator` for a bank caller), and only hook-compatible tactics
    for a `hook_only` cell."""
    core = tuple(scheme.generation.core)
    own_imp = roles.get(role).imp_tactic if (role and roles is not None) else None
    pool = []
    for tactic in scheme.generation.optional:
        if tactic in core:
            continue
        if tactic.startswith("imp_") and tactic != own_imp:
            continue
        if stage == "hook_only" and tactic not in cfg.hook_compatible_tactics:
            continue
        pool.append(tactic)
    return pool


def _scam_tactics(rng: random.Random, scheme, role: str | None, cfg: GridConfig, schemes, pool: Sequence[str] | None = None) -> tuple[str, ...]:
    from qorgan.explain.schemes import derive_scheme

    core = tuple(scheme.generation.core)
    pool = list(pool) if pool is not None else [t for t in scheme.generation.optional if t not in core]
    for _ in range(_OPTIONAL_RETRIES):
        k = rng.randint(cfg.optional_tactics.min, min(cfg.optional_tactics.max, len(pool)))
        candidate = core + tuple(rng.sample(pool, k))
        if derive_scheme(candidate, role, schemes) == scheme.id:
            return candidate
    return core


def _negative_slots(n_neg: int, scheme_counts: Mapping[str, int], cfg: GridConfig, schemes) -> list[tuple[str, str | None]]:
    """`(negative class, lookalike_of)` per negative: `twin_share` of them allocated over the
    schemes by scam count (each the scheme's twin class), the rest by `negative_weights`."""
    n_twin = round(n_neg * cfg.twin_share)
    twin_counts = allocate(n_twin, {k: v for k, v in scheme_counts.items() if v}) if n_twin else {}
    slots: list[tuple[str, str | None]] = []
    for scheme_id, n in twin_counts.items():
        twin = schemes.get(scheme_id).generation.twin
        if twin not in cfg.negative_roles:
            raise GridError(f"scheme {scheme_id}'s twin {twin!r} has no caller role in the grid")
        slots += [(twin, scheme_id)] * n
    rest = allocate(n_neg - n_twin, cfg.negative_weights) if n_neg - n_twin else {}
    slots += [(cls, None) for cls, n in rest.items() for _ in range(n)]
    return sorted(slots, key=lambda s: (list(cfg.negative_roles).index(s[0]), s[1] or ""))


def _scenario_decks(slots, cfg: GridConfig, rng_key: str, schemes) -> dict[str, list]:
    """Per negative class with scenarios: one scenario per cell of that class, every scenario
    used evenly (largest remainder), in a seeded order that is independent of the register
    interleave. A separate RNG, so a grid without scenarios draws exactly as before."""
    decks: dict[str, list] = {}
    for cls, scenarios in cfg.negative_scenarios.items():
        for scenario in scenarios:
            if scenario.lookalike_of and schemes.get(scenario.lookalike_of).generation.twin != cls:
                raise GridError(f"scenario look-alike {scenario.lookalike_of!r} is not a twin of {cls!r}")
        n = sum(1 for kind, c, _ in slots if kind == "legit" and c == cls)
        counts = allocate(n, {str(i): 1.0 for i in range(len(scenarios))}) if n else {}
        deck = [scenarios[int(i)] for i, k in counts.items() for _ in range(k)]
        random.Random(f"{rng_key}:scenarios:{cls}").shuffle(deck)
        decks[cls] = deck
    return decks


def _apply_scam_scenarios(cells: Sequence[GridCell], cfg: GridConfig, rng_key: str, schemes) -> tuple[GridCell, ...]:
    """Give each `scam_scenarios` entry's brief to `cells` scam cells of its schemes (a cell gets
    at most one brief). New cells are returned; the input is not modified."""
    assigned: dict[str, str] = {}
    for index, entry in enumerate(cfg.scam_scenarios):
        for scheme_id in entry.schemes:
            try:
                schemes.get(scheme_id)
            except KeyError as exc:
                raise GridError(f"scam scenario names an unknown scheme {scheme_id!r}") from exc
        unknown_stages = set(entry.stages) - set(cfg.stages)
        if unknown_stages:
            raise GridError(f"scam scenario {index} names unknown stages {sorted(unknown_stages)}")
        by_scheme = {sid: [c for c in cells if c.kind == "scam" and c.scheme == sid and c.cell_id not in assigned
                           and (not entry.stages or c.stage in entry.stages)] for sid in entry.schemes}
        available = sum(len(v) for v in by_scheme.values())
        if entry.cells > available:
            raise GridError(f"scam scenario {index} wants {entry.cells} cells, the run has {available} of {list(entry.schemes)}")
        quotas = allocate(entry.cells, {sid: len(v) for sid, v in by_scheme.items() if v})
        rng = random.Random(f"{rng_key}:scam_scenarios:{index}")
        for sid, n in quotas.items():
            for cell in rng.sample(by_scheme[sid], n):
                assigned[cell.cell_id] = entry.brief
    return tuple(c.model_copy(update={"scenario": assigned[c.cell_id]}) if c.cell_id in assigned else c for c in cells)


def with_constraint(cells: Sequence[GridCell], cell_ids: Sequence[str], text: str) -> tuple[GridCell, ...]:
    """New cells with `constraint=text` on the named cells (a regeneration instruction); every
    other cell is returned unchanged. Unknown ids or a blank text raise `GridError`."""
    if not text.strip():
        raise GridError("a constraint must not be blank")
    wanted = set(cell_ids)
    unknown = wanted - {c.cell_id for c in cells}
    if unknown:
        raise GridError(f"unknown cell ids: {sorted(unknown)}")
    return tuple(c.model_copy(update={"constraint": text}) if c.cell_id in wanted else c for c in cells)


def build_manifest(cfg: GridConfig, *, run: str, size: int, schemes, taxonomy, roles=None) -> tuple[GridCell, ...]:
    """The run's cells, deterministic in (cfg, run, size). Scams first, then negatives,
    each block ordered by class; registers interleaved evenly across the whole sequence.
    `roles` (the role lexicon) enables the role-consistent optional-tactic filter."""
    if roles is None:
        from qorgan.explain.roles import get_role_lexicon

        roles = get_role_lexicon()
    if size <= 0:
        raise GridError("size must be positive")
    schemes = schemes.within(taxonomy.label_space(cfg.label_space))
    rng = random.Random(f"{cfg.seed}:{run}:{size}")
    n_scam = round(size * cfg.scam_share)
    scheme_counts = allocate(n_scam, {s.id: cfg.scheme_weights.get(s.id, 1.0) for s in schemes.schemes})
    unknown = set(cfg.negative_weights) - set(taxonomy.negative_ids())
    if unknown:
        raise GridError(f"unknown negative classes in grid: {sorted(unknown)}")
    slots: list[tuple[Kind, str, str | None]] = [("scam", k, None) for k, n in scheme_counts.items() for _ in range(n)]
    slots += [("legit", cls, look) for cls, look in _negative_slots(size - n_scam, scheme_counts, cfg, schemes)]
    registers = smooth_round_robin(allocate(size, cfg.registers))
    decks = _scenario_decks(slots, cfg, f"{cfg.seed}:{run}:{size}", schemes)
    hard = set(taxonomy.hard_negative_ids())
    cells: list[GridCell] = []
    for index, ((kind, cls, lookalike), register) in enumerate(zip(slots, registers)):
        common = dict(
            cell_id=f"{run}_{index:04d}", run=run, kind=kind, language=register, persona=_persona(rng, cfg),
            outcome=str(_draw(rng, cfg.outcomes)), channel=str(_draw(rng, cfg.channels)),
            length_target=int(_draw(rng, cfg.length_targets)),
        )
        if kind == "scam":
            scheme = schemes.get(cls)
            role = rng.choice(scheme.generation.roles) if scheme.generation.roles else None
            stage = _stage(rng, scheme, cfg)
            pool = _optional_pool(scheme, role, stage, cfg, roles)
            cells.append(GridCell(
                **common, scheme=cls, role=role, stage=stage,
                tactics=_scam_tactics(rng, scheme, role, cfg, schemes, pool),
                phrasing="indirect" if rng.random() < cfg.indirect_scam_share else "direct",
            ))
        else:
            is_hard = cls in hard
            scenario = decks[cls].pop() if decks.get(cls) else None
            if scenario is not None and scenario.lookalike_of and not lookalike:
                lookalike = scenario.lookalike_of
            legit_roles = list(cfg.negative_roles[cls])
            if lookalike:  # prefer the role the scheme impersonates, when it can call legitimately
                legit_roles = [r for r in legit_roles if r in schemes.get(lookalike).generation.roles] or legit_roles
            cells.append(GridCell(
                **common, negative_class=cls, hard_negative=is_hard, role=rng.choice(legit_roles),
                stage="none", confusable=is_hard and rng.random() < cfg.confusable_negative_share,
                lookalike_of=lookalike, scenario=scenario.brief if scenario is not None else None,
            ))
    if cfg.scam_scenarios:
        return _apply_scam_scenarios(cells, cfg, f"{cfg.seed}:{run}:{size}", schemes)
    return tuple(cells)


def manifest_line(cell: GridCell) -> str:
    """A cell's manifest JSON; `constraint` is left out when unset (earlier manifests unchanged)."""
    return cell.model_dump_json(exclude={"constraint"} if cell.constraint is None else None)


def write_manifest(cells: Sequence[GridCell], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(manifest_line(c) + "\n" for c in cells), encoding="utf-8")
    return path


def read_manifest(path: Path) -> tuple[GridCell, ...]:
    return tuple(GridCell.model_validate_json(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def manifest_stats(cells: Sequence[GridCell]) -> dict:
    """Counts per axis -- the numbers the QC report quotes for grid coverage."""
    def count(values) -> dict:
        out: dict = {}
        for v in values:
            out[str(v)] = out.get(str(v), 0) + 1
        return dict(sorted(out.items()))

    return {
        "size": len(cells),
        "kind": count(c.kind for c in cells),
        "language": count(c.language for c in cells),
        "scheme": count(c.scheme for c in cells if c.scheme),
        "negative_class": count(c.negative_class for c in cells if c.negative_class),
        "role": count(c.role for c in cells),
        "outcome": count(c.outcome for c in cells),
        "channel": count(c.channel for c in cells),
        "stage": count(c.stage for c in cells),
        "length_target": count(c.length_target for c in cells),
        "phrasing": count(c.phrasing for c in cells if c.phrasing),
        "confusable_negatives": sum(1 for c in cells if c.confusable),
        "lookalike_of": count(c.lookalike_of for c in cells if c.lookalike_of),
    }


def to_json(cells: Sequence[GridCell]) -> list[dict]:
    return [json.loads(c.model_dump_json()) for c in cells]
