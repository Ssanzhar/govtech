"""The generation grid: a deterministic manifest of cells, one per dialogue to write (ADR D63).

Exact quotas (largest remainder) for the axes that must hit a share -- scam vs negative,
scheme / negative class, register -- and seeded draws for the rest. Registers are spread
evenly inside every class by a smooth weighted round-robin, so a 4-dialogue scheme is not
four Kazakh calls by chance. A scam cell's tactics are its scheme's `core` plus 0-2 seeded
`optional` ones, kept only if the scheme rule still derives the same scheme (so the label the
generator is asked for is the label the rules would give).
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

    @model_validator(mode="after")
    def _kind_consistent(self) -> "GridCell":
        if self.kind == "scam" and (not self.scheme or self.negative_class or not self.tactics):
            raise ValueError(f"{self.cell_id}: a scam cell needs a scheme and tactics, no negative class")
        if self.kind == "legit" and (self.scheme or not self.negative_class or self.tactics):
            raise ValueError(f"{self.cell_id}: a negative cell needs a negative class and no tactics")
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


class GridConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    version: int
    seed: int
    registers: dict[str, float]
    scam_share: float = Field(gt=0.0, lt=1.0)
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

    @model_validator(mode="after")
    def _registers_known(self) -> "GridConfig":
        unknown = set(self.registers) - {"kk", "ru", "mixed", "en"}
        if unknown or not self.registers:
            raise ValueError(f"unknown registers {sorted(unknown)}")
        missing_roles = set(self.negative_weights) - set(self.negative_roles)
        if missing_roles:
            raise ValueError(f"negative classes without a caller role: {sorted(missing_roles)}")
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


def _scam_tactics(rng: random.Random, scheme, role: str | None, cfg: GridConfig, schemes) -> tuple[str, ...]:
    from qorgan.explain.schemes import derive_scheme

    core = tuple(scheme.generation.core)
    pool = [t for t in scheme.generation.optional if t not in core]
    for _ in range(_OPTIONAL_RETRIES):
        k = rng.randint(cfg.optional_tactics.min, min(cfg.optional_tactics.max, len(pool)))
        candidate = core + tuple(rng.sample(pool, k))
        if derive_scheme(candidate, role, schemes) == scheme.id:
            return candidate
    return core


def build_manifest(cfg: GridConfig, *, run: str, size: int, schemes, taxonomy) -> tuple[GridCell, ...]:
    """The run's cells, deterministic in (cfg, run, size). Scams first, then negatives,
    each block ordered by class; registers interleaved evenly across the whole sequence."""
    if size <= 0:
        raise GridError("size must be positive")
    rng = random.Random(f"{cfg.seed}:{run}:{size}")
    n_scam = round(size * cfg.scam_share)
    scheme_counts = allocate(n_scam, {s.id: cfg.scheme_weights.get(s.id, 1.0) for s in schemes.schemes})
    unknown = set(cfg.negative_weights) - set(taxonomy.negative_ids())
    if unknown:
        raise GridError(f"unknown negative classes in grid: {sorted(unknown)}")
    negative_counts = allocate(size - n_scam, cfg.negative_weights)
    slots: list[tuple[Kind, str]] = [("scam", k) for k, n in scheme_counts.items() for _ in range(n)]
    slots += [("legit", k) for k, n in negative_counts.items() for _ in range(n)]
    registers = smooth_round_robin(allocate(size, cfg.registers))
    hard = set(taxonomy.hard_negative_ids())
    cells: list[GridCell] = []
    for index, ((kind, cls), register) in enumerate(zip(slots, registers)):
        persona = Persona(
            age_band=str(_draw(rng, cfg.persona.age_band)),
            region=rng.choice(cfg.persona.region),
            occupation=rng.choice(cfg.persona.occupation),
        )
        common = dict(
            cell_id=f"{run}_{index:04d}", run=run, kind=kind, language=register, persona=persona,
            outcome=str(_draw(rng, cfg.outcomes)), channel=str(_draw(rng, cfg.channels)),
            length_target=int(_draw(rng, cfg.length_targets)),
        )
        if kind == "scam":
            scheme = schemes.get(cls)
            role = rng.choice(scheme.generation.roles) if scheme.generation.roles else None
            cells.append(GridCell(
                **common, scheme=cls, role=role, stage=str(_draw(rng, cfg.stages)),
                tactics=_scam_tactics(rng, scheme, role, cfg, schemes),
                phrasing="indirect" if rng.random() < cfg.indirect_scam_share else "direct",
            ))
        else:
            is_hard = cls in hard
            cells.append(GridCell(
                **common, negative_class=cls, hard_negative=is_hard, role=rng.choice(cfg.negative_roles[cls]),
                stage="none", confusable=is_hard and rng.random() < cfg.confusable_negative_share,
            ))
    return tuple(cells)


def write_manifest(cells: Sequence[GridCell], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(c.model_dump_json() + "\n" for c in cells), encoding="utf-8")
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
    }


def to_json(cells: Sequence[GridCell]) -> list[dict]:
    return [json.loads(c.model_dump_json()) for c in cells]
