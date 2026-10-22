"""Scam-only runs, per-scheme scam scenario briefs and per-cell constraints (ADR D63 v2.1, A2)."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest
import yaml

from qorgan.data.gen.grid import (
    GridConfig,
    GridError,
    build_manifest,
    load_grid_config,
    manifest_line,
    manifest_stats,
    read_manifest,
    with_constraint,
    write_manifest,
)
from qorgan.data.gen.spec import (
    CONSTRAINT_INSTRUCTION,
    SCAM_SCENARIO_INSTRUCTION,
    SCENARIO_INSTRUCTION,
    build_chunk_spec,
)
from qorgan.explain.roles import get_role_lexicon
from qorgan.explain.schemes import get_schemes
from qorgan.taxonomy import get_taxonomy

BASE = Path("configs/gen_grid.yaml")
SCAM = Path("configs/gen_grid_v21_scam.yaml")
SIZE = 236
BRIEF = {"brief": "The caller says early that they never ask for codes.", "schemes": ["parcel_code", "bank_card_phish"], "cells": 5}


def _cfg(**override) -> GridConfig:
    raw = yaml.safe_load(BASE.read_text(encoding="utf-8"))
    return GridConfig.model_validate({**raw, **override})


def _build(cfg: GridConfig, size: int = 120, run: str = "z"):
    return build_manifest(cfg, run=run, size=size, schemes=get_schemes(), taxonomy=get_taxonomy())


def _spec(cells):
    return build_chunk_spec(cells, run="z", chunk="c", taxonomy=get_taxonomy(), schemes=get_schemes(), roles=get_role_lexicon())


@pytest.fixture(scope="module")
def scam_run():
    return build_manifest(load_grid_config(SCAM), run="v21_scam", size=SIZE, schemes=get_schemes(), taxonomy=get_taxonomy())


def test_scam_share_one_builds_a_scam_only_run():
    cells = _build(_cfg(scam_share=1.0, twin_share=0.0), size=40)
    assert {c.kind for c in cells} == {"scam"} and len(cells) == 40


def test_scam_share_above_one_is_rejected():
    with pytest.raises(ValueError):
        _cfg(scam_share=1.01)


def test_scam_scenarios_go_to_exactly_n_cells_of_the_listed_schemes():
    cells = _build(_cfg(scam_scenarios=[BRIEF]))
    briefed = [c for c in cells if c.scenario]
    assert len(briefed) == 5
    assert {c.scheme for c in briefed} <= {"parcel_code", "bank_card_phish"}
    assert all(c.kind == "scam" for c in briefed)


def test_scam_scenarios_respect_the_stage_filter():
    cells = _build(_cfg(scam_scenarios=[{**BRIEF, "cells": 2, "stages": ["full"]}]))
    assert {c.stage for c in cells if c.scenario} == {"full"}


def test_scam_scenarios_leave_every_other_draw_unchanged():
    plain = _build(_cfg())
    briefed = _build(_cfg(scam_scenarios=[BRIEF]))
    assert [c.model_copy(update={"scenario": None}) for c in briefed] == list(plain)


def test_scam_scenarios_with_unknown_scheme_or_too_many_cells_fail():
    with pytest.raises(GridError):
        _build(_cfg(scam_scenarios=[{**BRIEF, "schemes": ["not_a_scheme"]}]))
    with pytest.raises(GridError):
        _build(_cfg(scam_scenarios=[{**BRIEF, "cells": 10_000}]))
    with pytest.raises(GridError):
        _build(_cfg(scam_scenarios=[{**BRIEF, "stages": ["not_a_stage"]}]))


def test_base_grid_has_no_scam_scenario_constraint_or_new_instruction():
    cells = _build(load_grid_config(BASE))
    assert all(c.scenario is None and c.constraint is None for c in cells)
    assert all('"constraint"' not in manifest_line(c) for c in cells)
    spec = _spec(cells[:10])
    assert SCAM_SCENARIO_INSTRUCTION not in spec["instructions"]
    assert CONSTRAINT_INSTRUCTION not in spec["instructions"]
    assert all("constraint" not in c for c in spec["cells"])


def test_scam_run_composition(scam_run):
    stats = manifest_stats(scam_run)
    assert stats["kind"] == {"scam": SIZE}
    assert stats["language"] == {"kk": 106, "mixed": 59, "ru": 71}
    assert stats["scheme"]["parcel_code"] > stats["scheme"]["bank_card_phish"]  # base scheme weights kept
    assert 0.4 <= stats["phrasing"]["indirect"] / SIZE <= 0.6


def test_scam_run_reassurance_cells(scam_run):
    briefed = [c for c in scam_run if c.scenario]
    assert 25 <= len(briefed) <= 30
    assert "hook_only" not in {c.stage for c in briefed}
    assert len({c.language for c in briefed}) == 3
    assert len({c.scheme for c in briefed}) >= 8
    assert max(Counter(c.scheme for c in briefed).values()) <= 4


def test_scam_spec_carries_the_scam_instruction_not_the_negative_one(scam_run):
    spec = _spec([c for c in scam_run if c.scenario][:5])
    assert SCAM_SCENARIO_INSTRUCTION in spec["instructions"]
    assert SCENARIO_INSTRUCTION not in spec["instructions"]
    briefs = " ".join(c["scenario"] for c in spec["cells"]).lower()
    assert not any("а" <= ch <= "я" or ch in "ёәғқңөұүһі" for ch in briefs)  # English only


def test_with_constraint_returns_new_cells_and_rejects_unknown_ids():
    cells = _build(_cfg(), size=20)
    out = with_constraint(cells, [cells[3].cell_id], "Do not say X.")
    assert out[3].constraint == "Do not say X." and cells[3].constraint is None
    assert out[:3] == cells[:3] and out[4:] == cells[4:]
    with pytest.raises(GridError):
        with_constraint(cells, ["nope"], "Do not say X.")
    with pytest.raises(GridError):
        with_constraint(cells, [cells[0].cell_id], "  ")


def test_constraint_reaches_the_spec_and_roundtrips_the_manifest(tmp_path):
    cells = with_constraint(_build(_cfg(), size=20), ["z_0001"], "Do not say X.")
    spec = _spec(cells[:3])
    assert CONSTRAINT_INSTRUCTION in spec["instructions"]
    assert [c.get("constraint") for c in spec["cells"]] == [None, "Do not say X.", None]
    assert read_manifest(write_manifest(cells, tmp_path / "m.jsonl")) == cells
