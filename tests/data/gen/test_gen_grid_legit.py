"""Legit-only generation runs and per-class scenario briefs (ADR D63 v2.1, PLAN_2026-10 A2)."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest
import yaml

from qorgan.data.gen.grid import GridConfig, GridError, build_manifest, load_grid_config, manifest_stats
from qorgan.data.gen.spec import SCENARIO_INSTRUCTION, build_chunk_spec
from qorgan.explain.roles import get_role_lexicon
from qorgan.explain.schemes import get_schemes
from qorgan.taxonomy import get_taxonomy

BASE = Path("configs/gen_grid.yaml")
LEGIT = Path("configs/gen_grid_v21_legit.yaml")
SIZE = 300


def _cfg(**override) -> GridConfig:
    raw = yaml.safe_load(BASE.read_text(encoding="utf-8"))
    return GridConfig.model_validate({**raw, **override})


@pytest.fixture(scope="module")
def legit():
    return build_manifest(load_grid_config(LEGIT), run="v21_legit", size=SIZE, schemes=get_schemes(), taxonomy=get_taxonomy())


def test_scam_share_zero_builds_a_legit_only_run():
    cells = build_manifest(_cfg(scam_share=0.0, twin_share=0.0), run="z", size=40, schemes=get_schemes(), taxonomy=get_taxonomy())
    assert {c.kind for c in cells} == {"legit"} and len(cells) == 40


def test_scenario_lookalike_must_be_the_schemes_twin():
    bad = {"family_money_request": [{"brief": "a cousin asks to borrow", "lookalike_of": "bank_card_phish"}]}
    with pytest.raises(GridError):
        build_manifest(_cfg(scam_share=0.0, twin_share=0.0, negative_scenarios=bad), run="z", size=40,
                       schemes=get_schemes(), taxonomy=get_taxonomy())


def test_scenarios_for_unknown_classes_are_rejected():
    with pytest.raises(ValueError):
        _cfg(negative_scenarios={"not_a_class": [{"brief": "x"}]})


def test_base_grid_cells_carry_no_scenario():
    cells = build_manifest(load_grid_config(BASE), run="t", size=120, schemes=get_schemes(), taxonomy=get_taxonomy())
    assert all(c.scenario is None for c in cells)
    spec = build_chunk_spec(cells[:5], run="t", chunk="c", taxonomy=get_taxonomy(), schemes=get_schemes(), roles=get_role_lexicon())
    assert all("scenario" not in c for c in spec["cells"])
    assert SCENARIO_INSTRUCTION not in spec["instructions"]


def test_legit_run_composition(legit):
    stats = manifest_stats(legit)
    assert stats["kind"] == {"legit": SIZE}
    assert stats["language"] == {"kk": 135, "mixed": 75, "ru": 90}
    assert stats["negative_class"] == {
        "family_money_request": 120, "legit_bank_call": 120, "legit_clinic_appointment": 15,
        "legit_delivery_code": 15, "legit_gov_service": 15, "legit_utility_notice": 15,
    }
    assert 0.5 <= stats["confusable_negatives"] / SIZE <= 0.7


def test_every_legit_cell_has_a_scenario_spread_evenly(legit):
    cfg = load_grid_config(LEGIT)
    for cls, scenarios in cfg.negative_scenarios.items():
        counts = Counter(c.scenario for c in legit if c.negative_class == cls)
        assert set(counts) == {s.brief for s in scenarios}, cls
        assert max(counts.values()) - min(counts.values()) <= 1, cls


def test_scenarios_are_not_tied_to_one_register(legit):
    family = [c for c in legit if c.negative_class == "family_money_request"]
    for brief in {c.scenario for c in family}:
        assert len({c.language for c in family if c.scenario == brief}) >= 2, brief


def test_scenario_lookalikes_are_twins_with_a_legitimate_role(legit):
    schemes = get_schemes()
    cfg = load_grid_config(LEGIT)
    looks = [c for c in legit if c.lookalike_of]
    assert looks
    for cell in looks:
        assert schemes.get(cell.lookalike_of).generation.twin == cell.negative_class
        assert cell.role in cfg.negative_roles[cell.negative_class]


def test_legit_manifest_is_deterministic(legit):
    again = build_manifest(load_grid_config(LEGIT), run="v21_legit", size=SIZE, schemes=get_schemes(), taxonomy=get_taxonomy())
    assert again == legit


def test_legit_spec_carries_the_scenario_and_no_lexicon_phrase(legit):
    spec = build_chunk_spec(legit[:20], run="v21_legit", chunk="c", taxonomy=get_taxonomy(), schemes=get_schemes(), roles=get_role_lexicon())
    assert SCENARIO_INSTRUCTION in spec["instructions"]
    assert all(c["scenario"] for c in spec["cells"])
    briefs = " ".join(c["scenario"] for c in spec["cells"]).lower()
    assert not any("а" <= ch <= "я" or ch in "ёәғқңөұүһі" for ch in briefs)  # briefs are English: no ru/kk cue can leak
