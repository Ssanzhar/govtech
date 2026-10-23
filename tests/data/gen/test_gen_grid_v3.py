"""Taxonomy v3 grids (ADR D73): a grid that declares `label_space: v3` draws the five v3 schemes
and their twins; `class_personas` pins the callee of a class (a child for `minor_targeted`)."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from qorgan.data.gen.grid import GridError, build_manifest, load_grid_config
from qorgan.explain.schemes import get_schemes
from qorgan.taxonomy import get_taxonomy

BASE = Path("configs/gen_grid.yaml")
V3_SCHEMES = {"minor_targeted", "blackmail", "survey_data_harvest", "romance_scam", "charity_fraud"}
CHILD = {"age_band": {"10-13": 0.4, "14-17": 0.6}, "occupation": ["school pupil"]}


def _v3_config(tmp_path, **overrides):
    raw = yaml.safe_load(BASE.read_text(encoding="utf-8"))
    raw["label_space"] = "v3"
    raw["negative_roles"] = {**raw["negative_roles"], "legit_child_call": ["relative_friend"], "legit_debt_notice": ["bank_security"],
                             "legit_survey": ["pollster"], "legit_partner_call": [None], "legit_charity_call": ["charity_fund"]}
    raw.update(overrides)
    path = tmp_path / "grid.yaml"
    path.write_text(yaml.safe_dump(raw, allow_unicode=True), encoding="utf-8")
    return load_grid_config(path)


def _build(cfg, size=600):
    return build_manifest(cfg, run="v3t", size=size, schemes=get_schemes(), taxonomy=get_taxonomy())


def test_a_v3_grid_draws_the_v3_schemes_and_their_twins(tmp_path):
    cells = _build(_v3_config(tmp_path))
    assert V3_SCHEMES <= {c.scheme for c in cells if c.kind == "scam"}
    twins = {c.negative_class for c in cells if c.kind == "legit" and c.lookalike_of in V3_SCHEMES}
    assert twins == {"legit_child_call", "legit_debt_notice", "legit_survey", "legit_partner_call", "legit_charity_call"}


def test_class_personas_pin_the_callee_and_leave_other_cells_alone(tmp_path):
    plain = _build(_v3_config(tmp_path))
    pinned = _build(_v3_config(tmp_path, class_personas={"minor_targeted": CHILD, "legit_child_call": CHILD}))
    for a, b in zip(plain, pinned):
        if b.scheme == "minor_targeted" or b.negative_class == "legit_child_call":
            assert b.persona.age_band in CHILD["age_band"] and b.persona.occupation == "school pupil"
        else:
            assert a == b  # same draws for every other cell
    assert any(b.scheme == "minor_targeted" for b in pinned)


def test_class_personas_must_name_a_scheme_or_negative_class(tmp_path):
    with pytest.raises(GridError):
        _build(_v3_config(tmp_path, class_personas={"no_such_class": CHILD}))
