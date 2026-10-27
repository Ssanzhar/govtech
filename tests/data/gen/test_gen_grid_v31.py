"""The v3.1 grid (ADR D74): family and everyday hard negatives next to the personal scams they
must be told apart from, at a scam share that keeps the Claude train block at the Gemini rate."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from qorgan.data.gen.grid import build_manifest, load_grid_config
from qorgan.explain.schemes import get_schemes
from qorgan.taxonomy import get_taxonomy

GRID = Path("configs/gen_grid_v31_claude.yaml")
PERSONAL = {"relative_in_trouble", "romance_scam", "minor_targeted", "charity_fraud", "fake_boss",
            "marketplace_seller_phish", "investment_broker", "blackmail"}
FAMILY_EVERYDAY = {"family_money_request", "general_conversation", "legit_child_call", "legit_partner_call", "legit_charity_call"}


def _cells():
    return build_manifest(load_grid_config(GRID), run="v31_claude", size=800, schemes=get_schemes(), taxonomy=get_taxonomy())


def test_v31_draws_only_personal_scams_and_family_everyday_negatives():
    cells = _cells()
    assert {c.scheme for c in cells if c.kind == "scam"} == PERSONAL
    assert {c.negative_class for c in cells if c.kind == "legit"} == FAMILY_EVERYDAY
    negatives = Counter(c.negative_class for c in cells if c.kind == "legit")
    assert negatives["family_money_request"] + negatives["general_conversation"] >= 300


def test_v31_scam_share_keeps_the_style_balance_and_languages_are_kk_ru_mixed():
    cells = _cells()
    assert sum(c.kind == "scam" for c in cells) == 364
    assert set(Counter(c.language for c in cells)) == {"kk", "ru", "mixed"}


def test_v31_every_family_and_everyday_negative_has_a_brief_and_children_are_pupils():
    cells = _cells()
    for c in cells:
        if c.negative_class in {"family_money_request", "general_conversation"}:
            assert c.scenario
        if c.scheme == "minor_targeted" or c.negative_class == "legit_child_call":
            assert c.persona.occupation == "school pupil"


def test_v31_manifest_on_disk_is_the_grid_output():
    on_disk = Path("data/synthetic/gen/v31_claude/manifest.jsonl")
    if on_disk.exists():  # local-only data; the build is deterministic
        from qorgan.data.gen.grid import read_manifest

        assert read_manifest(on_disk) == _cells()
