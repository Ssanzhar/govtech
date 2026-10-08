"""The taxonomy-v2 generation grid (ADR D63): exact quotas, determinism, label consistency."""

from __future__ import annotations

from pathlib import Path

import pytest

from qorgan.data.gen.grid import (
    GridError,
    allocate,
    build_manifest,
    load_grid_config,
    manifest_stats,
    read_manifest,
    smooth_round_robin,
    write_manifest,
)
from qorgan.data.gen.spec import build_chunk_spec, chunk_cells
from qorgan.explain.roles import get_role_lexicon
from qorgan.explain.schemes import derive_scheme, get_schemes
from qorgan.taxonomy import get_taxonomy

GRID = Path("configs/gen_grid.yaml")


@pytest.fixture(scope="module")
def pilot():
    return build_manifest(load_grid_config(GRID), run="t", size=120, schemes=get_schemes(), taxonomy=get_taxonomy())


def test_allocate_is_exact_and_largest_remainder():
    assert allocate(10, {"a": 1, "b": 1, "c": 1}) == {"a": 4, "b": 3, "c": 3}
    assert sum(allocate(7, {"x": 0.3, "y": 0.7}).values()) == 7
    with pytest.raises(GridError):
        allocate(5, {"a": 0})


def test_smooth_round_robin_spreads_evenly():
    seq = smooth_round_robin({"a": 2, "b": 2})
    assert sorted(seq) == ["a", "a", "b", "b"] and seq[0] != seq[1]


def test_manifest_hits_the_adr_shares(pilot):
    stats = manifest_stats(pilot)
    assert stats["size"] == 120
    assert stats["kind"]["legit"] / 120 >= 0.35
    assert stats["language"] == {"en": 18, "kk": 36, "mixed": 30, "ru": 36}
    assert set(stats["scheme"]) == set(get_schemes().ids())  # every scheme present
    v2_negatives = {"legit_delivery_code", "legit_refund_request", "legit_employer_request",
                    "legit_clinic_appointment", "legit_telecom_offer", "legit_utility_notice"}
    assert v2_negatives <= set(stats["negative_class"])


def test_manifest_is_deterministic(pilot):
    again = build_manifest(load_grid_config(GRID), run="t", size=120, schemes=get_schemes(), taxonomy=get_taxonomy())
    assert again == pilot


def test_every_scam_cell_derives_its_own_scheme(pilot):
    for cell in pilot:
        if cell.kind == "scam":
            assert derive_scheme(cell.tactics, cell.role, get_schemes()) == cell.scheme, cell.cell_id


def test_negatives_have_no_tactics_and_a_role_from_the_config(pilot):
    cfg = load_grid_config(GRID)
    for cell in pilot:
        if cell.kind == "legit":
            assert not cell.tactics and cell.role in cfg.negative_roles[cell.negative_class]


def test_manifest_roundtrip(tmp_path, pilot):
    path = write_manifest(pilot, tmp_path / "m.jsonl")
    assert read_manifest(path) == pilot


def test_chunks_interleave_and_cover_every_cell(pilot):
    chunks = chunk_cells(pilot, 20)
    assert len(chunks) == 6 and sum(len(c) for _, c in chunks) == 120
    assert all({c.kind for c in cells} == {"scam", "legit"} for _, cells in chunks)


def _cyrillic(text: str) -> bool:
    return any("а" <= ch.lower() <= "я" or ch in "әғқңөұүһі" for ch in text)


def test_chunk_spec_never_carries_a_russian_or_kazakh_lexicon_phrase(pilot):
    """The glossary is English names + descriptions. No ru/kk cue, role cue or taxonomy example
    can reach the agent. (English definitions unavoidably share a few words with English cues --
    "safe account", "TeamViewer", role names -- measured and reported in the pilot QC, ADR D63.)"""
    spec = build_chunk_spec(pilot, run="t", chunk="c", taxonomy=get_taxonomy(), schemes=get_schemes(), roles=get_role_lexicon())
    blob = str(spec).lower()
    from qorgan.classifier.cue_lexicon import load_cue_lexicon

    phrases = [c for r in get_role_lexicon().roles for c in r.cues]
    phrases += [e for t in get_taxonomy().tactics for e in t.examples_ru + t.examples_kk]
    phrases += [c for cues in load_cue_lexicon().entries.values() for c in cues]
    leaked = [p for p in phrases if _cyrillic(p) and len(p) > 4 and p.lower() in blob]
    assert leaked == []
    assert set(spec["glossary"]["tactics"]) == set(get_taxonomy().label_space("v2"))
    assert "examples_ru" not in blob and "cues" not in blob
