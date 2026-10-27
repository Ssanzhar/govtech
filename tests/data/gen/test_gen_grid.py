"""The taxonomy-v2 generation grid (ADR D63): exact quotas, determinism, label consistency."""

from __future__ import annotations

from pathlib import Path

import pytest

from qorgan.config import GEN_LABEL_SPACE
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



def _v2_schemes():
    """The base grid is a v2 grid (no `label_space`): it draws only v2-expressible schemes (D73)."""
    return get_schemes().within(get_taxonomy().label_space("v2"))

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
    assert set(stats["scheme"]) == set(_v2_schemes().ids())  # every scheme of the grid's space present
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
    assert set(spec["glossary"]["tactics"]) == set(get_taxonomy().label_space(GEN_LABEL_SPACE))
    assert "examples_ru" not in blob and "cues" not in blob


# ---------------------------------------------------------------- phase 2 grid fixes (ADR D63)


@pytest.fixture(scope="module")
def big():
    return build_manifest(load_grid_config(GRID), run="t2", size=1800, schemes=get_schemes(), taxonomy=get_taxonomy())


def test_persona_age_and_occupation_are_coherent(big):
    cfg = load_grid_config(GRID)
    for cell in big:
        allowed = cfg.occupation_age_bands.get(cell.persona.occupation)
        assert allowed is None or cell.persona.age_band in allowed, (cell.cell_id, cell.persona)
    pensioners = [c for c in big if c.persona.occupation == "pensioner"]
    assert pensioners and all(c.persona.age_band == "61+" for c in pensioners)


def test_optional_impersonation_tactics_agree_with_the_role(big):
    lexicon = get_role_lexicon()
    schemes = get_schemes()
    for cell in big:
        if cell.kind != "scam":
            continue
        core = set(schemes.get(cell.scheme).generation.core)
        own = lexicon.get(cell.role).imp_tactic if cell.role else None
        for tactic in set(cell.tactics) - core:
            if tactic.startswith("imp_"):
                assert tactic == own, (cell.cell_id, cell.role, tactic)


def test_hook_only_cells_never_require_an_extraction_tactic(big):
    cfg = load_grid_config(GRID)
    hook_only = [c for c in big if c.stage == "hook_only"]
    assert hook_only
    for cell in hook_only:
        assert set(cell.tactics) <= set(cfg.hook_compatible_tactics), (cell.cell_id, cell.tactics)


def test_scam_share_is_the_corpus_base_rate(big):
    stats = manifest_stats(big)
    assert 0.45 <= stats["kind"]["scam"] / 1800 <= 0.50


def test_every_scheme_gets_twin_lookalike_negatives_in_proportion(big):
    schemes = _v2_schemes()
    cfg = load_grid_config(GRID)
    scams = {s: sum(1 for c in big if c.scheme == s) for s in schemes.ids()}
    looks = {s: [c for c in big if c.lookalike_of == s] for s in schemes.ids()}
    n_neg = sum(1 for c in big if c.kind == "legit")
    n_twin = sum(len(v) for v in looks.values())
    assert n_twin == round(n_neg * cfg.twin_share)
    for scheme_id, cells in looks.items():
        assert cells, scheme_id
        assert all(c.kind == "legit" and c.negative_class == schemes.get(scheme_id).generation.twin for c in cells)
        expected = n_twin * scams[scheme_id] / sum(scams.values())
        assert abs(len(cells) - expected) <= 1, (scheme_id, len(cells), expected)
    assert all(c.lookalike_of is None for c in big if c.kind == "scam")


def test_lookalike_negatives_keep_a_legitimate_role(big):
    cfg = load_grid_config(GRID)
    for cell in big:
        if cell.kind == "legit":
            assert cell.role in cfg.negative_roles[cell.negative_class], cell.cell_id
            assert cell.role not in {"regulator", "antifraud_service"}  # these never call legitimately


def test_pilot_manifest_still_reads():
    """The pilot manifest was written by grid v1 (no `lookalike_of`); it must stay readable."""
    pilot_path = Path("data/synthetic/gen/pilot/manifest.jsonl")
    if not pilot_path.exists():
        pytest.skip("pilot is gitignored and only exists on the machine that wrote it")
    cells = read_manifest(pilot_path)
    assert len(cells) == 120 and all(c.lookalike_of is None for c in cells)


def test_lookalike_prefers_the_impersonated_role_when_it_is_legitimate(big):
    schemes = get_schemes()
    cfg = load_grid_config(GRID)
    for cell in big:
        if cell.lookalike_of:
            shared = [r for r in cfg.negative_roles[cell.negative_class] if r in schemes.get(cell.lookalike_of).generation.roles]
            if shared:
                assert cell.role in shared, cell.cell_id
