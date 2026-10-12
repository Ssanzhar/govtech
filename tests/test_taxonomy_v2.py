"""Label-space override for the taxonomy-v2 dry run (ADR D63 phase 2)."""

from __future__ import annotations


def test_label_space_override_switches_the_active_space_for_the_process(monkeypatch):
    """QORGAN_LABEL_SPACE (ADR D63 phase 2): the v2 dry run trains on v2 while the yaml stays v1."""
    from qorgan.taxonomy import TaxonomyError, get_taxonomy

    assert get_taxonomy().active_label_space == "v1"
    monkeypatch.setenv("QORGAN_LABEL_SPACE", "v2")
    taxonomy = get_taxonomy()
    assert taxonomy.active_label_space == "v2" and taxonomy.tactic_ids() == taxonomy.label_space("v2")
    monkeypatch.setenv("QORGAN_LABEL_SPACE", "v9")
    import pytest

    with pytest.raises(TaxonomyError):
        get_taxonomy()
    monkeypatch.delenv("QORGAN_LABEL_SPACE")
    assert get_taxonomy().active_label_space == "v1"
