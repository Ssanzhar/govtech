"""Tests for the hard-signal cue lexicon loader + hashing (Phase 1).

The lexicon is the offline knowledge that turns a transcript into the 5 hard-signal cue
features. These tests pin its validation (must cover exactly the taxonomy's hard-signal
tactics, no blanks) and the content hash (stable + order-independent) that the model bundle
will later use to guarantee train/inference feature consistency.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from qorgan.classifier.cue_lexicon import (
    CueLexicon,
    CueLexiconError,
    lexicon_hash,
    load_cue_lexicon,
)
from qorgan.config import get_config
from qorgan.taxonomy import get_taxonomy


def _write_lexicon(tmp_path: Path, entries: dict[str, list[str]], *, version: int = 1) -> Path:
    lines = [f"version: {version}", "cues:"]
    for tactic_id, cues in entries.items():
        lines.append(f"  {tactic_id}:")
        for cue in cues:
            lines.append(f'    - "{cue}"')
    path = tmp_path / "cues.yaml"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _full_entries() -> dict[str, list[str]]:
    return {tid: [f"cue for {tid}"] for tid in get_taxonomy().hard_signal_ids()}


def test_committed_lexicon_loads_and_covers_all_hard_signal_ids():
    lexicon = load_cue_lexicon(get_config().cue_lexicon_path)
    assert set(lexicon.entries) == set(get_taxonomy().hard_signal_ids())
    assert all(lexicon.entries[tid] for tid in lexicon.entries)


def test_missing_hard_signal_id_raises(tmp_path):
    entries = _full_entries()
    entries.pop(get_taxonomy().hard_signal_ids()[0])
    with pytest.raises(CueLexiconError):
        load_cue_lexicon(_write_lexicon(tmp_path, entries))


def test_unknown_tactic_id_raises(tmp_path):
    entries = _full_entries()
    entries["not_a_real_tactic"] = ["x"]
    with pytest.raises(CueLexiconError):
        load_cue_lexicon(_write_lexicon(tmp_path, entries))


def test_blank_cue_raises(tmp_path):
    entries = _full_entries()
    entries[get_taxonomy().hard_signal_ids()[0]] = ["   "]
    with pytest.raises(CueLexiconError):
        load_cue_lexicon(_write_lexicon(tmp_path, entries))


def test_missing_file_raises(tmp_path):
    with pytest.raises(CueLexiconError):
        load_cue_lexicon(tmp_path / "does_not_exist.yaml")


def test_lexicon_hash_is_stable_and_order_independent():
    ids = get_taxonomy().hard_signal_ids()
    a = CueLexicon(version=1, entries={tid: ("beta", "alpha") for tid in ids})
    b = CueLexicon(version=1, entries={tid: ("alpha", "beta") for tid in reversed(ids)})
    assert lexicon_hash(a) == lexicon_hash(a)  # deterministic
    assert lexicon_hash(a) == lexicon_hash(b)  # key + cue order independent


def test_lexicon_hash_changes_with_content():
    ids = get_taxonomy().hard_signal_ids()
    a = CueLexicon(version=1, entries={tid: ("alpha",) for tid in ids})
    b = CueLexicon(version=1, entries={tid: ("alpha", "gamma") for tid in ids})
    assert lexicon_hash(a) != lexicon_hash(b)
