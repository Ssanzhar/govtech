"""TDD tests for `qorgan.taxonomy` — load + validate `tactics.yaml`."""

import pytest

from qorgan.config import get_config
from qorgan.taxonomy import TaxonomyError, get_taxonomy, load_taxonomy

REAL_TAXONOMY_PATH = get_config().taxonomy_path


def test_load_real_taxonomy_has_expected_tactic_count():
    taxonomy = load_taxonomy(REAL_TAXONOMY_PATH)
    assert len(taxonomy.tactics) == 15


def test_hard_signal_ids_match_expected_set():
    taxonomy = load_taxonomy(REAL_TAXONOMY_PATH)
    assert set(taxonomy.hard_signal_ids()) == {
        "secrecy",
        "otp_request",
        "credentials_request",
        "safe_account",
        "remote_access",
    }


def test_negative_ids_and_hard_negative_ids():
    taxonomy = load_taxonomy(REAL_TAXONOMY_PATH)
    assert len(taxonomy.negative_ids()) == 5
    assert set(taxonomy.hard_negative_ids()) == {
        "legit_bank_call",
        "family_money_request",
        "legit_gov_service",
    }


def test_display_name_ru_and_kk():
    taxonomy = load_taxonomy(REAL_TAXONOMY_PATH)
    assert taxonomy.display_name("otp_request", "ru") == taxonomy.get("otp_request").ru
    assert taxonomy.display_name("otp_request", "kk") == taxonomy.get("otp_request").kk
    assert taxonomy.display_name("otp_request", "ru") != taxonomy.display_name("otp_request", "kk")


def test_display_name_unsupported_locale_raises():
    taxonomy = load_taxonomy(REAL_TAXONOMY_PATH)
    with pytest.raises(TaxonomyError):
        taxonomy.display_name("otp_request", "en")


def test_get_unknown_tactic_raises_keyerror():
    taxonomy = load_taxonomy(REAL_TAXONOMY_PATH)
    with pytest.raises(KeyError):
        taxonomy.get("does_not_exist")


def test_get_taxonomy_matches_load_taxonomy_default():
    assert get_taxonomy().tactic_ids() == load_taxonomy(REAL_TAXONOMY_PATH).tactic_ids()


def test_tactic_examples_are_tuples_of_strings():
    taxonomy = load_taxonomy(REAL_TAXONOMY_PATH)
    tactic = taxonomy.get("otp_request")
    assert isinstance(tactic.examples_ru, tuple)
    assert all(isinstance(example, str) for example in tactic.examples_ru)
    assert len(tactic.examples_ru) >= 1
    assert len(tactic.examples_kk) >= 1


def test_missing_file_raises_taxonomy_error(tmp_path):
    with pytest.raises(TaxonomyError):
        load_taxonomy(tmp_path / "nope.yaml")


def test_duplicate_tactic_id_raises(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        """
version: 1
tactics:
  - id: dup
    ru: a
    kk: a
    description: a
    hard_signal: false
  - id: dup
    ru: b
    kk: b
    description: b
    hard_signal: false
negatives: []
""",
        encoding="utf-8",
    )
    with pytest.raises(TaxonomyError):
        load_taxonomy(bad)


def test_duplicate_negative_id_raises(tmp_path):
    bad = tmp_path / "bad_neg.yaml"
    bad.write_text(
        """
version: 1
tactics:
  - id: t1
    ru: a
    kk: a
    description: a
    hard_signal: false
negatives:
  - id: dup_neg
    note: a
    hard_negative: false
  - id: dup_neg
    note: b
    hard_negative: true
""",
        encoding="utf-8",
    )
    with pytest.raises(TaxonomyError):
        load_taxonomy(bad)


def test_missing_required_field_raises(tmp_path):
    bad = tmp_path / "bad2.yaml"
    bad.write_text(
        """
version: 1
tactics:
  - id: only_id
negatives: []
""",
        encoding="utf-8",
    )
    with pytest.raises(TaxonomyError):
        load_taxonomy(bad)


def test_blank_field_raises(tmp_path):
    bad = tmp_path / "bad3.yaml"
    bad.write_text(
        """
version: 1
tactics:
  - id: t1
    ru: "   "
    kk: a
    description: a
    hard_signal: false
negatives: []
""",
        encoding="utf-8",
    )
    with pytest.raises(TaxonomyError):
        load_taxonomy(bad)


def test_no_tactics_raises(tmp_path):
    bad = tmp_path / "bad4.yaml"
    bad.write_text("version: 1\ntactics: []\nnegatives: []\n", encoding="utf-8")
    with pytest.raises(TaxonomyError):
        load_taxonomy(bad)


def test_unsupported_version_raises(tmp_path):
    bad = tmp_path / "bad5.yaml"
    bad.write_text(
        """
version: 2
tactics:
  - id: t1
    ru: a
    kk: a
    description: a
    hard_signal: false
negatives: []
""",
        encoding="utf-8",
    )
    with pytest.raises(TaxonomyError):
        load_taxonomy(bad)


def test_non_mapping_yaml_raises(tmp_path):
    bad = tmp_path / "bad6.yaml"
    bad.write_text("- just\n- a\n- list\n", encoding="utf-8")
    with pytest.raises(TaxonomyError):
        load_taxonomy(bad)


def test_invalid_yaml_syntax_raises(tmp_path):
    bad = tmp_path / "bad7.yaml"
    bad.write_text("version: 1\ntactics: [unclosed\n", encoding="utf-8")
    with pytest.raises(TaxonomyError):
        load_taxonomy(bad)


def test_taxonomy_is_frozen():
    taxonomy = load_taxonomy(REAL_TAXONOMY_PATH)
    with pytest.raises(Exception):  # noqa: B017 - pydantic frozen-model assignment error
        taxonomy.version = 99
