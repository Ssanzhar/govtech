"""TDD tests for `qorgan.taxonomy` — load + validate `tactics.yaml`."""

import pytest
from pathlib import Path

from qorgan.config import get_config
from qorgan.taxonomy import TaxonomyError, get_taxonomy, load_taxonomy

REAL_TAXONOMY_PATH = get_config().taxonomy_path


V1_IDS = (
    "impersonation_bank", "impersonation_gov_police", "impersonation_telecom_delivery",
    "urgency", "fear_threat", "secrecy", "otp_request", "credentials_request", "safe_account",
    "payment_redirect", "remote_access", "prize_lottery", "investment_scam", "mule_recruitment",
    "verification_ploy",
)
V2_NEW_IDS = {
    "imp_bank", "imp_law_enforcement", "imp_gov_service", "imp_regulator",
    "imp_telecom_delivery_utility", "imp_personal",
    "loan_inducement", "cash_courier_handoff", "refund_request", "task_advance_fee",
    "callback_instruction",
}


def test_load_real_taxonomy_has_expected_tactic_count():
    taxonomy = load_taxonomy(REAL_TAXONOMY_PATH)
    assert len(taxonomy.tactics) == 26  # 15 v1 + 11 v2-only (ADR D63)
    assert taxonomy.version == 2


def test_active_label_space_is_still_the_shipped_v1_space_in_order():
    """The heads in site/models/weights.json were trained on v1; until the phase-2 retrain the
    classifier's label space must stay exactly that, in that order (ADR D63)."""
    taxonomy = load_taxonomy(REAL_TAXONOMY_PATH)
    assert taxonomy.active_label_space == "v1"
    assert taxonomy.tactic_ids() == V1_IDS
    assert taxonomy.label_space("v1") == V1_IDS


def test_shipped_weights_label_space_matches_the_active_space():
    import json

    weights = json.loads(Path("site/models/weights.json").read_text(encoding="utf-8"))
    assert tuple(weights["label_space"]) == load_taxonomy(REAL_TAXONOMY_PATH).tactic_ids()


def test_v2_label_space_replaces_the_three_impersonation_ids():
    taxonomy = load_taxonomy(REAL_TAXONOMY_PATH)
    v2 = taxonomy.label_space("v2")
    assert len(v2) == 23
    assert not any(t.startswith("impersonation_") for t in v2)
    assert V2_NEW_IDS <= set(v2)
    kept = set(V1_IDS) - {"impersonation_bank", "impersonation_gov_police", "impersonation_telecom_delivery"}
    assert kept <= set(v2)


def test_all_tactic_ids_cover_both_spaces():
    taxonomy = load_taxonomy(REAL_TAXONOMY_PATH)
    assert set(taxonomy.all_tactic_ids()) == set(V1_IDS) | V2_NEW_IDS


def test_retired_ids_name_their_v2_successors():
    taxonomy = load_taxonomy(REAL_TAXONOMY_PATH)
    assert taxonomy.get("impersonation_bank").superseded_by == ("imp_bank", "imp_regulator")
    for old in ("impersonation_bank", "impersonation_gov_police", "impersonation_telecom_delivery"):
        successors = taxonomy.get(old).superseded_by
        assert successors and all(s in taxonomy.label_space("v2") for s in successors)


def test_hard_signals_are_identical_in_both_spaces():
    """The cue-feature block is 5 columns in both spaces -- a v2 tactic is never a hard signal."""
    taxonomy = load_taxonomy(REAL_TAXONOMY_PATH)
    assert taxonomy.hard_signal_ids("v2") == taxonomy.hard_signal_ids("v1") == taxonomy.hard_signal_ids()


def test_every_tactic_in_every_space_has_three_distinct_names():
    taxonomy = load_taxonomy(REAL_TAXONOMY_PATH)
    for tactic_id in taxonomy.all_tactic_ids():
        names = {loc: taxonomy.display_name(tactic_id, loc) for loc in ("ru", "kk", "en")}
        assert len(set(names.values())) == 3, tactic_id


def test_active_tactics_are_the_active_space_definitions():
    taxonomy = load_taxonomy(REAL_TAXONOMY_PATH)
    assert tuple(t.id for t in taxonomy.active_tactics()) == taxonomy.tactic_ids()


def test_unknown_label_space_raises():
    taxonomy = load_taxonomy(REAL_TAXONOMY_PATH)
    with pytest.raises(TaxonomyError):
        taxonomy.label_space("v9")


def test_version_1_file_is_still_read_as_a_single_v1_space(tmp_path):
    legacy = tmp_path / "v1.yaml"
    legacy.write_text(
        "version: 1\ntactics:\n  - {id: t1, ru: a, kk: b, en: c, description: d, hard_signal: false}\nnegatives: []\n",
        encoding="utf-8",
    )
    taxonomy = load_taxonomy(legacy)
    assert taxonomy.tactic_ids() == ("t1",) and taxonomy.label_space("v1") == ("t1",)


def test_successor_must_exist(tmp_path):
    bad = tmp_path / "bad_successor.yaml"
    bad.write_text(
        "version: 2\nactive_label_space: v1\ntactics:\n"
        "  - {id: t1, label_spaces: [v1], superseded_by: [nope], ru: a, kk: b, en: c, description: d, hard_signal: false}\n",
        encoding="utf-8",
    )
    with pytest.raises(TaxonomyError):
        load_taxonomy(bad)


def test_active_space_must_be_populated(tmp_path):
    bad = tmp_path / "bad_active.yaml"
    bad.write_text(
        "version: 2\nactive_label_space: v2\ntactics:\n"
        "  - {id: t1, label_spaces: [v1], ru: a, kk: b, en: c, description: d, hard_signal: false}\n",
        encoding="utf-8",
    )
    with pytest.raises(TaxonomyError):
        load_taxonomy(bad)


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
    assert len(taxonomy.negative_ids()) == 11
    assert set(taxonomy.hard_negative_ids()) == {
        "legit_bank_call",
        "family_money_request",
        "legit_gov_service",
        # v2 guardrails (ADR D63)
        "legit_delivery_code",
        "legit_refund_request",
        "legit_employer_request",
        "legit_clinic_appointment",
        "legit_telecom_offer",
        "legit_utility_notice",
    }


def test_display_name_ru_and_kk():
    taxonomy = load_taxonomy(REAL_TAXONOMY_PATH)
    assert taxonomy.display_name("otp_request", "ru") == taxonomy.get("otp_request").ru
    assert taxonomy.display_name("otp_request", "kk") == taxonomy.get("otp_request").kk
    assert taxonomy.display_name("otp_request", "ru") != taxonomy.display_name("otp_request", "kk")


def test_display_name_unsupported_locale_raises():
    taxonomy = load_taxonomy(REAL_TAXONOMY_PATH)
    with pytest.raises(TaxonomyError):
        taxonomy.display_name("otp_request", "de")


def test_display_name_covers_every_supported_locale():
    """Each shipped locale has its own reviewed name -- English is no longer an alias for RU."""
    taxonomy = load_taxonomy(REAL_TAXONOMY_PATH)
    for tactic_id in taxonomy.tactic_ids():
        names = {loc: taxonomy.display_name(tactic_id, loc) for loc in ("ru", "kk", "en")}
        assert all(n.strip() for n in names.values())
        assert len(set(names.values())) == 3, f"{tactic_id} reuses a name across locales"


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
version: 3
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


def test_get_taxonomy_is_cached_per_path_and_invalidated_by_edits(tmp_path, monkeypatch):
    import os
    import shutil

    from qorgan import taxonomy as tx

    copy = tmp_path / "tactics.yaml"
    shutil.copy(Path("data/taxonomy/tactics.yaml"), copy)
    monkeypatch.setenv("QORGAN_TAXONOMY_PATH", str(copy))
    first = tx.get_taxonomy()
    assert tx.get_taxonomy() is first  # same file, same object
    copy.write_text(copy.read_text(encoding="utf-8"), encoding="utf-8")  # rewrite -> new mtime
    os.utime(copy, ns=(copy.stat().st_atime_ns, copy.stat().st_mtime_ns + 1_000_000))
    second = tx.get_taxonomy()
    assert second is not first and second.tactic_ids() == first.tactic_ids()
    monkeypatch.setenv("QORGAN_TAXONOMY_PATH", str(tmp_path / "missing.yaml"))
    with pytest.raises(tx.TaxonomyError):
        tx.get_taxonomy()
