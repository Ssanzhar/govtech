"""Layer 3 of taxonomy v2 (ADR D63): rule-derived schemes, advice keyed by scheme."""

from __future__ import annotations

from pathlib import Path

import pytest

from qorgan.explain.roles import load_role_lexicon
from qorgan.explain.schemes import (
    SchemeError,
    derive_scheme,
    evaluate_rule,
    load_schemes,
    migrate_tags_for_schemes,
    scheme_advice,
)
from qorgan.taxonomy import get_taxonomy

REAL = Path("data/taxonomy/schemes.yaml")


def test_about_twenty_schemes_with_unique_ids():
    schemes = load_schemes(REAL)
    assert 18 <= len(schemes.schemes) <= 24
    assert len({s.id for s in schemes.schemes}) == len(schemes.schemes)


@pytest.mark.parametrize("scheme_id", [s.id for s in load_schemes(REAL).schemes])
def test_generation_spec_derives_back_to_its_own_scheme(scheme_id):
    """A dialogue written to a scheme's core tactics + first role must be classified as that
    scheme -- otherwise the rule order or the spec is wrong (the generator would mislabel)."""
    schemes = load_schemes(REAL)
    scheme = schemes.get(scheme_id)
    role = scheme.generation.roles[0] if scheme.generation.roles else None
    assert derive_scheme(scheme.generation.core, role, schemes) == scheme_id


def test_every_referenced_id_exists():
    taxonomy = get_taxonomy()
    v2 = set(taxonomy.label_space("v2"))
    roles = {r.id for r in load_role_lexicon().roles}
    for scheme in load_schemes(REAL).schemes:
        assert set(scheme.generation.core) <= v2
        assert set(scheme.generation.optional) <= v2
        assert set(scheme.generation.roles) <= roles
        assert scheme.generation.twin in taxonomy.negative_ids()


def test_rule_language():
    tags = {"imp_bank", "safe_account"}
    assert evaluate_rule({"tactic": "imp_bank"}, tags, None)
    assert not evaluate_rule({"tactic": "otp_request"}, tags, None)
    assert evaluate_rule({"any_tactic": ["otp_request", "safe_account"]}, tags, None)
    assert evaluate_rule({"role": ["bank_security"]}, tags, "bank_security")
    assert not evaluate_rule({"role": ["bank_security"]}, tags, None)
    assert evaluate_rule({"all": [{"tactic": "imp_bank"}, {"any": [{"role": ["x"]}, {"tactic": "safe_account"}]}]}, tags, None)


def test_bad_rule_node_rejected():
    with pytest.raises(SchemeError):
        evaluate_rule({"tactic": "a", "role": ["b"]}, set(), None)
    with pytest.raises(SchemeError):
        evaluate_rule({"nope": 1}, set(), None)


def test_first_matching_scheme_wins_and_none_without_match():
    schemes = load_schemes(REAL)
    assert derive_scheme(("loan_inducement", "imp_bank", "safe_account"), "bank_security", schemes) == "mirror_loan"
    assert derive_scheme(("imp_bank", "safe_account"), None, schemes) == "bank_safe_account"
    assert derive_scheme(("urgency",), None, schemes) is None
    assert derive_scheme((), "bank_security", schemes) is None


def test_v1_tags_are_mapped_through_the_role_before_rules_apply():
    """The shipped model still emits v1 ids; the scheme layer reads them via the migration."""
    assert set(migrate_tags_for_schemes(("impersonation_bank", "otp_request"), "bank_security")) == {"imp_bank", "otp_request"}
    assert "imp_gov_service" in migrate_tags_for_schemes(("impersonation_gov_police",), "egov_con")
    assert derive_scheme(migrate_tags_for_schemes(("impersonation_bank", "safe_account"), None), None, load_schemes(REAL)) == "bank_safe_account"


def test_scheme_advice_exists_for_every_scheme_in_every_locale():
    schemes = load_schemes(REAL)
    for locale in ("ru", "kk", "en"):
        for scheme in schemes.schemes:
            assert scheme_advice(scheme.id, locale).strip()


def test_every_scheme_is_named_in_every_locale():
    for scheme in load_schemes(REAL).schemes:
        assert all(scheme.names[loc].strip() for loc in ("ru", "kk", "en"))


def test_unknown_tactic_in_rule_rejected(tmp_path):
    bad = tmp_path / "schemes.yaml"
    bad.write_text(
        "version: 1\nschemes:\n  - id: s\n    names: {ru: a, kk: b, en: c}\n    description: d\n"
        "    rule: {tactic: not_a_tactic}\n"
        "    generation: {core: [urgency], optional: [], roles: [], twin: legit_bank_call}\n",
        encoding="utf-8",
    )
    with pytest.raises(SchemeError):
        load_schemes(bad)
