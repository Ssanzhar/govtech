"""The deterministic v1 -> v2 label migration (ADR D63)."""

from __future__ import annotations

from qorgan.data.migrate_v2 import (
    REVIEW_NO_ROLE,
    REVIEW_OUTSIDE_FAMILY,
    map_v1_tag,
    migrate_dialogue,
    migrate_tag_ids,
    review_list,
    summarize,
)
from qorgan.data.schema import Dialogue, Label, TacticTag, Utterance


def _dialogue(text: str, tags: tuple[str, ...]) -> Dialogue:
    return Dialogue(
        id="d1",
        language="ru",
        utterances=(Utterance(speaker="caller", text=text),),
        label=Label(risk=0.9, tactic_tags=tuple(TacticTag(id=t, weight=0.8) for t in tags)),
    )


def test_kept_ids_map_to_themselves():
    assert map_v1_tag("otp_request", ["imp_bank"]).new_ids == ("otp_request",)


def test_bank_defaults_to_imp_bank_and_follows_a_regulator_role():
    assert map_v1_tag("impersonation_bank", ["imp_bank"]).new_ids == ("imp_bank",)
    assert map_v1_tag("impersonation_bank", ["imp_regulator"]).new_ids == ("imp_regulator",)


def test_gov_police_splits_by_role():
    assert map_v1_tag("impersonation_gov_police", ["imp_law_enforcement"]).new_ids == ("imp_law_enforcement",)
    assert map_v1_tag("impersonation_gov_police", ["imp_gov_service"]).new_ids == ("imp_gov_service",)
    both = map_v1_tag("impersonation_gov_police", ["imp_law_enforcement", "imp_regulator"])
    assert both.new_ids == ("imp_law_enforcement", "imp_regulator") and both.review_reason is None


def test_no_role_falls_back_to_default_and_is_flagged():
    mapping = map_v1_tag("impersonation_telecom_delivery", [])
    assert mapping.new_ids == ("imp_telecom_delivery_utility",)
    assert mapping.review_reason == REVIEW_NO_ROLE


def test_role_outside_the_family_is_flagged():
    mapping = map_v1_tag("impersonation_bank", ["imp_law_enforcement"])
    assert mapping.new_ids == ("imp_bank",) and mapping.review_reason == REVIEW_OUTSIDE_FAMILY


def test_migrate_tag_ids_dedups_and_keeps_order():
    assert migrate_tag_ids(("impersonation_bank", "otp_request", "impersonation_bank"), ["imp_bank"]) == ("imp_bank", "otp_request")


def test_migrate_dialogue_uses_the_role_lexicon_and_is_pure():
    original = _dialogue("Здравствуйте, вас беспокоит служба безопасности банка. Назовите код.", ("impersonation_bank", "otp_request"))
    migrated, record = migrate_dialogue(original)
    assert [t.id for t in migrated.label.tactic_tags] == ["imp_bank", "otp_request"]
    assert original.label.tactic_tags[0].id == "impersonation_bank"  # input untouched
    assert record.roles_found == ("bank_security",) and not record.needs_review


def test_review_list_and_summary_carry_no_text():
    _, record = migrate_dialogue(_dialogue("Алло, это по поводу вашего номера.", ("impersonation_telecom_delivery",)))
    rows = review_list([record])
    assert rows == [{"id": "d1", "mapped": {"impersonation_telecom_delivery": ["imp_telecom_delivery_utility"]}, "reasons": [REVIEW_NO_ROLE], "roles_found": []}]
    summary = summarize([record])
    assert summary["rows_needing_review"] == 1
    assert "Алло" not in str(rows) + str(summary)
