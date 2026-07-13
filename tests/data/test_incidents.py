"""TDD tests for `qorgan.data.incidents` — deterministic Level-2 incident synthesis."""

from datetime import datetime, timedelta

import pytest

from qorgan.data.incidents import ScriptFamily, synthesize_incidents
from qorgan.data.schema import Incident

_START = datetime(2026, 7, 1, 9, 0, 0)

_FAMILIES = (
    ScriptFamily(
        id="bank_otp",
        transcripts=("Это банк, продиктуйте код из SMS", "Служба безопасности банка, код из СМС"),
        phone_numbers=("+7 700 111 22 33", "+7 700 111 44 55"),
        weight=3.0,
    ),
    ScriptFamily(
        id="police_threat",
        transcripts=("Вас беспокоит следователь, на вас дело", "Финпол, подтвердите личность"),
        phone_numbers=("+7 701 999 88 77",),
        weight=2.0,
    ),
    ScriptFamily(
        id="crypto_giveaway_new",
        transcripts=("Раздача криптовалюты, пришлите 0.01 BTC и получите вдвое",),
        phone_numbers=("+7 702 555 00 11",),
        is_novel=True,
        weight=1.0,
    ),
)


def test_synthesize_produces_requested_count_of_incidents():
    incidents = synthesize_incidents(_FAMILIES, count=60, seed=42, start_time=_START)
    assert len(incidents) == 60
    assert all(isinstance(i, Incident) for i in incidents)


def test_each_incident_belongs_to_a_family_with_matching_transcript_and_number():
    incidents = synthesize_incidents(_FAMILIES, count=60, seed=42, start_time=_START)
    by_id = {f.id: f for f in _FAMILIES}
    for inc in incidents:
        fam = by_id[inc.script_family]
        assert inc.transcript in fam.transcripts
        assert inc.phone_number in fam.phone_numbers
        assert inc.label.risk >= 0.5  # incidents are scams


def test_synthesis_is_deterministic_for_a_seed():
    a = synthesize_incidents(_FAMILIES, count=40, seed=7, start_time=_START)
    b = synthesize_incidents(_FAMILIES, count=40, seed=7, start_time=_START)
    assert [(i.id, i.script_family, i.phone_number, i.transcript) for i in a] == [
        (i.id, i.script_family, i.phone_number, i.transcript) for i in b
    ]


def test_different_seed_changes_assignment():
    a = synthesize_incidents(_FAMILIES, count=60, seed=1, start_time=_START)
    b = synthesize_incidents(_FAMILIES, count=60, seed=2, start_time=_START)
    assert [i.script_family for i in a] != [i.script_family for i in b]


def test_timestamps_within_window():
    incidents = synthesize_incidents(_FAMILIES, count=60, seed=42, start_time=_START, span_days=30.0)
    end = _START + timedelta(days=30.0)
    for inc in incidents:
        assert _START <= inc.timestamp <= end


def test_novel_family_is_represented_and_flagged_via_script_family():
    incidents = synthesize_incidents(_FAMILIES, count=120, seed=42, start_time=_START)
    novel = [i for i in incidents if i.script_family == "crypto_giveaway_new"]
    assert len(novel) >= 1


def test_larger_weight_gets_more_incidents():
    incidents = synthesize_incidents(_FAMILIES, count=300, seed=42, start_time=_START)
    counts = {f.id: 0 for f in _FAMILIES}
    for inc in incidents:
        counts[inc.script_family] += 1
    assert counts["bank_otp"] > counts["crypto_giveaway_new"]


def test_incident_ids_are_unique():
    incidents = synthesize_incidents(_FAMILIES, count=100, seed=42, start_time=_START)
    ids = [i.id for i in incidents]
    assert len(ids) == len(set(ids))


def test_empty_families_raises():
    with pytest.raises(ValueError):
        synthesize_incidents((), count=10, seed=1, start_time=_START)


def test_non_positive_count_raises():
    with pytest.raises(ValueError):
        synthesize_incidents(_FAMILIES, count=0, seed=1, start_time=_START)
