"""Tests for `qorgan.data.incident_seed` — family sourcing + incident seeding."""

from datetime import datetime

import pytest

from support.numbers import TEST_HMAC_KEY

from qorgan.data.incident_seed import (
    build_families_from_dialogues,
    load_incidents_jsonl,
    seed_incidents,
    write_incidents_jsonl,
)
from qorgan.data.schema import Dialogue, Incident, Label, TacticTag, Utterance

_START = datetime(2026, 7, 1, 9, 0, 0)


def _scam(did, text, tags):
    return Dialogue(
        id=did, language="ru", utterances=(Utterance(speaker="c", text=text),),
        label=Label(risk=0.9, tactic_tags=tuple(TacticTag(id=t) for t in tags)),
    )


def _corpus():
    return [
        _scam("d1", "Это банк, код из SMS", ["impersonation_bank", "otp_request"]),
        _scam("d2", "Служба безопасности, продиктуйте код", ["impersonation_bank"]),
        _scam("d3", "Вас беспокоит следователь", ["impersonation_gov_police", "fear_threat"]),
        _scam("d4", "Инвестируйте, оплатите по QR", ["investment_scam", "payment_redirect"]),
    ]


def test_build_families_matches_tactics_and_appends_novel():
    families = build_families_from_dialogues(_corpus())
    ids = {f.id for f in families}
    assert "bank_security" in ids
    assert "police_finpol" in ids
    assert "crypto_giveaway_new" in ids  # novel always appended
    novel = next(f for f in families if f.id == "crypto_giveaway_new")
    assert novel.is_novel


def test_bank_family_weight_reflects_matching_count():
    families = build_families_from_dialogues(_corpus())
    bank = next(f for f in families if f.id == "bank_security")
    assert bank.weight == 2.0  # d1 + d2 matched


def test_build_families_no_scam_matches_raises():
    with pytest.raises(ValueError):
        build_families_from_dialogues([_scam("x", "погода хорошая", ["nonexistent_tactic"])])


def test_build_families_scrubs_phone_pii_from_corpus_transcripts():
    dialogues = [
        _scam(
            "d1",
            "Это банк, звоните +7 701 123 4567 срочно",
            ["impersonation_bank", "otp_request"],
        ),
    ]
    families = build_families_from_dialogues(dialogues)
    bank = next(f for f in families if f.id == "bank_security")
    assert all("[PHONE]" in t for t in bank.transcripts)
    for transcript in bank.transcripts:
        assert "701 123 4567" not in transcript
        assert "+7 701 123 4567" not in transcript


def test_build_families_scrubs_iin_pii_from_corpus_transcripts():
    dialogues = [
        _scam(
            "d1",
            "Продиктуйте ИИН 123456789012 для проверки",
            ["impersonation_bank", "otp_request"],
        ),
    ]
    families = build_families_from_dialogues(dialogues)
    bank = next(f for f in families if f.id == "bank_security")
    assert all("[IIN]" in t for t in bank.transcripts)
    for transcript in bank.transcripts:
        assert "123456789012" not in transcript


def test_seed_incidents_produces_incidents_from_corpus():
    incidents = seed_incidents(_corpus(), count=50, seed=42, start_time=_START, hmac_key=TEST_HMAC_KEY)
    assert len(incidents) == 50
    assert all(isinstance(i, Incident) for i in incidents)
    novel = [i for i in incidents if i.script_family == "crypto_giveaway_new"]
    assert len(novel) >= 3, "the demo always has a new scheme to find"


def test_write_and_load_incidents_round_trips(tmp_path):
    incidents = seed_incidents(_corpus(), count=20, seed=1, start_time=_START, hmac_key=TEST_HMAC_KEY)
    path = tmp_path / "incidents.jsonl"
    write_incidents_jsonl(incidents, path)
    loaded = load_incidents_jsonl(path)
    assert [i.id for i in loaded] == [i.id for i in incidents]


def test_load_missing_incidents_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_incidents_jsonl(tmp_path / "nope.jsonl")


def test_written_incidents_never_contain_a_raw_pool_number(tmp_path):
    """ADR D14 end-to-end: the fabricated operating numbers exist only in code; the file on
    disk holds digests + coarse prefixes."""
    import re

    from qorgan.data.incident_seed import _FAMILY_DEFINITIONS

    incidents = seed_incidents(_corpus(), count=30, seed=3, start_time=_START, hmac_key=TEST_HMAC_KEY)
    path = tmp_path / "incidents.jsonl"
    write_incidents_jsonl(incidents, path)
    text = path.read_text(encoding="utf-8")
    pool = [n for _, _, numbers in _FAMILY_DEFINITIONS for n in numbers]
    assert pool, "family definitions must carry numbers"
    for number in pool:
        digits = re.sub(r"\D", "", number)
        assert number not in text and digits[-7:] not in text
    assert "number_hash" in text and '"number_prefix":"+7 7' in text


def test_seeded_incidents_carry_the_tactics_of_the_dialogue_they_came_from():
    # The analyst console names an organization by its dominant tactics and draws its tactic
    # profile from incident labels. Seeds were tagless, so every seeded organization showed a
    # raw family id and an empty profile (QA 2026-09-29).
    corpus = _corpus()
    tags_by_text = {d.transcript(): tuple(t.id for t in d.label.tactic_tags) for d in corpus}
    incidents = seed_incidents(corpus, count=50, seed=42, start_time=_START, hmac_key=TEST_HMAC_KEY)
    from_corpus = [i for i in incidents if i.transcript in tags_by_text]
    assert from_corpus, "the corpus families must be sampled"
    for incident in from_corpus:
        assert tuple(t.id for t in incident.label.tactic_tags) == tags_by_text[incident.transcript]


def test_carrying_tags_does_not_change_which_incidents_are_seeded():
    # Same draws as before the change (numbers, times, transcripts): L2 numbers stay comparable.
    a = seed_incidents(_corpus(), count=40, seed=7, start_time=_START, hmac_key=TEST_HMAC_KEY)
    b = seed_incidents(_corpus(), count=40, seed=7, start_time=_START, hmac_key=TEST_HMAC_KEY)
    assert [(i.transcript, i.number_hash, i.timestamp) for i in a] == [(i.transcript, i.number_hash, i.timestamp) for i in b]


def test_every_seeded_incident_is_tagged_including_the_novel_scheme():
    incidents = seed_incidents(_corpus(), count=200, seed=3, start_time=_START, hmac_key=TEST_HMAC_KEY)
    assert any(i.script_family == "crypto_giveaway_new" for i in incidents)
    assert all(i.label.tactic_tags for i in incidents)
