"""Layer 2 of taxonomy v2 (ADR D63): the deterministic, display-only role slot."""

from __future__ import annotations

from pathlib import Path

import pytest

from qorgan.explain.roles import (
    RoleLexiconError,
    detect_role,
    detect_roles,
    get_role_lexicon,
    load_role_lexicon,
    render_role_line,
    role_name,
)
from qorgan.taxonomy import get_taxonomy

REAL = Path("data/lexicon/roles.yaml")


def test_real_lexicon_has_the_nineteen_observed_roles():
    lexicon = load_role_lexicon(REAL)
    assert len(lexicon.roles) == 19
    assert len({r.id for r in lexicon.roles}) == 19


def test_every_role_maps_to_a_v2_impersonation_tactic_or_none():
    v2 = set(get_taxonomy().label_space("v2"))
    for role in load_role_lexicon(REAL).roles:
        assert role.imp_tactic is None or (role.imp_tactic in v2 and role.imp_tactic.startswith("imp_"))


def test_every_imp_tactic_has_at_least_one_role():
    imps = {t for t in get_taxonomy().label_space("v2") if t.startswith("imp_")}
    covered = {r.imp_tactic for r in load_role_lexicon(REAL).roles}
    assert imps <= covered


def test_every_role_is_named_in_every_locale():
    for role in load_role_lexicon(REAL).roles:
        for locale in ("ru", "kk", "en"):
            assert role_name(role.id, locale, load_role_lexicon(REAL)).strip()


def test_detects_the_claimed_role_as_a_verbatim_span():
    transcript = "Здравствуйте, вас беспокоит служба безопасности банка.\nУ вас подозрительная операция."
    claim = detect_role(transcript, load_role_lexicon(REAL))
    assert claim is not None and claim.role_id == "bank_security"
    assert transcript[claim.span.start : claim.span.end] == claim.span.text
    assert claim.span.text.lower() == "служба безопасности банка"


def test_kazakh_and_english_claims():
    lexicon = load_role_lexicon(REAL)
    assert detect_role("Сәлеметсіз бе, мен ҰҚК қызметкерімін.", lexicon).role_id == "knb"
    assert detect_role("Hello, I am calling from Kaspi about your card.", lexicon).role_id == "bank_security"


def test_earliest_claim_wins_and_all_claims_are_listed():
    transcript = "Это курьер, вам посылка.\nСейчас вас соединят, беспокоит следователь."
    lexicon = load_role_lexicon(REAL)
    assert detect_role(transcript, lexicon).role_id == "courier_post"
    assert [c.role_id for c in detect_roles(transcript, lexicon)] == ["courier_post", "police"]


def test_a_longer_overlapping_claim_beats_a_shorter_one():
    lexicon = load_role_lexicon(REAL)
    # "national bank" (regulator) contains no bank_security cue; "антифрод-центр" is longer than
    # nothing else here -- and "служба безопасности банка" must not be taken by the regulator.
    claim = detect_role("Вас беспокоит антифрод-центр Национального банка.", lexicon)
    assert claim.role_id == "regulator"


def test_no_claim_returns_none():
    assert detect_role("Привет, как дела? Вечером увидимся.", load_role_lexicon(REAL)) is None


def test_render_role_line_fills_role_and_span():
    line = render_role_line("Звонящий представился {role}: «{span}».", "сотрудником банка", "служба безопасности банка")
    assert line == "Звонящий представился сотрудником банка: «служба безопасности банка»."


def test_duplicate_role_ids_rejected(tmp_path):
    bad = tmp_path / "roles.yaml"
    bad.write_text(
        "version: 1\nroles:\n"
        "  - {id: a, imp_tactic: null, names: {ru: x, kk: y, en: z}, cues: [aaa]}\n"
        "  - {id: a, imp_tactic: null, names: {ru: x, kk: y, en: z}, cues: [bbb]}\n",
        encoding="utf-8",
    )
    with pytest.raises(RoleLexiconError):
        load_role_lexicon(bad)


def test_missing_locale_name_rejected(tmp_path):
    bad = tmp_path / "roles.yaml"
    bad.write_text(
        "version: 1\nroles:\n  - {id: a, imp_tactic: null, names: {ru: x, kk: y}, cues: [aaa]}\n",
        encoding="utf-8",
    )
    with pytest.raises(RoleLexiconError):
        load_role_lexicon(bad)


def test_unknown_imp_tactic_rejected(tmp_path):
    bad = tmp_path / "roles.yaml"
    bad.write_text(
        "version: 1\nroles:\n  - {id: a, imp_tactic: imp_nope, names: {ru: x, kk: y, en: z}, cues: [aaa]}\n",
        encoding="utf-8",
    )
    with pytest.raises(RoleLexiconError):
        load_role_lexicon(bad)


def test_role_lexicon_is_not_part_of_the_trained_bundle():
    """Display-only (ADR D63): the served weights hash only the cue + reassurance lexicons."""
    import json

    weights = json.loads(Path("site/models/weights.json").read_text(encoding="utf-8"))
    assert set(weights["lexicon"]) == {"cue_lexicon_hash", "cues", "reassurance", "reassurance_hash"}


def test_get_role_lexicon_reads_the_configured_path():
    assert get_role_lexicon().roles == load_role_lexicon(REAL).roles


# ---------------------------------------------------------------- self-introduction by organisation (D63 phase 2)


@pytest.mark.parametrize(
    ("transcript", "role", "span"),
    [
        ("Алло, это Асель из Kaspi, удобно говорить?", "bank_security", "это Асель из Kaspi"),
        ("Сәлеметсіз бе, Halyk банктен Ерлан.", "bank_security", "Halyk банктен"),
        ("Hello, this is John from Beeline, quick question.", "telecom_operator", "this is John from Beeline"),
        ("Алло, Tele2-ден звоним снова.", "telecom_operator", "Tele2-ден"),
        ("Добрый день, курьер Kaspi доставки, Нурлан.", "courier_post", "курьер Kaspi доставки"),
        ("Қарағандының №3 емханасынан, мейірбике Ақмарал.", "clinic", "емханасынан"),
        ("Hello, Beeline customer care here.", "telecom_operator", "Beeline customer care"),
    ],
)
def test_self_introduction_by_organisation(transcript, role, span):
    claim = detect_role(transcript, load_role_lexicon(REAL))
    assert claim is not None and claim.role_id == role, claim
    assert claim.span.text == span
    assert transcript[claim.span.start : claim.span.end] == span


def test_an_organisation_without_an_introduction_is_not_a_claim():
    lexicon = load_role_lexicon(REAL)
    assert detect_role("Я вчера оплатила свет через Kaspi, всё нормально.", lexicon) is None
    assert detect_role("Мен Каспий теңізіне барып келдім.", lexicon) is None  # «Каспий» is not «Каспи»


def test_the_introduction_cannot_cross_an_utterance_boundary():
    lexicon = load_role_lexicon(REAL)
    assert detect_role("Да, это я.\nKaspi мне вчера писал.", lexicon) is None


def test_the_window_is_bounded():
    lexicon = load_role_lexicon(REAL)
    far = "Это, ну как сказать, одна очень длинная история про Kaspi"
    assert detect_role(far, lexicon) is None


def test_longest_organisation_wins_between_roles():
    lexicon = load_role_lexicon(REAL)
    assert detect_role("Здравствуйте, это Kaspi доставка, ваш заказ приехал.", lexicon).role_id == "courier_post"


def test_lexicon_without_intro_section_still_loads(tmp_path):
    path = tmp_path / "roles.yaml"
    path.write_text(
        "version: 1\nroles:\n  - id: x\n    imp_tactic: null\n    names: {ru: a, kk: b, en: c}\n    cues: [abc]\n",
        encoding="utf-8",
    )
    lexicon = load_role_lexicon(path)
    assert lexicon.intro.markers == () and detect_role("abc", lexicon).role_id == "x"


def test_an_ablative_mid_sentence_is_a_transfer_not_an_introduction():
    lexicon = load_role_lexicon(REAL)
    assert detect_role("Ертең кешке дейін сол ақшаны бір досыңның атынан Jusan-нан аударасың.", lexicon) is None
