"""TDD tests for `qorgan.explain.recommend` — the tactic→advice engine (design spec §10).

Deterministic, taxonomy-grounded, localized RU/KK, confidence-gated. Never free-form:
every advice string comes from `explain/advice_{locale}.yaml`.
"""

import pytest

from qorgan.data.schema import TacticTag
from qorgan.explain.recommend import (
    LOW_CONFIDENCE_FLOOR,
    AdviceError,
    Recommendation,
    load_advice,
    recommend,
)
from qorgan.taxonomy import get_taxonomy

# --- recommend(): tactic-specific advice ---------------------------------------------------


def test_otp_request_gets_specific_advice_in_russian():
    rec = recommend([TacticTag(id="otp_request", weight=0.9)], "ru")

    assert len(rec.advices) == 1
    assert "код" in rec.advices[0].lower()


def test_kk_locale_returns_kazakh_strings():
    rec_ru = recommend([TacticTag(id="otp_request", weight=0.9)], "ru")
    rec_kk = recommend([TacticTag(id="otp_request", weight=0.9)], "kk")

    assert rec_kk.advices != rec_ru.advices
    assert "код" in rec_kk.advices[0].lower()  # KK advice mentions the code too


def test_hard_signal_advice_ranks_before_contextual_advice():
    tags = [
        TacticTag(id="urgency", weight=0.95),  # contextual, higher weight
        TacticTag(id="otp_request", weight=0.7),  # hard signal, lower weight
    ]
    rec = recommend(tags, "ru")

    templates = load_advice("ru")
    assert rec.advices[0] == templates.tactic_advice["otp_request"]
    assert rec.advices[1] == templates.tactic_advice["urgency"]


def test_contextual_advice_ordered_by_weight():
    tags = [
        TacticTag(id="urgency", weight=0.4),
        TacticTag(id="impersonation_bank", weight=0.9),
    ]
    rec = recommend(tags, "ru")

    templates = load_advice("ru")
    assert rec.advices[0] == templates.tactic_advice["impersonation_bank"]


def test_duplicate_tags_do_not_duplicate_advice():
    tags = [TacticTag(id="urgency", weight=0.9), TacticTag(id="urgency", weight=0.5)]
    rec = recommend(tags, "ru")

    assert len(rec.advices) == 1


def test_unknown_tactic_id_is_ignored_not_fatal():
    tags = [TacticTag(id="future_tactic_v2", weight=0.9), TacticTag(id="urgency", weight=0.5)]
    rec = recommend(tags, "ru")

    assert len(rec.advices) == 1


def test_verification_questions_surface_alongside_advice():
    rec = recommend([TacticTag(id="impersonation_gov_police", weight=0.8)], "ru")

    assert len(rec.verification_questions) >= 3
    assert rec.softened is False
    assert rec.note is None


def test_no_tags_means_no_recommendations():
    rec = recommend([], "ru")

    assert rec == Recommendation(
        advices=(), verification_questions=(), softened=False, note=None
    )


# --- confidence gating ----------------------------------------------------------------------


def test_low_confidence_softens_to_verification_questions_only():
    rec = recommend(
        [TacticTag(id="impersonation_bank", weight=0.6)],
        "ru",
        confidence=LOW_CONFIDENCE_FLOOR - 0.05,
    )

    assert rec.softened is True
    assert rec.advices == ()
    assert rec.note is not None
    assert len(rec.verification_questions) >= 3


def test_confident_detection_is_not_softened():
    rec = recommend(
        [TacticTag(id="impersonation_bank", weight=0.6)],
        "ru",
        confidence=LOW_CONFIDENCE_FLOOR + 0.1,
    )

    assert rec.softened is False
    assert rec.advices != ()


def test_unknown_confidence_is_not_softened():
    rec = recommend([TacticTag(id="impersonation_bank", weight=0.6)], "ru", confidence=None)

    assert rec.softened is False


# --- validation / fail-loud -------------------------------------------------------------------


def test_unsupported_locale_raises():
    with pytest.raises(AdviceError):
        recommend([TacticTag(id="urgency", weight=0.5)], "en")


def test_missing_advice_file_raises(tmp_path):
    with pytest.raises(AdviceError):
        load_advice("ru", advice_dir=tmp_path)


def test_advice_file_missing_required_key_raises(tmp_path):
    (tmp_path / "advice_ru.yaml").write_text(
        "tactic_advice:\n  urgency: 'x'\n", encoding="utf-8"
    )
    with pytest.raises(AdviceError):
        load_advice("ru", advice_dir=tmp_path)


@pytest.mark.parametrize("locale", ["ru", "kk"])
def test_every_taxonomy_tactic_has_advice(locale):
    """Coverage invariant: a detected tactic must never lack localized advice."""
    templates = load_advice(locale)
    taxonomy = get_taxonomy()

    missing = set(taxonomy.tactic_ids()) - set(templates.tactic_advice)
    assert missing == set()


def test_recommendation_is_frozen():
    rec = recommend([TacticTag(id="urgency", weight=0.5)], "ru")
    with pytest.raises(Exception):
        rec.softened = True  # type: ignore[misc]
