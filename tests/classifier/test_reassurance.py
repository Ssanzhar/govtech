"""Tests for the reassurance feature (anti-scam NEGATIVE signal).

Core invariants: fires when a legit caller says sensitive data is NOT needed; NEVER fires on
a scam's demand for that same data (the reassurance terms are negation-of-need only); respects
the co-occurrence window.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from qorgan.classifier.reassurance import (
    ReassuranceError,
    ReassurancePatterns,
    load_reassurance_patterns,
    reassurance_scores,
    reassures,
)
from qorgan.config import get_config


def _patterns():
    return load_reassurance_patterns(get_config().reassurance_patterns_path)


def test_committed_patterns_load_and_validate():
    patterns = _patterns()
    assert patterns.sensitive_terms and patterns.reassurance_terms
    assert patterns.window_chars > 0


def test_missing_file_raises(tmp_path):
    with pytest.raises(ReassuranceError):
        load_reassurance_patterns(tmp_path / "nope.yaml")


def test_blank_and_empty_terms_raise():
    with pytest.raises(ValueError):
        ReassurancePatterns(version=1, sensitive_terms=("код",), reassurance_terms=("  ",))
    with pytest.raises(ValueError):
        ReassurancePatterns(version=1, sensitive_terms=(), reassurance_terms=("не нужно",))


def test_fires_on_reassurance_phrase():
    patterns = _patterns()
    assert reassures("Наш банк не запрашиваем код из SMS у клиентов.", patterns)
    assert reassures("Никаких данных карты называть не нужно.", patterns)


def test_does_not_fire_on_scam_demand():
    patterns = _patterns()
    # a scam DEMANDS the code -- no negation-of-need phrasing, so no reassurance
    assert not reassures("Продиктуйте код из SMS прямо сейчас, чтобы отменить операцию.", patterns)
    # secrecy phrasing ("никому не говорите") must NOT count as reassurance
    assert not reassures("Никому не говорите код из SMS, это секретная операция.", patterns)


def test_does_not_fire_on_bare_sensitive_mention():
    patterns = _patterns()
    assert not reassures("Ваш код подтверждения отправлен в приложение.", patterns)


def test_window_is_respected():
    near = ReassurancePatterns(
        version=1, sensitive_terms=("код",), reassurance_terms=("не нужно",), window_chars=6
    )
    assert reassures("код не нужно", near)
    far = "код" + (" x" * 20) + " не нужно"
    assert not reassures(far, near)


def test_reassurance_scores_shape_and_values():
    patterns = _patterns()
    scores = reassurance_scores(
        ["мы не запрашиваем код по телефону", "просто разговор о погоде"], patterns
    )
    assert scores.shape == (2, 1)
    assert scores[0, 0] == 1.0
    assert scores[1, 0] == 0.0
