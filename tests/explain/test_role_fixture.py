"""The committed JS role-slot fixture is what the Python implementation produces today."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from qorgan.data.schema import ScoreResult
from qorgan.explain.explainer import explain

FIXTURE = Path("tests_js/fixtures/roles.json")


def _exporter():
    spec = importlib.util.spec_from_file_location("export_role_fixture", "scripts/export_role_fixture.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.skipif(not Path("data/processed/train.jsonl").exists(), reason="corpus not bootstrapped")
def test_committed_role_fixture_is_current():
    assert json.loads(FIXTURE.read_text(encoding="utf-8")) == _exporter().build()


def test_explanation_carries_the_role_slot_and_leaves_the_reason_alone():
    transcript = "Здравствуйте, вас беспокоит служба безопасности банка."
    result = ScoreResult(risk=0.1, backend="linear", raw_confidence=0.9)
    explanation = explain(result, transcript, "ru")
    assert explanation.role_id == "bank_security"
    assert transcript[explanation.role_span.start : explanation.role_span.end] == explanation.role_span.text
    assert explanation.role_line == "Звонящий представился сотрудником банка: «служба безопасности банка»."
    assert explanation.reason == "Явных признаков мошенничества не обнаружено."
    kk = explain(result, transcript, "kk").role_line
    assert kk == "Қоңырау шалушы өзін банк қызметкері деп таныстырды: «служба безопасности банка»."


def test_no_claim_means_no_role_line():
    explanation = explain(ScoreResult(risk=0.1, backend="linear"), "Привет, как дела?", "ru")
    assert explanation.role_id is None and explanation.role_line is None


def test_every_shipped_locale_has_a_role_template():
    from qorgan.explain.templates import load_templates

    for locale in ("ru", "kk", "en"):
        template = load_templates(locale, Path("src/qorgan/explain")).role_template
        assert "{role}" in template and "{span}" in template
