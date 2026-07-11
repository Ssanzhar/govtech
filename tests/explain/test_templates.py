"""TDD tests for `qorgan.explain.templates` -- localized (RU/KK) explanation templates."""

from __future__ import annotations

from pathlib import Path

import pytest

from qorgan.explain.templates import (
    ExplanationTemplates,
    TemplateError,
    confidence_band,
    load_templates,
    render_caveat,
    render_reason,
)

_VALID_YAML = """
reason_template: "Tags: {tags}. Spans: {spans}."
no_signal_reason: "Nothing detected."
spans_placeholder: "-"
caveat: "The model can be wrong."
uncalibrated_suffix: " Confidence from {backend} is uncalibrated."
human_note: "A human always decides."
confidence:
  high: "High confidence."
  medium: "Medium confidence."
  low: "Low confidence."
  unknown: "Confidence unknown."
"""


def _write(tmp_path: Path, locale: str, content: str) -> Path:
    path = tmp_path / f"templates_{locale}.yaml"
    path.write_text(content, encoding="utf-8")
    return tmp_path


# --- load_templates ----------------------------------------------------------------------


def test_load_templates_valid_yaml_populates_all_fields(tmp_path):
    templates_dir = _write(tmp_path, "xx", _VALID_YAML)

    templates = load_templates("xx", templates_dir)

    assert isinstance(templates, ExplanationTemplates)
    assert templates.reason_template == "Tags: {tags}. Spans: {spans}."
    assert templates.no_signal_reason == "Nothing detected."
    assert templates.spans_placeholder == "-"
    assert templates.caveat == "The model can be wrong."
    assert templates.uncalibrated_suffix == " Confidence from {backend} is uncalibrated."
    assert templates.human_note == "A human always decides."
    assert templates.confidence == {
        "high": "High confidence.",
        "medium": "Medium confidence.",
        "low": "Low confidence.",
        "unknown": "Confidence unknown.",
    }


def test_load_templates_missing_file_raises(tmp_path):
    with pytest.raises(TemplateError):
        load_templates("does_not_exist", tmp_path)


def test_load_templates_missing_required_key_raises(tmp_path):
    content = _VALID_YAML.replace('human_note: "A human always decides."\n', "")
    templates_dir = _write(tmp_path, "xx", content)

    with pytest.raises(TemplateError):
        load_templates("xx", templates_dir)


def test_load_templates_confidence_missing_medium_raises(tmp_path):
    content = """
reason_template: "Tags: {tags}. Spans: {spans}."
no_signal_reason: "Nothing detected."
spans_placeholder: "-"
caveat: "The model can be wrong."
uncalibrated_suffix: " Confidence from {backend} is uncalibrated."
human_note: "A human always decides."
confidence:
  high: "High confidence."
  low: "Low confidence."
  unknown: "Confidence unknown."
"""
    templates_dir = _write(tmp_path, "xx", content)

    with pytest.raises(TemplateError):
        load_templates("xx", templates_dir)


def test_load_templates_not_a_mapping_raises(tmp_path):
    templates_dir = _write(tmp_path, "xx", "- a\n- b\n")
    with pytest.raises(TemplateError):
        load_templates("xx", templates_dir)


def test_load_templates_invalid_yaml_syntax_raises(tmp_path):
    templates_dir = _write(tmp_path, "xx", "reason_template: [unbalanced\n")
    with pytest.raises(TemplateError):
        load_templates("xx", templates_dir)


def test_load_templates_real_shipped_ru_file_loads():
    repo_root = Path(__file__).resolve().parents[2]
    templates_dir = repo_root / "src" / "qorgan" / "explain"

    templates = load_templates("ru", templates_dir)

    assert "{tags}" in templates.reason_template
    assert "{spans}" in templates.reason_template
    assert "{backend}" in templates.uncalibrated_suffix
    assert set(templates.confidence.keys()) == {"high", "medium", "low", "unknown"}


# --- render_reason -------------------------------------------------------------------------


@pytest.fixture
def templates(tmp_path) -> ExplanationTemplates:
    templates_dir = _write(tmp_path, "xx", _VALID_YAML)
    return load_templates("xx", templates_dir)


def test_render_reason_with_tags_and_phrases_wraps_phrase_in_guillemets(templates):
    reason = render_reason(templates, ["otp_request"], ["код из SMS"])
    assert "«код из SMS»" in reason
    assert "otp_request" in reason


def test_render_reason_with_tags_but_no_phrases_uses_placeholder(templates):
    reason = render_reason(templates, ["otp_request"], [])
    assert templates.spans_placeholder in reason
    assert "«" not in reason


def test_render_reason_with_no_tags_returns_no_signal_reason(templates):
    reason = render_reason(templates, [], ["irrelevant"])
    assert reason == templates.no_signal_reason


def test_render_reason_multiple_tags_and_phrases_joined(templates):
    reason = render_reason(templates, ["a", "b"], ["one", "two"])
    assert "a, b" in reason
    assert "«one»" in reason
    assert "«two»" in reason


# --- render_caveat -------------------------------------------------------------------------


def test_render_caveat_calibrated_has_no_suffix(templates):
    caveat = render_caveat(templates, backend="xlmr", is_calibrated=True)
    assert caveat == templates.caveat
    assert "xlmr" not in caveat


def test_render_caveat_uncalibrated_appends_suffix_with_backend(templates):
    caveat = render_caveat(templates, backend="llm", is_calibrated=False)
    assert caveat.startswith(templates.caveat)
    assert "llm" in caveat


# --- confidence_band -----------------------------------------------------------------------


def test_confidence_band_none_is_unknown(templates):
    assert confidence_band(templates, None) == templates.confidence["unknown"]


def test_confidence_band_high_at_point_nine(templates):
    assert confidence_band(templates, 0.9) == templates.confidence["high"]


def test_confidence_band_high_boundary_at_point_eight(templates):
    assert confidence_band(templates, 0.8) == templates.confidence["high"]


def test_confidence_band_medium_boundary_at_point_six(templates):
    assert confidence_band(templates, 0.6) == templates.confidence["medium"]


def test_confidence_band_low_below_medium_boundary(templates):
    assert confidence_band(templates, 0.59) == templates.confidence["low"]
