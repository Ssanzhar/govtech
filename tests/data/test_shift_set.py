"""The generator-shift split (PLAN A12): hand-authored by a second generator, grounded,
scrubbed, and proven disjoint from every other split before it is written."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from qorgan.data.build_corpus import normalize_for_dedup
from qorgan.data.schema import Dialogue
from qorgan.data.shift_set import (
    EVAL_SPLITS_CHECKED,
    RAW_ROW_KEYS,
    build_shift_split,
    load_raw_shift_rows,
    parse_raw_row,
)

_REPO = Path(__file__).resolve().parents[2]
_RAW_DIR = _REPO / "data" / "authored" / "shift"
_PROCESSED = _REPO / "data" / "processed"
_TACTIC_IDS = {
    "impersonation_bank", "impersonation_gov_police", "impersonation_telecom_delivery", "urgency",
    "fear_threat", "secrecy", "otp_request", "credentials_request", "safe_account", "payment_redirect",
    "remote_access", "prize_lottery", "investment_scam", "mule_recruitment", "verification_ploy",
}


def _raw(id_="shift_scam_ru_99", *, phrases=("назовите код из СМС",), scam=True):
    return {
        "id": id_,
        "language": "ru",
        "scenario": "test row",
        "utterances": [
            {"speaker": "caller", "text": "Здравствуйте, это банк, назовите код из СМС."},
            {"speaker": "callee", "text": "Какой код?"},
        ],
        "risk": 0.95 if scam else 0.02,
        "is_hard_negative": not scam,
        "tactic_tags": [{"id": "otp_request", "weight": 1.0}] if scam else [],
        "trigger_phrases": list(phrases) if scam else [],
    }


def _write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")


def _fake_processed(tmp_path: Path, *, leak: dict | None = None) -> Path:
    processed = tmp_path / "processed"
    filler_row = {**_raw("filler_ru_0", scam=False), "utterances": [
        {"speaker": "caller", "text": "Добрый день, это поликлиника, напоминаю о приёме завтра."},
        {"speaker": "callee", "text": "Спасибо, приду."},
    ]}
    filler = parse_raw_row(filler_row)
    for name in EVAL_SPLITS_CHECKED:
        rows = [filler]
        if leak and name == leak["split"]:
            rows.append(parse_raw_row(leak["row"]).model_copy(update={"id": f"{name}_leaked"}))
        _write_jsonl(processed / f"{name}.jsonl", [d.model_dump(mode="json") for d in rows])
    return processed


def test_parse_raw_row_grounds_phrases_into_verbatim_spans():
    dialogue = parse_raw_row(_raw())
    assert isinstance(dialogue, Dialogue)
    assert dialogue.label.risk == 0.95 and not dialogue.label.is_hard_negative
    assert [t.id for t in dialogue.label.tactic_tags] == ["otp_request"]
    (span,) = dialogue.label.trigger_spans
    assert dialogue.transcript()[span.start : span.end] == span.text == "назовите код из СМС"


def test_parse_raw_row_rejects_a_phrase_that_is_not_verbatim():
    with pytest.raises(ValueError, match="not verbatim"):
        parse_raw_row(_raw(phrases=("назовите код из смс",)))


def test_parse_raw_row_rejects_unknown_keys():
    row = {**_raw(), "extra": 1}
    with pytest.raises(ValueError, match="keys"):
        parse_raw_row(row)
    assert set(_raw()) == RAW_ROW_KEYS


def test_build_writes_grounded_scrubbed_rows_and_a_manifest(tmp_path):
    raw = tmp_path / "raw"
    legit = {**_raw("shift_legit_ru_01", scam=False), "utterances": [
        {"speaker": "caller", "text": "Здравствуйте, курьер банка, карта готова, завтра удобно?"},
        {"speaker": "callee", "text": "Да, после трёх."},
    ]}
    _write_jsonl(raw / "ru.jsonl", [_raw("shift_scam_ru_01"), legit])
    processed = _fake_processed(tmp_path)

    manifest = build_shift_split(raw_dir=raw, processed_dir=processed)

    written = [Dialogue.model_validate_json(l) for l in (processed / "shift.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [d.id for d in written] == ["shift_scam_ru_01", "shift_legit_ru_01"]
    assert manifest["counts"]["total"] == 2 and manifest["counts"]["positives"] == 1 and manifest["counts"]["hard_negatives"] == 1
    assert manifest["counts"]["by_language"] == {"ru": 2}
    assert manifest["scenarios"]["shift_scam_ru_01"] == "test row"
    assert json.loads((processed / "shift.manifest.json").read_text(encoding="utf-8")) == manifest


def test_build_refuses_a_row_that_overlaps_an_existing_split(tmp_path):
    raw = tmp_path / "raw"
    _write_jsonl(raw / "ru.jsonl", [_raw("shift_scam_ru_01")])
    processed = _fake_processed(tmp_path, leak={"split": "test", "row": _raw("whatever")})

    with pytest.raises(ValueError, match="overlaps test"):
        build_shift_split(raw_dir=raw, processed_dir=processed)
    assert not (processed / "shift.jsonl").exists()


# --- the committed set -------------------------------------------------------------------------

def test_committed_set_is_balanced_and_covers_every_tactic_in_every_language():
    rows = load_raw_shift_rows(_RAW_DIR)
    assert len(rows) == 66
    for language in ("ru", "kk", "mixed"):
        sub = [d for d in rows if d.language == language]
        assert len(sub) == 22
        assert sum(d.label.is_hard_negative for d in sub) == 11
        assert {t.id for d in sub for t in d.label.tactic_tags} == _TACTIC_IDS


def test_committed_set_shares_nothing_with_any_evaluation_or_training_split():
    own = {normalize_for_dedup(d.transcript()) for d in load_raw_shift_rows(_RAW_DIR)}
    for split in EVAL_SPLITS_CHECKED:
        path = _PROCESSED / f"{split}.jsonl"
        if not path.exists():
            continue
        other = {normalize_for_dedup(Dialogue.model_validate_json(l).transcript()) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()}
        assert not (own & other), f"shift set overlaps {split}"
