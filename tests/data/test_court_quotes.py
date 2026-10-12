"""`court_quotes` extraction, PII drop rule and publication guard (ADR D68).

The verdict snippets below are invented test fixtures in verdict register; they are not data
and not taken from any verdict."""

import json
from dataclasses import replace
from pathlib import Path

import pytest

from qorgan.data.court_quotes import (
    COURT_MASK_TOKEN,
    MANIFEST_RELPATH,
    QuoteRow,
    VerdictMeta,
    build_manifest,
    build_rows,
    clean_quote,
    extract_passages,
    is_criminal_act,
    load_quotes,
    parse_sudact,
    residual_pii,
    write_rows,
)

_REPO = Path(__file__).resolve().parents[2]

_VERDICT = (
    "ПРИГОВОР\n"
    "Неустановленное лицо позвонило на абонентский номер фио и, представившись сотрудником банка, "
    "сообщило, что с её счета пытаются похитить деньги и их нужно срочно перевести на безопасный счет. "
    "Он сказал: «Никому не говорите о нашем разговоре, иначе деньги пропадут». "
    "Подсудимый пояснил, что вину признает полностью и раскаивается.\n"
)
_META = VerdictMeta(
    verdict_id="TEST01", source="sudact.ru", url="https://sudact.ru/regular/doc/TEST01/",
    court="Тестовый районный суд", date="2025-01-02", title="Приговор № 1-1/2025 от 2 января 2025 г.",
)


def test_reported_and_direct_caller_speech_are_extracted_verbatim():
    passages = extract_passages(_VERDICT)
    texts = {p.text: p.speech for p in passages}
    assert texts["Никому не говорите о нашем разговоре, иначе деньги пропадут"] == "direct"
    reported = [t for t, s in texts.items() if s == "reported"]
    assert any("безопасный счет" in t for t in reported)
    for text in texts:
        assert text in _VERDICT  # verbatim spans only, nothing rewritten


def test_defendant_statements_are_not_caller_speech():
    assert not any("вину признает" in p.text for p in extract_passages(_VERDICT))


def test_a_verdict_without_an_impersonation_call_yields_nothing():
    text = "Подсудимый сообщил потерпевшей, что изготовит ей шкаф, и попросил предоплату 50 000 рублей."
    assert extract_passages(text) == ()


def test_court_masks_do_not_split_sentences():
    text = (
        "Неустановленное лицо по телефону, представившись сотрудником «...», сообщило, что "
        "необходимо следовать инструкциям сотрудников «...», которые с ней свяжутся."
    )
    passages = extract_passages(text)
    assert any(COURT_MASK_TOKEN in p.text and p.text.endswith("свяжутся") for p in passages)


@pytest.mark.parametrize(
    "text",
    [
        "переведите деньги на карту 4400 1234 5678 9010 прямо сейчас",
        "ваш ИИН 770808300300 заблокирован банком",
        "это Иванова А.Б. из службы безопасности банка",
        "курьер ждет у подъезда № 6 дома № 78 прямо сейчас",
        "напишите на почту ivanov@example.com код подтверждения",
    ],
)
def test_rows_the_scrubber_cannot_make_safe_are_dropped(text):
    assert clean_quote(text) is None


def test_a_phone_number_is_scrubbed_and_the_row_kept():
    assert clean_quote("перезвоните мне на номер +7 701 123 45 67 срочно") == "перезвоните мне на номер [PHONE] срочно"


def test_residual_pii_ignores_money_and_codes():
    assert not residual_pii("продиктуйте код 123456 и переведите 50 000 рублей")


def test_build_rows_counts_drops_and_cites_the_source():
    rows, dropped = build_rows(_META, _VERDICT)
    assert rows and dropped == 0
    assert all(r.source_url == _META.url and r.verdict_id == "TEST01" and r.language == "ru" for r in rows)
    assert len({r.id for r in rows}) == len(rows)


def test_only_criminal_acts_are_mined():
    assert is_criminal_act(_META)
    civil = replace(_META, title="Решение № 2-148/2025 от 23 апреля 2025 г.")
    assert not is_criminal_act(civil)
    assert build_rows(civil, _VERDICT) == ((), 0)


def test_parse_sudact_reads_title_court_date_and_body():
    page = (
        '<h1>Приговор № 1-1/2025 от 27 ноября 2025 г. по делу № 1-1/2025</h1>'
        '<div class="b-justice"><a href="/regular/court/x/">Бутырский районный суд (Город Москва) </a> - Уголовное</div>'
        '<hr class="hr-h1"><b>ПРИГОВОР</b><br>Текст приговора.<div class="h-footer">футер</div>'
    )
    meta, text = parse_sudact(page, "abc")
    assert meta.date == "2025-11-27" and meta.court.startswith("Бутырский")
    assert meta.url == "https://sudact.ru/regular/doc/abc/"
    assert "Текст приговора." in text and "футер" not in text


def test_rows_round_trip_and_manifest_holds_no_text(tmp_path):
    rows, _ = build_rows(_META, _VERDICT)
    path = tmp_path / "quotes.jsonl"
    write_rows(rows, path)
    assert load_quotes(path) == rows
    manifest = build_manifest(rows, collected_on="2026-10-09", dropped_pii=0, verdicts_read=1)
    blob = json.dumps(manifest, ensure_ascii=False)
    assert manifest["n_quotes"] == len(rows)
    for row in rows:
        assert row.text not in blob


def test_committed_manifest_carries_no_quote_text():
    path = _REPO / MANIFEST_RELPATH
    if not path.exists():
        pytest.skip("court_quotes manifest not committed yet")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    assert set(manifest) >= {"n_quotes", "citations", "by_language"}
    assert "text" not in json.dumps(manifest["citations"], ensure_ascii=False).replace("text_location", "")
    local = _REPO / "data" / "raw" / "court_quotes" / "quotes.jsonl"
    if local.exists():
        blob = path.read_text(encoding="utf-8")
        assert not any(r.text in blob for r in load_quotes(local))


def test_court_quotes_are_never_published():
    upload = (_REPO / "scripts" / "hf_upload.py").read_text(encoding="utf-8")
    assert "assert_not_publishing_court_quotes" in upload
    from qorgan.data.publish_guard import PUBLISHED_SPLITS, assert_not_publishing_court_quotes

    assert not any("court" in name for name in PUBLISHED_SPLITS)
    with pytest.raises(SystemExit):
        assert_not_publishing_court_quotes(["train.jsonl", "court_quotes.jsonl"])
    with pytest.raises(SystemExit):
        assert_not_publishing_court_quotes(["raw/court_quotes/quotes.jsonl"])
    assert_not_publishing_court_quotes(["train.jsonl", "shift.jsonl"])  # no raise


def test_court_quotes_text_is_gitignored():
    ignore = (_REPO / ".gitignore").read_text(encoding="utf-8")
    assert "data/raw/court_quotes/" in ignore


def test_quote_row_is_immutable():
    row = QuoteRow(
        id="x", language="ru", text="t", speech="direct", tactic_guess="", source="s",
        source_url="u", court="c", date="", verdict_id="v",
    )
    with pytest.raises(AttributeError):
        row.text = "changed"  # type: ignore[misc]
