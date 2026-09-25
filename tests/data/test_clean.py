"""TDD tests for `qorgan.data.clean` (ADR D34): generation artefacts found in the corpus on
2026-09-20 -- utterances wrapped in `"\\r\\n … "`, backspaces inside words, literal `\\uXXXX`
escapes -- are repaired where the text is recoverable and the dialogue is dropped where it
is not (control characters standing in for lost letters)."""

from __future__ import annotations

from qorgan.data.clean import clean_dialogue, clean_text, is_corrupted
from qorgan.data.schema import Dialogue, Label, Utterance


def test_wrapping_quotes_and_carriage_returns_are_removed():
    assert clean_text('"\r\nҚайырлы күн! Бұл \'Сенімді Курьер\'.\r\n"\r\n') == "Қайырлы күн! Бұл 'Сенімді Курьер'."
    assert clean_text('"\r\nИә, тыңдап тұрмын. Қандай сәлемдеме?\r\n"') == "Иә, тыңдап тұрмын. Қандай сәлемдеме?"


def test_backspaces_and_literal_unicode_escapes_are_repaired():
    assert clean_text("Алло, с\x08лушаю") == "Алло, слушаю"
    assert clean_text("Men Kaspi Bank-ten qo\\u0144yrau shalyap turmyn") == "Men Kaspi Bank-ten qońyrau shalyap turmyn"


def test_clean_text_is_idempotent_and_leaves_clean_text_alone():
    clean = "Здравствуйте, это банк. «Сенім» банкі емес пе?"
    assert clean_text(clean) == clean
    once = clean_text('"\r\nа\x08бв\r\n"')
    assert clean_text(once) == once == "абв"


def test_control_characters_inside_words_mark_a_dialogue_as_corrupted():
    assert is_corrupted("S\ntyzba. B\rghyn ta\rdan M\x0cdykanyq")  # lost Latin-Kazakh letters
    assert is_corrupted(clean_text("Iya, d\r s. Men k\r tetin edim."))  # a lone \r is not a line ending
    assert not is_corrupted("Алло, слушаю")
    assert not is_corrupted(clean_text("Алло, с\x08лушаю"))
    assert not is_corrupted(clean_text('"\r\nБұл банк."\r\n\n"\r\nИә?"'))  # turn breaks are formatting


def test_clean_dialogue_splits_jammed_turns_back_into_utterances():
    jammed = Dialogue(id="j", language="kk", utterances=(Utterance(speaker="caller", text='"\r\nҚайырлы күн! Сәлемдеме бар."\r\n\n"\r\nИә, тыңдап тұрмын.\r\n"\r\n'), Utterance(speaker="caller", text='"\r\nБұл құжаттар."')), label=Label(risk=0.05, is_hard_negative=True))
    repaired = clean_dialogue(jammed)
    assert [(u.speaker, u.text) for u in repaired.utterances] == [("caller", "Қайырлы күн! Сәлемдеме бар."), ("callee", "Иә, тыңдап тұрмын."), ("caller", "Бұл құжаттар.")]
    assert repaired.label.trigger_spans == ()


def test_clean_dialogue_repairs_or_drops():
    label = Label(risk=0.05)
    repaired = clean_dialogue(Dialogue(id="ok", language="kk", utterances=(Utterance(speaker="caller", text='"\r\nҚайырлы күн!\r\n"'), Utterance(speaker="callee", text="Иә?")), label=label))
    assert [u.text for u in repaired.utterances] == ["Қайырлы күн!", "Иә?"]
    assert repaired.id == "ok" and repaired.label == label
    dropped = clean_dialogue(Dialogue(id="bad", language="mixed", utterances=(Utterance(speaker="caller", text="Allo, s\ntyzba?"),), label=label))
    assert dropped is None


def test_a_turn_that_is_only_a_stray_quote_is_removed_not_fatal():
    d = Dialogue(id="q", language="ru", utterances=(Utterance(speaker="caller", text='"\n'), Utterance(speaker="callee", text="Алло?")), label=Label(risk=0.05))
    assert [u.text for u in clean_dialogue(d).utterances] == ["Алло?"]
    assert clean_dialogue(Dialogue(id="e", language="ru", utterances=(Utterance(speaker="caller", text='"'),), label=Label(risk=0.05))) is None


def test_cli_repairs_a_jsonl_in_place_and_reports_drops(tmp_path, capsys):
    from qorgan.data.clean import main

    rows = [
        Dialogue(id="ok", language="ru", utterances=(Utterance(speaker="caller", text="Алло, с\x08лушаю"),), label=Label(risk=0.05)),
        Dialogue(id="bad", language="mixed", utterances=(Utterance(speaker="caller", text="Alo s\nlemetsiz be?"),), label=Label(risk=0.05)),
    ]
    path = tmp_path / "split.jsonl"
    path.write_text("".join(d.model_dump_json() + "\n" for d in rows), encoding="utf-8")
    main([str(path)])
    kept = [Dialogue.model_validate_json(l) for l in path.read_text(encoding="utf-8").splitlines() if l]
    assert [d.id for d in kept] == ["ok"] and kept[0].utterances[0].text == "Алло, слушаю"
    out = capsys.readouterr().out
    assert "dropped 1" in out and "bad" in out and "repaired 1" in out
    main([str(path)])  # idempotent
    assert "repaired 0" in capsys.readouterr().out
