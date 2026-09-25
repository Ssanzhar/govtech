"""TDD tests for `qorgan.data.asr_style` (PLAN A10): a deterministic transform that makes
clean, punctuated, capitalised corpus text look like what the on-device Vosk recogniser
emits (ADR D25): lowercase, no punctuation, numerals spelled out in the dialogue's language."""

from __future__ import annotations

import pytest

from qorgan.data.asr_style import asr_style, asr_style_dialogue
from qorgan.data.schema import Dialogue, Label, TacticTag, Utterance, spans_from_phrases


def test_lowercases_strips_punctuation_and_collapses_whitespace():
    assert asr_style("Здравствуйте! Это — банк... Вы  меня слышите?", "ru") == "здравствуйте это банк вы меня слышите"
    assert asr_style("«Код из SMS» (никому)", "ru") == "код из sms никому"


@pytest.mark.parametrize(
    ("text", "language", "expected"),
    [
        ("переведите 5000 тенге", "ru", "переведите пять тысяч тенге"),
        ("5000 теңге аударыңыз", "kk", "бес мың теңге аударыңыз"),
        ("сумма 1 250 000 тенге", "ru", "сумма один миллион двести пятьдесят тысяч тенге"),
        ("код 4821", "mixed", "код четыре тысячи восемьсот двадцать один"),
        ("в 14:30", "ru", "в четырнадцать тридцать"),
    ],
)
def test_numerals_become_words_in_the_dialogue_language(text, language, expected):
    assert asr_style(text, language) == expected


def test_hyphenated_and_latin_tokens():
    assert asr_style("пин-код и CVV", "ru") == "пин код и cvv"  # Latin kept, lowercased
    assert asr_style("пин-код и CVV в Kaspi", "ru", drop_latin=True) == "пин код и в"  # worst case: the recogniser has no such words


def test_is_idempotent_and_leaves_clean_asr_text_alone():
    once = asr_style("Назовите КОД из SMS: 12 цифр!", "ru")
    assert asr_style(once, "ru") == once
    assert asr_style("ешкімге айтпаңыз келген кодты айтыңыз", "kk") == "ешкімге айтпаңыз келген кодты айтыңыз"


def test_asr_style_dialogue_transforms_every_utterance_and_drops_the_verbatim_spans():
    text = "Продиктуйте код из SMS."
    dialogue = Dialogue(
        id="d1", language="ru",
        utterances=(Utterance(speaker="caller", text=text), Utterance(speaker="callee", text="Зачем? 2 раза?")),
        label=Label(risk=0.9, tactic_tags=(TacticTag(id="otp_request"),), trigger_spans=spans_from_phrases(["код из SMS"], text)),
    )
    styled = asr_style_dialogue(dialogue)
    assert styled.id == dialogue.id and styled.language == "ru"
    assert [u.text for u in styled.utterances] == ["продиктуйте код из sms", "зачем два раза"]
    assert [u.speaker for u in styled.utterances] == ["caller", "callee"]
    assert styled.label.risk == 0.9 and [t.id for t in styled.label.tactic_tags] == ["otp_request"]
    assert styled.label.trigger_spans == ()  # spans are no longer verbatim in the styled text
    assert dialogue.utterances[0].text == text  # the source is untouched


def test_utterances_that_style_to_nothing_are_dropped_like_a_recogniser_would():
    dialogue = Dialogue(
        id="d2", language="ru",
        utterances=(Utterance(speaker="callee", text="…?"), Utterance(speaker="caller", text="Kaspi"), Utterance(speaker="caller", text="Назовите код.")),
        label=Label(risk=0.9, tactic_tags=(TacticTag(id="otp_request"),)),
    )
    assert [u.text for u in asr_style_dialogue(dialogue).utterances] == ["kaspi", "назовите код"]
    assert [u.text for u in asr_style_dialogue(dialogue, drop_latin=True).utterances] == ["назовите код"]
    silent = Dialogue(id="d3", language="ru", utterances=(Utterance(speaker="caller", text="Kaspi!"),), label=Label(risk=0.9))
    with pytest.raises(ValueError, match="d3"):
        asr_style_dialogue(silent, drop_latin=True)


def test_asr_style_augment_is_a_seeded_subset_with_suffixed_ids():
    from qorgan.data.asr_style import ASR_STYLE_ID_SUFFIX, asr_style_augment

    dialogues = [
        Dialogue(id=f"d{i}", language="ru", utterances=(Utterance(speaker="caller", text=f"Назовите код {i}."),), label=Label(risk=0.9))
        for i in range(40)
    ]
    assert asr_style_augment(dialogues, fraction=0.0, seed=1) == ()
    everything = asr_style_augment(dialogues, fraction=1.0, seed=1)
    assert len(everything) == 40 and all(d.id.endswith(ASR_STYLE_ID_SUFFIX) for d in everything)
    assert everything[3].utterances[0].text == "назовите код три"
    half = asr_style_augment(dialogues, fraction=0.5, seed=1)
    assert 10 < len(half) < 30 and half == asr_style_augment(dialogues, fraction=0.5, seed=1)  # deterministic
    assert {d.id for d in half} != {d.id for d in asr_style_augment(dialogues, fraction=0.5, seed=2)}
    assert set(d.id for d in half) <= set(d.id for d in everything)


def test_asr_style_augment_skips_what_cannot_be_styled():
    from qorgan.data.asr_style import asr_style_augment

    only_latin = Dialogue(id="x", language="ru", utterances=(Utterance(speaker="caller", text="Kaspi"),), label=Label(risk=0.05))
    assert asr_style_augment([only_latin], fraction=1.0, seed=1, drop_latin=True) == ()

