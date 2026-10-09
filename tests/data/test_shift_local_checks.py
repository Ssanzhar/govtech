"""Automated acceptance for `shift_local` (ADR D65): format, turn count, language / script,
prompt echo, degeneration and duplicates. These are the ONLY reasons a generated call may
be rejected -- never a classifier score."""

from __future__ import annotations

import pytest

from qorgan.data.shift_local_checks import (
    REJECT_REASONS,
    accept,
    char_ngram_jaccard,
    parse_turns,
    script_stats,
)

_KK_TURNS = [
    "Сәлеметсіз бе, сізге қала поликлиникасынан хабарласып тұрмын.",
    "Иә, сәлеметсіз бе, тыңдап тұрмын.",
    "Сіздің ертеңгі қабылдауыңыз туралы ескерткім келді.",
    "Рақмет, қай уақытта болады?",
    "Таңғы оныншы жарымда, екінші қабатта.",
    "Жақсы, мен келемін.",
    "Жеке куәлігіңізді ала келіңіз.",
    "Әрине, ұмытпаймын. Сау болыңыз.",
]
_RU_TURNS = [
    "Добрый день, это курьерская служба, у вас посылка.",
    "Здравствуйте, да, я жду посылку.",
    "Когда вам удобно её получить?",
    "Завтра после обеда, если можно.",
    "Хорошо, курьер позвонит за час.",
    "А оплачивать нужно что-то?",
    "Нет, всё уже оплачено отправителем.",
    "Отлично, спасибо, до свидания.",
]
_MIXED_TURNS = [
    "Сәлеметсіз бе, это из банка звоню, сіздің картаңыз бойынша вопрос.",
    "Иә, слушаю вас, что случилось?",
    "Сізде новая карта готова, можно забрать в отделении.",
    "Жақсы, а когда можно подойти?",
    "Ертең с девяти до шести, жеке куәлік ала келіңіз.",
    "Хорошо, рақмет, обязательно зайду.",
    "Тағы бір нәрсе: если будут вопросы, звоните по номеру на карте.",
    "Түсіндім, спасибо, сау болыңыз.",
]
_EN_TURNS = [
    "Hello, this is the clinic calling about your appointment tomorrow.",
    "Oh hi, yes, what about it?",
    "We wanted to check whether ten thirty still works for you.",
    "Could we move it to the afternoon?",
    "Sure, we have a slot at three o'clock.",
    "Three is perfect, thank you.",
    "Please bring your ID card with you.",
    "Will do. Goodbye.",
]


def _raw(turns: list[str], *, preamble: str = "") -> str:
    lines = [f"{'CALLER' if i % 2 == 0 else 'CALLEE'}: {t}" for i, t in enumerate(turns)]
    return (preamble + "\n" if preamble else "") + "\n".join(lines)


_PROMPT = "Write a realistic phone call. Use the tags CALLER and CALLEE. Brief: a nurse reminds a patient."


# --- parsing --------------------------------------------------------------------------------


def test_parse_turns_maps_tags_to_speakers_and_tolerates_markdown_bold():
    turns, stray = parse_turns("**CALLER:** Hello there\n**Callee**: Hi\n")
    assert turns == (("caller", "Hello there"), ("callee", "Hi"))
    assert stray == 0


def test_parse_turns_counts_stray_lines_after_the_dialogue_starts_but_not_a_short_preamble():
    raw = "Here you go:\nCALLER: Hello\nsome narration\nCALLEE: Hi"
    turns, stray = parse_turns(raw)
    assert [s for s, _ in turns] == ["caller", "callee"]
    assert stray == 1


# --- script statistics ----------------------------------------------------------------------


def test_script_stats_measure_cyrillic_latin_and_kazakh_letters():
    stats = script_stats("Сәлем hello")
    assert stats.letters == 10
    assert stats.cyrillic == 5 and stats.latin == 5
    assert stats.kazakh_specific == 1


# --- accept: the happy path per language ----------------------------------------------------


@pytest.mark.parametrize(
    ("language", "turns"),
    [("kk", _KK_TURNS), ("ru", _RU_TURNS), ("mixed", _MIXED_TURNS), ("en", _EN_TURNS)],
)
def test_well_formed_calls_are_accepted_in_their_language(language, turns):
    verdict = accept(_raw(turns), language=language, prompt=_PROMPT, accepted=())
    assert verdict.reason is None, verdict.reason
    assert verdict.accepted
    assert len(verdict.utterances) == len(turns)


# --- accept: each rejection reason ----------------------------------------------------------


def test_no_tags_is_a_format_rejection():
    verdict = accept("Just some prose about a call.", language="en", prompt=_PROMPT, accepted=())
    assert verdict.reason == "format"


def test_too_few_turns_is_rejected():
    verdict = accept(_raw(_EN_TURNS[:5]), language="en", prompt=_PROMPT, accepted=())
    assert verdict.reason == "turn_count"


def test_one_sided_call_is_rejected():
    raw = "\n".join(f"CALLER: {t}" for t in _EN_TURNS)
    assert accept(raw, language="en", prompt=_PROMPT, accepted=()).reason == "speakers"


@pytest.mark.parametrize(
    ("language", "turns"),
    [("kk", _RU_TURNS), ("ru", _KK_TURNS), ("en", _RU_TURNS), ("mixed", _RU_TURNS), ("mixed", _KK_TURNS)],
)
def test_the_wrong_language_is_rejected(language, turns):
    assert accept(_raw(turns), language=language, prompt=_PROMPT, accepted=()).reason == "language"


def test_meta_words_inside_an_utterance_are_prompt_echo():
    turns = [*_EN_TURNS[:-1], "This transcript follows the brief exactly."]
    assert accept(_raw(turns), language="en", prompt=_PROMPT, accepted=()).reason == "prompt_echo"


def test_copying_eight_words_of_the_prompt_is_prompt_echo():
    turns = [*_EN_TURNS[:-1], "write a realistic phone call use the tags now"]
    assert accept(_raw(turns), language="en", prompt=_PROMPT, accepted=()).reason == "prompt_echo"


def test_a_looping_generation_is_degenerate():
    turns = [*_EN_TURNS[:4], "Sure, we have a slot.", "Sure, we have a slot.", "Sure, we have a slot.", "Bye."]
    assert accept(_raw(turns), language="en", prompt=_PROMPT, accepted=()).reason == "degenerate"


def test_an_exact_or_near_duplicate_of_an_accepted_call_is_rejected():
    first = accept(_raw(_EN_TURNS), language="en", prompt=_PROMPT, accepted=())
    again = accept(_raw(_EN_TURNS), language="en", prompt=_PROMPT, accepted=(first.transcript,))
    assert again.reason == "duplicate"
    near = [*_EN_TURNS[:-1], "Will do. Bye now."]
    assert accept(_raw(near), language="en", prompt=_PROMPT, accepted=(first.transcript,)).reason == "duplicate"


def test_a_very_long_turn_is_rejected():
    turns = [*_EN_TURNS[:-1], "word " * 400]
    assert accept(_raw(turns), language="en", prompt=_PROMPT, accepted=()).reason == "turn_length"


def test_every_reason_is_declared():
    assert set(REJECT_REASONS) == {
        "format", "turn_count", "speakers", "turn_length", "prompt_echo", "language", "degenerate", "duplicate",
    }


def test_jaccard_is_one_for_identical_and_zero_for_disjoint_text():
    assert char_ngram_jaccard("abcdefgh", "abcdefgh") == 1.0
    assert char_ngram_jaccard("aaaaaaa", "zzzzzzz") == 0.0
