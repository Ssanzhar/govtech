"""TDD tests for `qorgan.data.adversarial` (PLAN_2026-09 A9): lexicon-free paraphrases of
scam calls, verified locally (zero cue hits), retried with the offending cues named, and
stored as a scrubbed eval split with the source labels."""

from __future__ import annotations

import pytest

from qorgan.classifier.cue_lexicon import CueLexicon
from qorgan.data.adversarial import (
    ADVERSARIAL_ID_PREFIX,
    ParaphraseFailure,
    build_paraphrase_prompt,
    cue_hits,
    language_matches,
    paraphrase_dialogue,
    source_positives,
)
from qorgan.data.schema import Dialogue, Label, TacticTag, Utterance

LEXICON = CueLexicon(version=1, entries={
    "secrecy": ("никому не говорите",),
    "otp_request": ("код из смс", "назовите код"),
    "credentials_request": ("номер карты",),
    "safe_account": ("безопасный счёт",),
    "remote_access": ("anydesk",),
})


def _scam(dialogue_id: str = "scam_1", language: str = "ru") -> Dialogue:
    return Dialogue(
        id=dialogue_id, language=language,
        utterances=(
            Utterance(speaker="caller", text="Это служба безопасности банка, назовите код из СМС."),
            Utterance(speaker="callee", text="Какой код?"),
        ),
        label=Label(risk=0.9, tactic_tags=(TacticTag(id="impersonation_bank"), TacticTag(id="otp_request"))),
    )


def _legit() -> Dialogue:
    return Dialogue(
        id="legit_1", language="ru", utterances=(Utterance(speaker="caller", text="Ваша карта готова."),),
        label=Label(risk=0.05, is_hard_negative=True),
    )


class FakeClient:
    """Returns scripted payloads in order; records the prompts it saw."""

    def __init__(self, payloads):
        self._payloads = list(payloads)
        self.prompts: list[str] = []

    def paraphrase(self, prompt: str) -> dict:
        self.prompts.append(prompt)
        return self._payloads.pop(0)


def test_cue_hits_lists_the_offending_cues_case_insensitively():
    assert cue_hits("НАЗОВИТЕ КОД из смс, номер карты тоже", LEXICON) == ("код из смс", "назовите код", "номер карты")
    assert cue_hits("скажите цифры, которые пришли", LEXICON) == ()


def test_source_positives_keeps_only_scams():
    assert [d.id for d in source_positives([_legit(), _scam()])] == ["scam_1"]


def test_prompt_names_tactics_language_and_every_cue():
    prompt = build_paraphrase_prompt(_scam(), LEXICON, previous_hits=("код из смс",))
    for cue in ("никому не говорите", "код из смс", "anydesk"):
        assert cue in prompt
    assert "otp_request" in prompt and "impersonation_bank" in prompt and "ru" in prompt
    assert "код из смс" in prompt.split("previous attempt", 1)[1]  # retry names what still leaked


def test_paraphrase_retries_until_cue_free_and_keeps_labels():
    leaking = {"utterances": [{"speaker": "caller", "text": "Банк. Назовите код, пожалуйста."}, {"speaker": "callee", "text": "Ок"}]}
    clean = {"utterances": [{"speaker": "caller", "text": "Банк. Продиктуйте цифры из сообщения, +7 700 111 22 33 это мы."}, {"speaker": "callee", "text": "Ок"}]}
    client = FakeClient([leaking, clean])

    result = paraphrase_dialogue(_scam(), LEXICON, paraphrase=client.paraphrase, max_attempts=3)

    assert result.id == f"{ADVERSARIAL_ID_PREFIX}scam_1" and result.language == "ru"
    assert result.label.tactic_tags == _scam().label.tactic_tags and result.label.risk == 0.9
    assert result.label.trigger_spans == () and not result.label.is_hard_negative
    assert cue_hits(result.transcript(), LEXICON) == ()
    assert "[PHONE]" in result.transcript() and "111 22 33" not in result.transcript()  # scrubbed
    assert len(client.prompts) == 2 and "назовите код" in client.prompts[1]


def test_paraphrase_gives_up_after_max_attempts():
    leaking = {"utterances": [{"speaker": "caller", "text": "Назовите код."}]}
    client = FakeClient([leaking, leaking])
    with pytest.raises(ParaphraseFailure):
        paraphrase_dialogue(_scam(), LEXICON, paraphrase=client.paraphrase, max_attempts=2)


def test_empty_or_malformed_payloads_count_as_attempts():
    client = FakeClient([{"utterances": []}, {"nope": 1}])
    with pytest.raises(ParaphraseFailure):
        paraphrase_dialogue(_scam(), LEXICON, paraphrase=client.paraphrase, max_attempts=2)


def test_language_matches_tells_kazakh_from_russian():
    assert language_matches("Здравствуйте, это служба безопасности банка.", "ru")
    assert not language_matches("Сәлеметсіз бе, бұл банктің қауіпсіздік қызметі.", "ru")
    assert language_matches("Сәлеметсіз бе, бұл банктің қауіпсіздік қызметі.", "kk")
    assert not language_matches("Здравствуйте, это служба безопасности банка.", "kk")
    assert language_matches("Сәлеметсіз бе, это банк.", "mixed")
    assert not language_matches("...", "ru")


def test_a_paraphrase_in_the_wrong_language_is_retried():
    switched = {"utterances": [{"speaker": "caller", "text": "Сәлеметсіз бе, бұл банк. Цифрларды айтыңыз."}]}
    russian = {"utterances": [{"speaker": "caller", "text": "Здравствуйте, это банк. Продиктуйте цифры."}]}
    client = FakeClient([switched, russian])

    result = paraphrase_dialogue(_scam(language="ru"), LEXICON, paraphrase=client.paraphrase, max_attempts=2)

    assert result.transcript() == "Здравствуйте, это банк. Продиктуйте цифры."
    assert "Russian only" in client.prompts[1] and "not written in" in client.prompts[1]


def test_legit_sounding_style_adds_the_register_instruction_and_keeps_the_cue_ban():
    from qorgan.data.adversarial import PARAPHRASE_STYLES, build_paraphrase_prompt, split_name_for

    prompt = build_paraphrase_prompt(_scam(), LEXICON, style="legit_sounding")
    assert "reassur" in prompt.lower() and "never ask" in prompt.lower()
    assert "HARD CONSTRAINT" in prompt and "код из смс" in prompt  # the lexicon ban still applies
    assert "never ask" not in build_paraphrase_prompt(_scam(), LEXICON).lower()
    assert set(PARAPHRASE_STYLES) == {"cue_free", "legit_sounding"}
    assert split_name_for("cue_free") == "adversarial" and split_name_for("legit_sounding") == "adversarial_legit"
    with pytest.raises(ValueError):
        build_paraphrase_prompt(_scam(), LEXICON, style="polite")
