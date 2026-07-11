"""Curated `real_heldout` anchor dialogues (D2-3, manual-transcript fallback).

No public Kazakh/Russian scam-call transcript corpus exists, and streaming ASR of real
recordings is explicitly out of scope this sprint (`docs/SCOPE.md`). So the `real_heldout`
generalization set is a small, hand-curated set of realistic Kazakhstani calls --
paraphrased from publicly documented scam scripts and everyday call patterns, with every
identifier already abstracted (no real phone numbers/cards/IINs). It is kept deliberately
separate from the synthetic train/val/test corpus: FPR on this set is the honest
generalization signal reported by `qorgan.eval.run` (CLAUDE.md SS6).

Anchors are authored as plain text + verbatim trigger phrases; `build_anchor_dialogues()`
grounds each phrase into a `Span` via `spans_from_phrases`, so offsets are never
hand-computed and the schema's verbatim-span invariant holds by construction.
"""

from __future__ import annotations

from typing import NamedTuple

from qorgan.data.schema import (
    Dialogue,
    Label,
    SupportedLanguage,
    TacticTag,
    Utterance,
    spans_from_phrases,
)

# Provisional risk for anchors. These are curated ground truth, not model output; the
# eval harness binarizes at the configured risk threshold, so exact values only need to
# sit clearly on the correct side of it.
_ANCHOR_POSITIVE_RISK = 0.92
_ANCHOR_NEGATIVE_RISK = 0.03


class AnchorSpec(NamedTuple):
    """A curated anchor before schema grounding: raw turns + verbatim trigger phrases."""

    id: str
    language: SupportedLanguage
    turns: tuple[tuple[str, str], ...]  # (speaker, text)
    tactic_ids: tuple[str, ...]
    trigger_phrases: tuple[str, ...]
    is_hard_negative: bool


# --- Scam positives (paraphrased from documented KZ/RU social-engineering scripts) -------

_POSITIVE_ANCHORS: tuple[AnchorSpec, ...] = (
    AnchorSpec(
        id="real_scam_bank_otp_ru",
        language="ru",
        turns=(
            ("caller", "Здравствуйте, это служба безопасности вашего банка."),
            ("caller", "По вашей карте зафиксирована подозрительная операция, действовать нужно прямо сейчас."),
            ("callee", "Ой, а что случилось?"),
            ("caller", "Никому не говорите об этом звонке, даже родным. Продиктуйте код из SMS, чтобы отменить списание."),
            ("caller", "И переведите деньги на безопасный счёт, который я вам сейчас продиктую."),
        ),
        tactic_ids=("impersonation_bank", "urgency", "secrecy", "otp_request", "safe_account"),
        trigger_phrases=(
            "это служба безопасности вашего банка",
            "действовать нужно прямо сейчас",
            "Никому не говорите об этом звонке",
            "Продиктуйте код из SMS",
            "переведите деньги на безопасный счёт",
        ),
        is_hard_negative=False,
    ),
    AnchorSpec(
        id="real_scam_police_case_ru",
        language="ru",
        turns=(
            ("caller", "Вас беспокоит следователь финансовой полиции."),
            ("caller", "На ваше имя оформлено уголовное дело о финансировании мошенников."),
            ("callee", "Этого не может быть, я ничего не делала."),
            ("caller", "Если не подтвердите личность прямо сейчас, счёт заблокируют. Никому не сообщайте, идёт секретная проверка."),
            ("caller", "Назовите ваш ИИН и код из push-уведомления для сверки."),
        ),
        tactic_ids=("impersonation_gov_police", "fear_threat", "secrecy", "credentials_request", "otp_request"),
        trigger_phrases=(
            "Вас беспокоит следователь финансовой полиции",
            "На ваше имя оформлено уголовное дело",
            "счёт заблокируют",
            "идёт секретная проверка",
            "код из push-уведомления",
        ),
        is_hard_negative=False,
    ),
    AnchorSpec(
        id="real_scam_telecom_disconnect_mixed",
        language="mixed",
        turns=(
            ("caller", "Сәлеметсіз бе, байланыс операторынан мазалап тұрмыз."),
            ("caller", "Сіздің нөміріңіз бүгін өшіріледі, если не продлить договор прямо сейчас."),
            ("callee", "Как продлить? Я не хочу терять номер."),
            ("caller", "Назовите номер карты и три цифры на обороте, чтобы мы оформили продление."),
        ),
        tactic_ids=("impersonation_telecom_delivery", "urgency", "credentials_request"),
        trigger_phrases=(
            "байланыс операторынан",
            "Сіздің нөміріңіз бүгін өшіріледі",
            "Назовите номер карты и три цифры на обороте",
        ),
        is_hard_negative=False,
    ),
    AnchorSpec(
        id="real_scam_investment_qr_kk",
        language="kk",
        turns=(
            ("caller", "Сәлеметсіз бе! Сізге тиесілі жүлде бар, тек растау қажет."),
            ("caller", "Айына 30% кепілді табыс беретін бірлескен инвестиция ұсынамыз."),
            ("callee", "Қалай қатысуға болады?"),
            ("caller", "Дәл қазір осы QR-код арқылы төлеңіз, әйтпесе ұсыныс жанып кетеді."),
        ),
        tactic_ids=("prize_lottery", "investment_scam", "urgency", "payment_redirect"),
        trigger_phrases=(
            "Сізге тиесілі жүлде бар",
            "Айына 30% кепілді табыс",
            "осы QR-код арқылы төлеңіз",
        ),
        is_hard_negative=False,
    ),
    AnchorSpec(
        id="real_scam_remote_access_ru",
        language="ru",
        turns=(
            ("caller", "Здравствуйте, звоним из технической поддержки банка."),
            ("caller", "Чтобы защитить ваш счёт, установите приложение AnyDesk и продиктуйте номер сессии."),
            ("callee", "А это точно безопасно?"),
            ("caller", "Конечно, только не сообщайте сотрудникам банка, это внутренняя процедура."),
        ),
        tactic_ids=("impersonation_bank", "remote_access", "secrecy"),
        trigger_phrases=(
            "установите приложение AnyDesk",
            "не сообщайте сотрудникам банка",
        ),
        is_hard_negative=False,
    ),
)


# --- Hard negatives + benign calls (must NOT trigger -- the FPR anchors) ------------------

_NEGATIVE_ANCHORS: tuple[AnchorSpec, ...] = (
    AnchorSpec(
        id="real_neg_bank_confirm_ru",
        language="ru",
        turns=(
            ("caller", "Здравствуйте, это оператор банка."),
            ("caller", "Подтверждаем, что перевод на 15000 тенге в пользу Ивана И. успешно проведён."),
            ("callee", "Да, это я переводил, спасибо."),
            ("caller", "Если вы не совершали операцию, перезвоните на номер на обороте карты. Хорошего дня!"),
        ),
        tactic_ids=(),
        trigger_phrases=(),
        is_hard_negative=True,
    ),
    AnchorSpec(
        id="real_neg_family_money_mixed",
        language="mixed",
        turns=(
            ("caller", "Алло, апа, это я. Маған бүгін азырақ ақша керек еді."),
            ("callee", "Что случилось, всё нормально?"),
            ("caller", "Да всё хорошо, просто до зарплаты не хватает, кешке қайтарамын."),
            ("callee", "Хорошо, сейчас переведу на твою карту как обычно."),
        ),
        tactic_ids=(),
        trigger_phrases=(),
        is_hard_negative=True,
    ),
    AnchorSpec(
        id="real_neg_clinic_reminder_kk",
        language="kk",
        turns=(
            ("caller", "Сәлеметсіз бе, емханадан хабарласып тұрмыз."),
            ("caller", "Ертең сағат он бесте дәрігерге жазылғаныңызды еске саламыз."),
            ("callee", "Рахмет, келемін."),
            ("caller", "Жақсы, күтеміз. Сау болыңыз!"),
        ),
        tactic_ids=(),
        trigger_phrases=(),
        is_hard_negative=True,
    ),
    AnchorSpec(
        id="real_neg_delivery_ru",
        language="ru",
        turns=(
            ("caller", "Добрый день, курьерская служба, ваша посылка прибыла в отделение."),
            ("callee", "О, отлично, когда можно забрать?"),
            ("caller", "Сегодня до восьми вечера, паспорт не нужен, только номер заказа."),
            ("callee", "Спасибо, подъеду после работы."),
        ),
        tactic_ids=(),
        trigger_phrases=(),
        is_hard_negative=False,
    ),
    AnchorSpec(
        id="real_neg_chitchat_mixed",
        language="mixed",
        turns=(
            ("caller", "Сәлем, қалайсың? Что делаешь на выходных?"),
            ("callee", "Да ничего особенного, үйдемін. А ты?"),
            ("caller", "Думаю в горы съездить, ауа райы жақсы болады дейді."),
            ("callee", "О, круто, возьми меня с собой!"),
        ),
        tactic_ids=(),
        trigger_phrases=(),
        is_hard_negative=False,
    ),
)

ANCHOR_SPECS: tuple[AnchorSpec, ...] = (*_POSITIVE_ANCHORS, *_NEGATIVE_ANCHORS)


def _to_dialogue(spec: AnchorSpec) -> Dialogue:
    utterances = tuple(Utterance(speaker=speaker, text=text) for speaker, text in spec.turns)
    transcript = "\n".join(u.text for u in utterances)
    spans = spans_from_phrases(spec.trigger_phrases, transcript)
    if len(spans) != len(spec.trigger_phrases):
        # Build-time self-check: a curated trigger phrase must appear verbatim, or the
        # anchor is authored wrong. Fail loudly rather than ship a silently-dropped span.
        raise ValueError(
            f"Anchor {spec.id!r}: {len(spec.trigger_phrases) - len(spans)} trigger phrase(s) "
            "are not verbatim substrings of the transcript"
        )
    risk = _ANCHOR_NEGATIVE_RISK if spec.is_hard_negative or not spec.tactic_ids else _ANCHOR_POSITIVE_RISK
    tags = tuple(TacticTag(id=tid, weight=1.0) for tid in spec.tactic_ids)
    return Dialogue(
        id=spec.id,
        language=spec.language,
        utterances=utterances,
        label=Label(
            risk=risk,
            tactic_tags=tags,
            trigger_spans=spans,
            is_hard_negative=spec.is_hard_negative,
        ),
    )


def build_anchor_dialogues() -> tuple[Dialogue, ...]:
    """Ground every `AnchorSpec` into a schema-valid `Dialogue` (verbatim spans enforced)."""
    return tuple(_to_dialogue(spec) for spec in ANCHOR_SPECS)
