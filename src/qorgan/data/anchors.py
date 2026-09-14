"""Curated `authored_heldout` anchor dialogues (D2-3, manual-transcript fallback).

No public Kazakh/Russian scam-call transcript corpus exists, and streaming ASR of real
recordings is explicitly out of scope this sprint (`docs/SCOPE.md`). So the `authored_heldout`
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
    AnchorSpec(
        id="real_scam_prize_iin_ru",
        language="ru",
        turns=(
            ("caller", "Поздравляем! Ваш номер выиграл автомобиль в республиканской акции."),
            ("callee", "Правда? А что нужно сделать?"),
            ("caller", "Чтобы оформить выигрыш, назовите ваш ИИН и номер карты для зачисления приза."),
            ("caller", "Только поторопитесь, приз нужно забрать сегодня."),
        ),
        tactic_ids=("prize_lottery", "credentials_request", "urgency"),
        trigger_phrases=(
            "Ваш номер выиграл автомобиль",
            "назовите ваш ИИН и номер карты",
            "приз нужно забрать сегодня",
        ),
        is_hard_negative=False,
    ),
    AnchorSpec(
        id="real_scam_delivery_customs_ru",
        language="ru",
        turns=(
            ("caller", "Здравствуйте, служба доставки, для вашей посылки нужно доплатить таможенный сбор."),
            ("callee", "Какой сбор? Я ничего из-за границы не заказывал."),
            ("caller", "Оплатите по ссылке, которую я отправлю в SMS, иначе посылку вернут отправителю."),
        ),
        tactic_ids=("impersonation_telecom_delivery", "payment_redirect", "urgency"),
        trigger_phrases=(
            "нужно доплатить таможенный сбор",
            "Оплатите по ссылке, которую я отправлю в SMS",
        ),
        is_hard_negative=False,
    ),
    AnchorSpec(
        id="real_scam_knb_secret_kk",
        language="kk",
        turns=(
            ("caller", "Сәлеметсіз бе, сізді Ұлттық қауіпсіздік комитетінен мазалап тұрмыз."),
            ("callee", "Не болды?"),
            ("caller", "Сіздің шотыңыздың күдікті операцияға қатысы бар, тексеру жүргіземіз."),
            ("caller", "Ешкімге айтпаңыз, бұл құпия тергеу."),
        ),
        tactic_ids=("impersonation_gov_police", "fear_threat", "secrecy"),
        trigger_phrases=(
            "Ұлттық қауіпсіздік комитетінен",
            "Ешкімге айтпаңыз, бұл құпия тергеу",
        ),
        is_hard_negative=False,
    ),
    AnchorSpec(
        id="real_scam_mule_recruit_mixed",
        language="mixed",
        turns=(
            ("caller", "Привет, есть лёгкая подработка, керемет ақша табуға болады."),
            ("callee", "А что делать нужно?"),
            ("caller", "Просто прими перевод на свою карту и переведи дальше, за это получишь процент."),
        ),
        tactic_ids=("mule_recruitment", "payment_redirect"),
        trigger_phrases=(
            "есть лёгкая подработка",
            "прими перевод на свою карту и переведи дальше",
        ),
        is_hard_negative=False,
    ),
    AnchorSpec(
        id="real_scam_telecom_verify_ru",
        language="ru",
        turns=(
            ("caller", "Добрый день, оператор связи, ваш договор на номер заканчивается сегодня."),
            ("callee", "Я ничего об этом не получал."),
            ("caller", "Для продления подтвердите личность — назовите код, который придёт в SMS."),
        ),
        tactic_ids=("impersonation_telecom_delivery", "verification_ploy", "otp_request"),
        trigger_phrases=(
            "ваш договор на номер заканчивается сегодня",
            "назовите код, который придёт в SMS",
        ),
        is_hard_negative=False,
    ),
    AnchorSpec(
        id="real_scam_bank_otp_kk",
        language="kk",
        turns=(
            ("caller", "Сәлеметсіз бе, банктің қауіпсіздік қызметінен хабарласып тұрмыз."),
            ("callee", "Иә, тыңдап тұрмын."),
            ("caller", "Картаңызбен күдікті төлем болды, оны тоқтату үшін SMS-тегі кодты айтыңыз."),
        ),
        tactic_ids=("impersonation_bank", "otp_request", "fear_threat"),
        trigger_phrases=(
            "банктің қауіпсіздік қызметінен",
            "SMS-тегі кодты айтыңыз",
        ),
        is_hard_negative=False,
    ),
    AnchorSpec(
        id="real_scam_compensation_creds_ru",
        language="ru",
        turns=(
            ("caller", "Здравствуйте, вам одобрена социальная компенсация от государства."),
            ("callee", "Какая компенсация?"),
            ("caller", "Для перечисления назовите номер карты и три цифры на обороте."),
        ),
        tactic_ids=("prize_lottery", "credentials_request"),
        trigger_phrases=(
            "вам одобрена социальная компенсация",
            "номер карты и три цифры на обороте",
        ),
        is_hard_negative=False,
    ),
    AnchorSpec(
        id="real_scam_crypto_invest_mixed",
        language="mixed",
        turns=(
            ("caller", "Сәлем! Криптовалютаға инвестиция салып, айына 40% табыс табасыз."),
            ("callee", "Это точно надёжно?"),
            ("caller", "Конечно, просто переведите деньги на наш кошелёк по этому QR-коду сейчас."),
        ),
        tactic_ids=("investment_scam", "payment_redirect", "urgency"),
        trigger_phrases=(
            "айына 40% табыс",
            "переведите деньги на наш кошелёк по этому QR-коду",
        ),
        is_hard_negative=False,
    ),
    # --- Phase 1A widening: one dedicated anchor each for previously-starved tail
    # tactics (mule_recruitment, secrecy, remote_access, investment_scam, prize_lottery),
    # so tail-tactic recall on `authored_heldout` becomes statistically measurable. -----------
    AnchorSpec(
        id="real_scam_mule_recruit_reward_ru",
        language="ru",
        turns=(
            ("caller", "Здравствуйте, ищем людей для лёгкой подработки на дому."),
            ("callee", "А что за работа?"),
            ("caller", "Оформите карту в банке на своё имя и получите вознаграждение сразу после оформления."),
            ("caller", "Дальше просто передавайте нам данные карты, через неё будут идти переводы, а вам процент с каждого."),
            ("callee", "А это законно?"),
            ("caller", "Конечно, просто никому не говорите детали этой работы, чтобы не было проблем."),
        ),
        tactic_ids=("mule_recruitment", "credentials_request", "secrecy"),
        trigger_phrases=(
            "Оформите карту в банке на своё имя и получите вознаграждение",
            "передавайте нам данные карты",
            "никому не говорите детали этой работы",
        ),
        is_hard_negative=False,
    ),
    AnchorSpec(
        id="real_scam_tax_refund_secrecy_kk",
        language="kk",
        turns=(
            ("caller", "Сәлеметсіз бе, салық қызметінен хабарласып тұрмыз, сізге артық төленген салық қайтарылады."),
            ("callee", "Қалай аламын?"),
            ("caller", "Қайтару үшін карта нөміріңізді растауымыз керек, бірақ бұл туралы ешкімге айтпаңыз, тексеру құпия жүріп жатыр."),
            ("callee", "Жарайды, айтпаймын."),
            ("caller", "Картаңыздың нөмірі мен мерзімін қазір айтыңыз."),
        ),
        tactic_ids=("impersonation_gov_police", "secrecy", "credentials_request"),
        trigger_phrases=(
            "бұл туралы ешкімге айтпаңыз, тексеру құпия жүріп жатыр",
            "Картаңыздың нөмірі мен мерзімін қазір айтыңыз",
        ),
        is_hard_negative=False,
    ),
    AnchorSpec(
        id="real_scam_remote_access_antivirus_mixed",
        language="mixed",
        turns=(
            ("caller", "Алло, это техподдержка, на вашем компьютере обнаружен вирус, который крадёт данные карты."),
            ("callee", "Ой, а что делать?"),
            ("caller", "Установите AnyDesk қосымшасын қазір, біз қашықтан тазалаймыз."),
            ("caller", "Сессия нөмірін айтыңыз, тек банкке хабарламаңыз, өзіміз шешеміз."),
        ),
        tactic_ids=("remote_access", "fear_threat", "secrecy"),
        trigger_phrases=(
            "Установите AnyDesk қосымшасын қазір",
            "Сессия нөмірін айтыңыз",
            "банкке хабарламаңыз",
        ),
        is_hard_negative=False,
    ),
    AnchorSpec(
        id="real_scam_investment_platform_ru",
        language="ru",
        turns=(
            ("caller", "Здравствуйте, звоним от инвестиционной платформы, где наши клиенты гарантированно зарабатывают."),
            ("callee", "А какая доходность?"),
            ("caller", "Мы гарантируем 25% в месяц без риска, наш аналитик будет вести ваш счёт лично."),
            ("caller", "Для начала переведите первый взнос на счёт платформы прямо сейчас, места в программе заканчиваются."),
        ),
        tactic_ids=("investment_scam", "urgency", "payment_redirect"),
        trigger_phrases=(
            "гарантируем 25% в месяц без риска",
            "переведите первый взнос на счёт платформы прямо сейчас",
        ),
        is_hard_negative=False,
    ),
    AnchorSpec(
        id="real_scam_prize_phone_kk",
        language="kk",
        turns=(
            ("caller", "Құттықтаймыз! Сіздің нөміріңіз SMS байқауында смартфон ұтты."),
            ("callee", "Шынымен бе? Қалай аламын?"),
            ("caller", "Жүлдені алу үшін жеткізу ақысын қазір төлеуіңіз керек, әйтпесе сыйлық басқа жеңімпазға беріледі."),
            ("caller", "Картаңыздың нөмірін айтыңыз, төлемді сол жерден аламыз."),
        ),
        tactic_ids=("prize_lottery", "urgency", "credentials_request"),
        trigger_phrases=(
            "Сіздің нөміріңіз SMS байқауында смартфон ұтты",
            "жеткізу ақысын қазір төлеуіңіз керек",
            "Картаңыздың нөмірін айтыңыз",
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
    AnchorSpec(
        id="real_neg_bank_fraud_alert_ru",
        language="ru",
        turns=(
            ("caller", "Здравствуйте, это банк. По вашей карте была попытка оплаты в другом городе, мы её отклонили."),
            ("callee", "Спасибо, это точно не я."),
            ("caller", "Карту мы уже заблокировали для безопасности. Новую получите в отделении с паспортом — код или данные карты называть не нужно."),
        ),
        tactic_ids=(),
        trigger_phrases=(),
        is_hard_negative=True,
    ),
    AnchorSpec(
        id="real_neg_egov_ready_kk",
        language="kk",
        turns=(
            ("caller", "Сәлеметсіз бе, ЦОН-нан хабарласып тұрмыз, өтінішіңіз дайын."),
            ("callee", "Жақсы, қашан келуге болады?"),
            ("caller", "Ертең сағат онда құжатты алып кетіңіз, өзіңізбен жеке куәлік болсын."),
        ),
        tactic_ids=(),
        trigger_phrases=(),
        is_hard_negative=True,
    ),
    AnchorSpec(
        id="real_neg_friend_borrow_mixed",
        language="mixed",
        turns=(
            ("caller", "Алло, дос, азырақ ақша қарызға берші, айлыққа дейін ғана."),
            ("callee", "Сколько нужно?"),
            ("caller", "Отыз мыңдай, кешке қайтарам, рахмет!"),
        ),
        tactic_ids=(),
        trigger_phrases=(),
        is_hard_negative=True,
    ),
    AnchorSpec(
        id="real_neg_courier_time_ru",
        language="ru",
        turns=(
            ("caller", "Добрый день, курьер, ваш заказ доставлю сегодня после обеда, удобно?"),
            ("callee", "Да, я буду дома."),
            ("caller", "Отлично, оплата при получении, до встречи."),
        ),
        tactic_ids=(),
        trigger_phrases=(),
        is_hard_negative=False,
    ),
    AnchorSpec(
        id="real_neg_telecom_tariff_ru",
        language="ru",
        turns=(
            ("caller", "Здравствуйте, ваш оператор связи, хотим предложить новый тариф выгоднее текущего."),
            ("callee", "Расскажите подробнее."),
            ("caller", "Абонплата ниже, интернета больше. Если интересно, подключим в приложении, никаких кодов не нужно."),
        ),
        tactic_ids=(),
        trigger_phrases=(),
        is_hard_negative=True,
    ),
    AnchorSpec(
        id="real_neg_pharmacy_kk",
        language="kk",
        turns=(
            ("caller", "Сәлеметсіз бе, дәріханадан хабарласып тұрмыз, сіз сұраған дәрі келді."),
            ("callee", "Жақсы, рахмет, барып аламын."),
            ("caller", "Жұмыс уақытында келе беріңіз, сау болыңыз."),
        ),
        tactic_ids=(),
        trigger_phrases=(),
        is_hard_negative=False,
    ),
    AnchorSpec(
        id="real_neg_colleague_meeting_mixed",
        language="mixed",
        turns=(
            ("caller", "Салам, ертеңгі жиналысқа презентацияны дайындадың ба?"),
            ("callee", "Почти закончил, вечером скину."),
            ("caller", "Жақсы, рахмет, асықпай істе."),
        ),
        tactic_ids=(),
        trigger_phrases=(),
        is_hard_negative=False,
    ),
    AnchorSpec(
        id="real_neg_bank_card_ready_ru",
        language="ru",
        turns=(
            ("caller", "Здравствуйте, это банк, ваша новая карта готова к выдаче."),
            ("callee", "А, я заказывал, да."),
            ("caller", "Можете забрать в любом отделении с удостоверением личности. Данные по телефону сообщать не нужно."),
        ),
        tactic_ids=(),
        trigger_phrases=(),
        is_hard_negative=True,
    ),
    AnchorSpec(
        id="real_neg_neighbor_lift_kk",
        language="kk",
        turns=(
            ("caller", "Сәлеметсіз бе, көрші, лифт жөнделді ме, білесіз бе?"),
            ("callee", "Иә, бүгін жөндеп кетті."),
            ("caller", "О, жақсы болған екен, рахмет!"),
        ),
        tactic_ids=(),
        trigger_phrases=(),
        is_hard_negative=False,
    ),
    # --- Phase 1A widening: more legit telecom / delivery / other calls, weighted toward
    # negatives (FPR-first discipline). These superficially resemble
    # `impersonation_telecom_delivery` scams but never request codes/credentials/money. --
    AnchorSpec(
        id="real_neg_telecom_maintenance_ru",
        language="ru",
        turns=(
            ("caller", "Здравствуйте, ваш оператор связи, звоним предупредить о плановых работах на сети."),
            ("callee", "А, хорошо, во сколько будут работы?"),
            ("caller", "Сегодня с двух до четырёх ночи связь в вашем районе может быть нестабильной, это плановое обслуживание."),
            ("caller", "Никаких действий от вас не требуется, данные и коды называть не нужно."),
            ("callee", "Понятно, спасибо за информацию."),
        ),
        tactic_ids=(),
        trigger_phrases=(),
        is_hard_negative=True,
    ),
    AnchorSpec(
        id="real_neg_telecom_contract_renewal_kk",
        language="kk",
        turns=(
            ("caller", "Сәлеметсіз бе, байланыс операторымын, қызмет көрсету шартыңызды еске салайын."),
            ("callee", "Иә, тыңдап тұрмын."),
            ("caller", "Шартыңыз келесі айда аяқталады, қаласаңыз қосымша арқылы өзіңіз ұзарта аласыз."),
            ("caller", "Ешқандай код немесе карта деректерін телефон арқылы сұрамаймыз."),
            ("callee", "Түсінікті, рахмет."),
        ),
        tactic_ids=(),
        trigger_phrases=(),
        is_hard_negative=True,
    ),
    AnchorSpec(
        id="real_neg_telecom_tariff_notice_mixed",
        language="mixed",
        turns=(
            ("caller", "Сәлеметсіз бе, оператор компаниясынан хабарласып тұрмыз. Ваш тариф с этого месяца немного меняется."),
            ("callee", "А что именно меняется?"),
            ("caller", "Интернет лимиті көбейеді, ай сайынғы төлем сол күйінде қалады, ешқандай төлем жасаудың қажеті жоқ."),
            ("callee", "Хорошо, спасибо, что предупредили."),
        ),
        tactic_ids=(),
        trigger_phrases=(),
        is_hard_negative=True,
    ),
    AnchorSpec(
        id="real_neg_telecom_sim_reregistration_ru",
        language="ru",
        turns=(
            ("caller", "Здравствуйте, звонит оператор связи по поводу перерегистрации SIM-карты."),
            ("callee", "А это обязательно?"),
            ("caller", "Да, до конца месяца, но сделать это можно самостоятельно в приложении оператора или в любом салоне связи."),
            ("caller", "По телефону код подтверждения мы никогда не запрашиваем, если кто-то просит код — это не мы."),
            ("callee", "Понял, спасибо, зайду в приложение."),
        ),
        tactic_ids=(),
        trigger_phrases=(),
        is_hard_negative=True,
    ),
    AnchorSpec(
        id="real_neg_courier_window_ru",
        language="ru",
        turns=(
            ("caller", "Добрый день, курьер, буду у вас с трёх до пяти, укажите, пожалуйста, этаж и домофон."),
            ("callee", "Второй этаж, домофон 25."),
            ("caller", "Записал, подъеду в это окно, оплата уже прошла онлайн, ничего дополнительно платить не нужно."),
            ("callee", "Хорошо, буду ждать."),
        ),
        tactic_ids=(),
        trigger_phrases=(),
        is_hard_negative=False,
    ),
    AnchorSpec(
        id="real_neg_courier_address_confirm_kk",
        language="kk",
        turns=(
            ("caller", "Сәлеметсіз бе, жеткізу қызметінен хабарласып тұрмыз, мекенжайыңызды растай аласыз ба?"),
            ("callee", "Иә, Абай көшесі, 12-үй, 5-пәтер."),
            ("caller", "Рахмет, тапсырысыңызды бүгін кешке жеткіземіз, төлем алдын ала жасалған."),
            ("callee", "Жақсы, күтемін."),
        ),
        tactic_ids=(),
        trigger_phrases=(),
        is_hard_negative=False,
    ),
    AnchorSpec(
        id="real_neg_postamat_pickup_mixed",
        language="mixed",
        turns=(
            ("caller", "Здравствуйте, посылка на постамат по улице Сатпаева уже пришла."),
            ("callee", "О, спасибо, я и не заметил уведомление."),
            (
                "caller",
                "Код для получения у вас в SMS от постамата, называть его нам по телефону не нужно, "
                "просто введите его на экране автомата.",
            ),
            ("callee", "Понял, заберу вечером."),
        ),
        tactic_ids=(),
        trigger_phrases=(),
        is_hard_negative=True,
    ),
    AnchorSpec(
        id="real_neg_dental_reminder_kk",
        language="kk",
        turns=(
            ("caller", "Сәлеметсіз бе, тіс емханасынан хабарласып тұрмыз."),
            ("caller", "Ертең сағат он бірде тіс дәрігеріне жазылғаныңызды еске саламыз."),
            ("callee", "Жақсы, есімде, барамын."),
            ("caller", "Рахмет, күтеміз. Сау болыңыз!"),
        ),
        tactic_ids=(),
        trigger_phrases=(),
        is_hard_negative=True,
    ),
    AnchorSpec(
        id="real_neg_bank_card_delivery_kk",
        language="kk",
        turns=(
            ("caller", "Сәлеметсіз бе, банктен хабарласып тұрмыз, жаңа картаңыз дайын болды."),
            ("callee", "Жақсы, қашан алуға болады?"),
            ("caller", "Кез келген бөлімшеден жеке куәлігіңізбен ала аласыз, картаның деректерін телефон арқылы айтудың қажеті жоқ."),
            ("callee", "Түсінікті, рахмет."),
        ),
        tactic_ids=(),
        trigger_phrases=(),
        is_hard_negative=True,
    ),
    AnchorSpec(
        id="real_neg_weekend_plans_mixed",
        language="mixed",
        turns=(
            ("caller", "Алло, сен ертең бос па? Есть идея сходить в кино."),
            ("callee", "Иә, бос болам, во сколько встречаемся?"),
            ("caller", "Сағат жетіде, кинотеатрдың алдында кездесейік."),
            ("callee", "Жарайды, жетіге дейін көрісеміз!"),
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
