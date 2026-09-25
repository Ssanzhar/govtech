"""Bundled demo transcripts for the Day-1 Streamlit app and the `mock` classifier backend.

Kept deliberately small (a handful of hand-authored calls) so the app and the `mock`
backend work with zero setup: no API key, no downloaded models, no generated corpus.
Full synthetic corpus generation lands Day 2 (`data/generate.py`).
"""

from __future__ import annotations

DEMO_TRANSCRIPTS: dict[str, str] = {
    "scam_bank_ru": (
        "Алло, это служба безопасности вашего банка. По вашей карте зафиксирована "
        "подозрительная операция, действовать нужно прямо сейчас. Никому не говорите об "
        "этом звонке, даже сотрудникам банка. Продиктуйте код из SMS, чтобы отменить "
        "операцию, и переведите деньги на безопасный счёт."
    ),
    "scam_investment_kk_ru": (
        "Сәлем! Сізге тиесілі ұтыс бар, экономикалық қиындықтан құтылу үшін бірлескен "
        "инвестиция ұсынамыз. Гарантированный доход 30% в месяц, но для оформления нужно "
        "оплатить по этому QR-коду прямо сейчас, иначе предложение сгорит."
    ),
    "hard_negative_bank_call_ru": (
        "Здравствуйте, это оператор банка. Мы подтверждаем, что перевод на 15000 тенге в "
        "пользу Ивана И. успешно проведён. Если вы не совершали эту операцию, позвоните "
        "на официальный номер банка на обратной стороне карты."
    ),
}

# Turn-by-turn scripts for the live-call replay (one committed utterance per line, the
# caller's side of the conversation). The scam script deliberately contains verbatim
# taxonomy example phrases so the zero-setup `mock` backend detects them; the hard
# negative is the "real bank call must NOT trigger" demo scene.
LIVE_DEMO_CALLS: dict[str, str] = {
    "live_scam_bank_ru": (
        "Алло, здравствуйте. Это служба безопасности вашего банка.\n"
        "По вашей карте зафиксирована подозрительная операция на крупную сумму.\n"
        "Действовать нужно прямо сейчас, у вас есть только пять минут.\n"
        "Никому не говорите, это секретная операция под контролем регулятора.\n"
        "Продиктуйте код из SMS, чтобы мы отменили операцию.\n"
        "Теперь переведите деньги на безопасный счёт, я продиктую реквизиты."
    ),
    # Kazakh scam — the scene that demonstrates the bilingual claim. Same script the ASR bench
    # drives through the recogniser (scripts/spikes/vosklet_bench, ADRs D25/D26), so replay and
    # microphone mode show the same call.
    "live_scam_bank_kk": (
        "Сәлеметсіз бе. Бұл банктің қауіпсіздік қызметі.\n"
        "Сіздің картаңыздан күдікті операция тіркелді, жағдай шұғыл.\n"
        "Ешкімге айтпаңыз, бұл құпия операция.\n"
        "SMS-тегі кодты айтыңыз, біз операцияны тоқтатамыз.\n"
        "Содан кейін ақшаны қауіпсіз шотқа аударамыз, реквизиттерін айтамын."
    ),
    "live_hard_negative_bank_ru": (
        "Здравствуйте, это оператор банка, звоню по вашей заявке.\n"
        "Мы подтверждаем, что перевод на 15000 тенге успешно проведён.\n"
        "Никакие коды и данные карты называть не нужно.\n"
        "Если вы не совершали эту операцию, позвоните на официальный номер банка "
        "на обратной стороне карты."
    ),
}
