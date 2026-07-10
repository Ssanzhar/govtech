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
