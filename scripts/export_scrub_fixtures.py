"""Generate `tests_js/fixtures/scrub.json`: `scrub_text` outputs the JS port
(`site/core/report.js::scrubText`) must reproduce exactly.

Run: `python scripts/export_scrub_fixtures.py`. `tests/data/test_scrub_fixture.py` fails
when the committed fixture no longer matches Python, so the two cannot drift silently.
"""

from __future__ import annotations

import json
from pathlib import Path

from qorgan.data.scrub import scrub_text

FIXTURE = Path(__file__).resolve().parents[1] / "tests_js" / "fixtures" / "scrub.json"

CASES: tuple[str, ...] = (
    "Перезвоните на +7 777 123 45 67 пожалуйста",
    "Перезвоните на 8 (727) 123-45-67 пожалуйста",
    "номер 87071234567 записан",
    "номер 77071234567 записан",
    "+77071234567",
    "карта 4400 1234 5678 9010, срок 12/27",
    "карта 4400-1234-5678-9010",
    "карта 4400123456789010",
    "ИИН 940101300123 и email test@bank.kz",
    "почта ivan.petrov+kaspi@mail.kz, перезвоните",
    "Код подтверждения 123456 никому не говорите",
    "Переведите 5000 тенге, скидка 30%",
    "12345678901 — одиннадцать цифр не с 7 или 8",
    "1234567890123 — тринадцать цифр",
    "карта4400123456789010 без пробела",
    "ЖСН940101300123 бірге жазылған",
    "ИИН940101300123",
    "номер87071234567записан",
    "a1b2c3940101300123d4",
    "f4400123456789010e",
    "report_940101300123",
    "rcpt-4f2c9a1b77",
    "Қоңырау шалыңыз +7 701 555 12 34, ешкімге айтпаңыз",
    "Позвоните +7 777 123 45 67 или карта 4400 1234 5678 9010",
    "[PHONE] уже заменён, [CARD] тоже",
    "Без PII здесь ничего нет",
    "",
    "   ",
)


def main() -> None:
    payload = {
        "generated_by": "scripts/export_scrub_fixtures.py",
        "cases": [{"input": text, "expected": scrub_text(text)} for text in CASES],
    }
    FIXTURE.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(CASES)} cases -> {FIXTURE}")


if __name__ == "__main__":
    main()
