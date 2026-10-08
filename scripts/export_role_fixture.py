"""Golden fixture for the JS role slot (ADR D63).

    python scripts/export_role_fixture.py   ->  tests_js/fixtures/roles.json

`site/core/role.js` must return the same claims (role id + span offsets) and the same
rendered line in every locale as `qorgan.explain.roles`; `tests_js/role.test.mjs` asserts it
and `tests/explain/test_role_fixture.py` keeps the committed file current. Cases: hand-written
claims (overlap, order, ASR-style spelling, no claim) plus every 20th `train` transcript when
the corpus is present (train is published; no held-out split is ever read here).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from qorgan.explain.roles import detect_roles, load_role_lexicon, render_role_line, role_name  # noqa: E402
from qorgan.explain.templates import load_templates  # noqa: E402

OUT = REPO / "tests_js/fixtures/roles.json"
TRAIN = REPO / "data/processed/train.jsonl"
TRAIN_STRIDE = 20
LOCALES = ("ru", "kk", "en")

HAND_CASES = (
    "Здравствуйте, вас беспокоит служба безопасности банка.\nУ вас подозрительная операция.",
    "Это курьер, вам посылка.\nСейчас вас соединят, беспокоит следователь.",
    "Вас беспокоит антифрод-центр Национального банка.",
    "Сәлеметсіз бе, мен ҰҚК қызметкерімін. Сізге қатысты іс бар.",
    "Сәлеметсіз бе, ХҚКО-дан хабарласып тұрмын, ЭЦҚ мерзіміңіз бітті.",
    "Hello, I am calling from Kaspi about your card.",
    "Hi, this is the courier, I'm at your door.",
    "Мам, это я, у меня новый номер, срочно нужны деньги.",
    "Апа, бұл менмін, нөмірім ауысты.",
    "Добрый день, это ваш директор, нужна помощь, только никому.",
    "Алло, служба безопасности каспи беспокоит вас.",
    "Слушай, вечером увидимся? Возьми хлеб.",
    "Здравствуйте, это поликлиника номер 5, регистратура. Напоминаем о приёме.",
    "Сізге Қазпошта арқылы сәлемдеме келді.",
    "Я по объявлению на OLX, хочу купить диван.",
    "",
)


def _cases() -> list[str]:
    texts = [t for t in HAND_CASES if t.strip()]
    if TRAIN.exists():
        rows = [json.loads(line) for line in TRAIN.read_text(encoding="utf-8").splitlines() if line.strip()]
        texts += ["\n".join(u["text"] for u in r["utterances"]) for r in rows[::TRAIN_STRIDE] if not r["id"].endswith("-asr")]
    return texts


def build() -> dict:
    lexicon = load_role_lexicon(REPO / "data/lexicon/roles.yaml")
    templates = {loc: load_templates(loc, REPO / "src/qorgan/explain") for loc in LOCALES}
    cases = []
    for text in _cases():
        claims = detect_roles(text, lexicon)
        first = claims[0] if claims else None
        cases.append({
            "transcript": text,
            "claims": [{"role_id": c.role_id, "span": {"text": c.span.text, "start": c.span.start, "end": c.span.end}} for c in claims],
            "lines": {
                loc: render_role_line(templates[loc].role_template, role_name(first.role_id, loc, lexicon), first.span.text) if first else None
                for loc in LOCALES
            },
        })
    return {"version": 1, "cases": cases}


def main() -> None:
    OUT.write_text(json.dumps(build(), ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
