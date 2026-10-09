"""ADR D66: self-introduced personal names are masked after a few unambiguous patterns,
conservatively. Only the name is replaced; the introduction and the claimed organisation
(the impersonation evidence) stay verbatim.

- strong triggers («меня зовут», «моё имя», «менің атым», «атым», «есімім», "my name is") take
  the next word in any case (recogniser output is lower-case) plus up to two more capitalised
  words or lower-case words with a surname ending;
- weak triggers («это X из …», "this is X from …", "I'm X from …") need a capitalised name and
  the following «из / с / от» / "from / calling / with"; a stop list keeps organisations out.
"""

import pytest

from qorgan.data.scrub import NAME_PLACEHOLDER, scrub_text


def test_placeholder():
    assert NAME_PLACEHOLDER == "[NAME]"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (
            "Здравствуйте, меня зовут Асет, я из службы безопасности банка.",
            "Здравствуйте, меня зовут [NAME], я из службы безопасности банка.",
        ),
        ("Меня зовут Алексей Петрович Иванов.", "Меня зовут [NAME]."),
        ("меня зовут алексей я из банка", "меня зовут [NAME] я из банка"),
        ("моё имя Айгерим", "моё имя [NAME]"),
        ("Менің атым — Айгүл, мен банктен хабарласып тұрмын", "Менің атым — [NAME], мен банктен хабарласып тұрмын"),
        ("Атым – Ерлан Сапаров.", "Атым – [NAME]."),
        ("My name is Elena Petrova from the bank", "My name is [NAME] from the bank"),
        ("my name is elena petrova from the bank", "my name is [NAME] from the bank"),
        ("my name is alex and I am calling", "my name is [NAME] and I am calling"),
        ("Это Канат из Capital Bank", "Это [NAME] из Capital Bank"),
        ("это Алмагуль из Egov", "это [NAME] из Egov"),
        ("Hello, this is Damir from Kaspi", "Hello, this is [NAME] from Kaspi"),
        ("I'm Aizhan from the security service", "I'm [NAME] from the security service"),
    ],
)
def test_self_introduced_names_are_masked(text, expected):
    assert scrub_text(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "это мы из банка",
        "Это посылка из интернет-магазина",
        "это аскар из homebank",  # weak trigger, lower case: too ambiguous, left alone
        "Это Служба безопасности из банка",
        "Атым ба?",
        "this is the bank calling",
        "Меня зовут не так",
        "служба безопасности банка, назовите код из SMS",
        "My name is [NAME] from the bank",
        # organisations are impersonation evidence, not names (found on train, ADR D66)
        "Hello, this is Kazakhstan Post calling about a package for you",
        "this is Kaspi calling",
        "Это Каспи из приложения",
        "this is Halyk Bank calling",
    ],
)
def test_ambiguous_or_name_free_text_is_untouched(text):
    assert scrub_text(text) == text


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        # a rank or title is impersonation evidence: it stays, only the name goes
        (
            "this is Major Ivanov from the Financial Crimes Unit",
            "this is Major [NAME] from the Financial Crimes Unit",
        ),
        ("Это майор Сапаров из полиции", "Это майор [NAME] из полиции"),
        ("Это Канат Сапаров из банка", "Это [NAME] из банка"),
    ],
)
def test_a_title_stays_and_only_the_name_is_masked(text, expected):
    assert scrub_text(text) == expected


def test_names_and_numbers_together():
    text = "Меня зовут Асет, мой номер восемь семь ноль семь один два три четыре пять шесть семь"
    assert scrub_text(text) == "Меня зовут [NAME], мой номер [PHONE]"
