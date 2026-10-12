"""`court_quotes`: real scammer phrasing quoted or reported in published court verdicts (ADR D68).

Held-out, positives-only, utterance-level. Every row is a verbatim span of a published
verdict (direct speech in quotes, or the reported content after «сообщил, что» / «о том,
что»), never rewritten. Rows are PII-scrubbed with `scrub_text`; any row that still carries
a residual identifier (a name with initials, an address, a long digit run) is **dropped**,
not edited. The text lives only in `data/raw/court_quotes/` (gitignored, never published,
`hf_upload.py` refuses it); the committed manifest carries counts and citations, no text.

Rules (CLAUDE.md, ADRs D35/D43): never tune on this set, never extend the lexicon from it.
The extraction rules below were written without reading `data/lexicon/`.
"""

from __future__ import annotations

import html as html_lib
import json
import re
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

from qorgan.data.scrub import scrub_text

SPLIT_NAME = "court_quotes"
RAW_SUBDIR = Path("raw") / "court_quotes"  # under config.data_dir; gitignored
QUOTES_FILENAME = "quotes.jsonl"
MANIFEST_RELPATH = Path("data") / "manifests" / "court_quotes.manifest.json"  # committed, no text

Speech = Literal["direct", "reported"]

MIN_CHARS = 20
MAX_CHARS = 350
MIN_WORDS = 4

# A verdict is only mined when it is about an impersonation call: a call AND a caller who
# "introduced themselves as" an official, a professional or a relative.
_CALL_DOC = re.compile(r"позвонил|звонок|звонк|по телефону|телефонн\w* разговор|мессенджер|whatsapp|ватсап|телеграм", re.I)
_SCHEME_DOC = re.compile(
    r"(?:представил(?:ся|ась|ись)|представившись|представивш\w+|под видом)[^.]{0,60}?"
    r"(?:сотрудник|следовател|работник|специалист|полицейск|оперуполномоченн|юрист|адвокат|врач"
    r"|сын|сыном|дочер|дочь|внук|внучк|родственник|прокурор|начальник)",
    re.I,
)
# Names and organisations the court itself masked («...», (...)) -- kept as a neutral marker.
_COURT_MASK = re.compile(r"«\s*\.\.\.\s*»|\(\s*\.\.\.\s*\)|«\s*…\s*»")
COURT_MASK_TOKEN = "«…»"
# A sentence that does not name the caller may still continue the previous one's speaker.
_PRONOUN_SUBJECT = re.compile(r"^(?:он|она|тот|та|мужчина|женщина|девушка|парень|звонивш\w*|собеседни\w*)\b", re.I)
# The sentence must tie the speech to the caller, not to the defendant, a witness or the court.
_CALLER_CUE = re.compile(
    r"неустановленн|соучастник|звонивш|собеседник|представ(?:ился|илась|ились|ившись|ившийся|ившаяся)"
    r"|по телефону|телефонн\w* разговор|в ходе разговора|позвонил|позвонила|мессенджер",
    re.I,
)
_SPEECH_VERB = (
    r"(?:сообщил|сообщила|сообщили|сообщал|сообщала|пояснил|пояснила|сказал|сказала|сказали"
    r"|заявил|заявила|говорил|говорила|убедил|убедила|убеждал|убеждала|уверил|уверила|объяснил|объяснила)"
)
_REPORTED_THAT = re.compile(_SPEECH_VERB + r"[^.;]{0,120}?(?:о том,?\s+)?что\s+(?P<body>[^;]{%d,%d}?)(?=[.;]|$)" % (MIN_CHARS, 600), re.I)
_REQUEST = re.compile(
    r"(?:попросил|попросила|потребовал|потребовала|предложил|предложила|просил|просила|велел|велела)"
    r"\s+(?P<body>[^.;]{%d,%d}?)(?=[.;]|$)" % (MIN_CHARS, 600),
    re.I,
)
_DIRECT = re.compile(r"[«“\"](?P<body>[^«»“”\"]{%d,%d})[»”\"]" % (MIN_CHARS, MAX_CHARS))
_DIRECT_LEAD = re.compile(_SPEECH_VERB + r"|спросил|спросила|произн[её]с|произнесла|кричал|плакал|ответил|ответила|:\s*$", re.I)
# Court framing, not the caller's words.
_LEGAL_FRAMING = re.compile(
    r"преступн|умысел|умысл|корыстн|подсудим|осужд|уголовн\w* дел|обвиняем|потерпевш\w+ признан"
    r"|куратор|вину признал|в судебном заседании|оглашен|протокол",
    re.I,
)

# Residual identifiers the digit scrubber cannot see. A hit drops the row.
_INITIALS = re.compile(r"\b[А-ЯЁA-Z]\.\s?[А-ЯЁA-Z]\.")
_SURNAME_INITIAL = re.compile(r"\b[А-ЯЁ][а-яё]{2,}(?:ов|ова|ев|ева|ин|ина|ский|ская|енко|ук|юк)\s+[А-ЯЁ]\.")
_ADDRESS = re.compile(
    r"\b(?:ул\.|улиц\w|пр-т|проспект\w*|переул\w*|мкр\.?|микрорайон\w*|д\.\s?\d|кв\.\s?\d"
    r"|дом\w*\s+№?\s?\d|квартир\w*\s+№?\s?\d|подъезд\w*\s+№?\s?\d)",
    re.I,
)
_LONG_DIGITS = re.compile(r"\d[\d\s\-]{6,}\d")
_EMAIL_LIKE = re.compile(r"@")
_PLATE_OR_ID = re.compile(r"\b[А-ЯЁA-Z]{1,2}\d{3,}[А-ЯЁA-Z]{0,3}\b")


@dataclass(frozen=True)
class Passage:
    text: str
    speech: Speech


@dataclass(frozen=True)
class VerdictMeta:
    verdict_id: str
    source: str  # e.g. "sudact.ru"
    url: str
    court: str
    date: str  # ISO yyyy-mm-dd, or "" when the title carries none
    title: str


@dataclass(frozen=True)
class QuoteRow:
    id: str
    language: str
    text: str
    speech: Speech
    tactic_guess: str
    source: str
    source_url: str
    court: str
    date: str
    verdict_id: str


_MONTHS = {
    "января": 1, "февраля": 2, "марта": 3, "апреля": 4, "мая": 5, "июня": 6,
    "июля": 7, "августа": 8, "сентября": 9, "октября": 10, "ноября": 11, "декабря": 12,
}


def _date_from_title(title: str) -> str:
    match = re.search(r"от\s+(\d{1,2})\s+([а-я]+)\s+(\d{4})", title)
    if not match or match.group(2) not in _MONTHS:
        return ""
    return f"{int(match.group(3)):04d}-{_MONTHS[match.group(2)]:02d}-{int(match.group(1)):02d}"


def _plain(fragment: str) -> str:
    fragment = re.sub(r"<(script|style)\b.*?</\1>", " ", fragment, flags=re.S | re.I)
    fragment = re.sub(r"<br\s*/?>|</p>|</div>", "\n", fragment, flags=re.I)
    fragment = re.sub(r"<[^>]+>", " ", fragment)
    fragment = html_lib.unescape(fragment)
    lines = [re.sub(r"[ \t ]+", " ", line).strip() for line in fragment.split("\n")]
    return "\n".join(line for line in lines if line)


def parse_sudact(html: str, verdict_id: str) -> tuple[VerdictMeta, str]:
    """Split a sudact.ru `/regular/doc/<id>/` page into citation metadata and plain verdict text."""
    title_match = re.search(r"<h1>(.*?)</h1>", html, re.S)
    title = _plain(title_match.group(1)) if title_match else ""
    court_match = re.search(r'<div class="b-justice">\s*<a[^>]*>(.*?)</a>', html, re.S)
    court = _plain(court_match.group(1)) if court_match else ""
    start = html.find('<hr class="hr-h1">')
    body = html[start:] if start >= 0 else html
    end = body.find('<div class="h-footer"')
    body = body[:end] if end >= 0 else body
    meta = VerdictMeta(
        verdict_id=verdict_id,
        source="sudact.ru",
        url=f"https://sudact.ru/regular/doc/{verdict_id}/",
        court=court,
        date=_date_from_title(title),
        title=title,
    )
    return meta, _plain(body)


_CRIMINAL_TITLE = re.compile(
    r"^(?:Приговор|Апелляционн\w+ (?:приговор|определение|постановление)|Кассационн\w+ (?:определение|постановление))",
    re.I,
)


def is_criminal_act(meta: VerdictMeta) -> bool:
    """Only criminal judgments are mined (a civil unjust-enrichment ruling is not a verdict)."""
    return bool(_CRIMINAL_TITLE.search(meta.title))


def _sentences(text: str) -> list[str]:
    out: list[str] = []
    for paragraph in text.split("\n"):
        out.extend(part.strip() for part in re.split(r"(?<=[.!?])\s+(?=[А-ЯЁ«])", paragraph) if part.strip())
    return out


def _trim(body: str) -> str:
    body = re.sub(r"\s+", " ", body).strip(" ,:;—-")
    if len(body) > MAX_CHARS:
        cut = body[:MAX_CHARS]
        body = cut[: cut.rfind(",")] if "," in cut[MAX_CHARS // 2 :] else cut[: cut.rfind(" ")]
    return body.strip(" ,:;—-")


# Court-template residue ("ДД.ММ.ГГГГ", case-file references) is not speech.
_TEMPLATE_RESIDUE = re.compile(r"ДД\.ММ|ГГГГ|\(т\.\s?\d|л\.д\.|\bт\.\s?\d+\s?л")


def _usable(body: str) -> bool:
    return (
        MIN_CHARS <= len(body) <= MAX_CHARS
        and len(body.split()) >= MIN_WORDS
        and not _LEGAL_FRAMING.search(body)
        and not _TEMPLATE_RESIDUE.search(body)
    )


def extract_passages(text: str) -> tuple[Passage, ...]:
    """Caller speech in a verdict: direct quotes first, then reported content. Deduplicated."""
    if not (_CALL_DOC.search(text) and _SCHEME_DOC.search(text)):
        return ()
    sentences = _sentences(_COURT_MASK.sub(COURT_MASK_TOKEN, text))
    seen: set[str] = set()
    found: list[Passage] = []

    def add(body: str, speech: Speech) -> None:
        body = _trim(body)
        key = re.sub(r"\W+", " ", body.lower()).strip()
        if _usable(body) and key not in seen:
            seen.add(key)
            found.append(Passage(text=body, speech=speech))

    for index, sentence in enumerate(sentences):
        continues_caller = (
            index > 0 and _PRONOUN_SUBJECT.search(sentence) and _CALLER_CUE.search(sentences[index - 1])
        )
        if not (_CALLER_CUE.search(sentence) or continues_caller):
            continue
        for match in _DIRECT.finditer(sentence):
            lead = sentence[max(0, match.start() - 60) : match.start()]
            if _DIRECT_LEAD.search(lead) and len(match.group("body").split()) >= MIN_WORDS:
                add(match.group("body"), "direct")
        if _LEGAL_FRAMING.search(sentence.split(" что ")[0]) and "неустановленн" not in sentence.lower():
            continue  # a defendant's or witness's statement, not the caller's
        for pattern in (_REPORTED_THAT, _REQUEST):
            for match in pattern.finditer(sentence):
                add(match.group("body"), "reported")
    return tuple(found)


def residual_pii(text: str) -> bool:
    """True when a scrubbed passage still carries an identifier the digit scrubber misses."""
    return any(
        pattern.search(text)
        for pattern in (_INITIALS, _SURNAME_INITIAL, _ADDRESS, _LONG_DIGITS, _EMAIL_LIKE, _PLATE_OR_ID)
    )


def clean_quote(text: str) -> str | None:
    """`scrub_text`, then drop (return None) anything the scrubber could not make safe."""
    scrubbed = scrub_text(text)
    if residual_pii(scrubbed) or "[IIN]" in scrubbed or "[CARD]" in scrubbed or "[EMAIL]" in scrubbed:
        return None
    return scrubbed


# Free-text tactic guess (deliberately NOT our tactic ids, and not derived from our lexicon).
_TACTIC_GUESSES: tuple[tuple[str, str], ...] = (
    (r"безопасн\w* сч|резервн\w* сч|защищ\w* сч|специальн\w* сч", "safe-account transfer"),
    (r"кредит|заявк\w* на (?:получение )?(?:кредит|заем|займ)", "loan in victim's name"),
    (r"код|пароль|смс|sms", "one-time code request"),
    (r"сын|доч|внук|внучк|родствен|мам|брат|сестр", "relative in trouble"),
    (r"дтп|авари|сбил|сбила|пострадав|больниц", "accident / injury pretext"),
    (r"полици|следовател|фсб|прокурат|уголовн|задерж|суд\b|ответственност", "law-enforcement pressure"),
    (r"банк|служб\w* безопасности|центральн\w* банк|цб\b", "bank / security-service impersonation"),
    (r"пенси|соцзащит|социальн|выплат|компенсац", "pension / benefit pretext"),
    (r"госуслуг|портал|аккаунт|личн\w* кабинет", "government-portal account pretext"),
    (r"замен\w* (?:замк|счетчик|домофон)|жилищн|управляющ", "utility / housing pretext"),
    (r"инвест|бирж|доход|заработ|криптовалют", "investment lure"),
    (r"курьер|передать|перед\w+ деньг|наличн", "cash hand-off"),
    (r"срочно|немедленно|быстро|сейчас же", "urgency"),
)


def tactic_guess(text: str) -> str:
    lowered = text.lower()
    hits = [label for pattern, label in _TACTIC_GUESSES if re.search(pattern, lowered)]
    return "; ".join(hits[:3]) if hits else "unspecified"


def build_rows(meta: VerdictMeta, text: str, *, language: str = "ru") -> tuple[tuple[QuoteRow, ...], int]:
    """(rows, n_dropped) for one verdict; uncleanable passages are dropped, never repaired by hand."""
    rows: list[QuoteRow] = []
    dropped = 0
    if meta.source == "sudact.ru" and not is_criminal_act(meta):
        return (), 0
    for passage in extract_passages(text):
        cleaned = clean_quote(passage.text)
        if cleaned is None:
            dropped += 1
            continue
        rows.append(
            QuoteRow(
                id=f"cq-{meta.source.split('.')[0]}-{meta.verdict_id}-{len(rows):02d}",
                language=language,
                text=cleaned,
                speech=passage.speech,
                tactic_guess=tactic_guess(cleaned),
                source=meta.source,
                source_url=meta.url,
                court=meta.court,
                date=meta.date,
                verdict_id=meta.verdict_id,
            )
        )
    return tuple(rows), dropped


def write_rows(rows: Iterable[QuoteRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(asdict(r), ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def load_quotes(path: Path) -> tuple[QuoteRow, ...]:
    """Load the local split; raises FileNotFoundError when it was never collected here."""
    rows = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            rows.append(QuoteRow(**json.loads(line)))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{path.name} line {number} is not a court_quotes row") from exc
    return tuple(rows)


def build_manifest(rows: Sequence[QuoteRow], *, collected_on: str, dropped_pii: int, verdicts_read: int) -> dict:
    """Counts, citations and dates only -- the manifest is committed, so it holds no quote text."""
    citations = sorted({(r.source, r.source_url, r.court, r.date, r.verdict_id) for r in rows})
    dates = sorted(r.date for r in rows if r.date)
    return {
        "split": SPLIT_NAME,
        "adr": "D68",
        "collected_on": collected_on,
        "label": "scam (positives only; utterance-level)",
        "n_quotes": len(rows),
        "by_language": dict(Counter(r.language for r in rows)),
        "by_speech": dict(Counter(r.speech for r in rows)),
        "by_source": dict(Counter(r.source for r in rows)),
        "verdicts_read": verdicts_read,
        "verdicts_with_quotes": len({r.verdict_id for r in rows}),
        "dropped_residual_pii": dropped_pii,
        "verdict_date_range": [dates[0], dates[-1]] if dates else [],
        "citations": [
            {"source": s, "url": u, "court": c, "date": d, "verdict_id": v} for s, u, c, d, v in citations
        ],
        "text_location": str(Path("data") / RAW_SUBDIR / QUOTES_FILENAME) + " (gitignored, local only)",
    }
