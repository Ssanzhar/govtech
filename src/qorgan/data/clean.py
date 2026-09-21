"""Generation-artefact repair for corpus text (ADR D34). Found on 2026-09-20 while styling
the corpus for ASR realism: fifteen synthetic rows carried artefacts of the generator's
output handling -- utterances wrapped in `"\\r\\n … "`, backspace characters inside words,
literal `\\uXXXX` escapes -- and two Latin-script Kazakh rows had lost letters to control
characters (`S\\ntyzba`), which no rule can put back. `clean_text` repairs what is
recoverable; `is_corrupted` names what is not, and `build_corpus` drops those rows and
records them in the manifest. Pure and idempotent, like `scrub_text`.
"""

from __future__ import annotations

import re

from qorgan.data.schema import Dialogue, Label, Utterance

_LITERAL_UNICODE_ESCAPE = re.compile(r"\\u([0-9a-fA-F]{4})")
_CRLF = "\r\n"                       # the generator's line ending around wrapped turns
_BACKSPACE = "\x08"                   # typed-over characters inside words: deletable
_WHITESPACE = re.compile(r"[ \t]+")
_WRAPPING_QUOTE = '"'
_TURN_BREAK = re.compile(r"\n+")
# A C0 control that survives the repairs (a lone `\r`, a form feed) or a newline glued
# between letters stands where a letter was: not formatting, a lost character.
_CONTROL = re.compile("[\x00-\x08\x0b-\x1f]")
_NEWLINE_INSIDE_WORD = re.compile(r"(?<=\w)\n(?=\w)")
_SPEAKERS = ("caller", "callee")


def clean_text(text: str) -> str:
    """Repair generation artefacts in one utterance: decode literal `\\uXXXX`, turn `\\r\\n`
    into `\\n`, drop backspaces, strip wrapping straight quotes and whitespace, collapse
    spaces. Internal newlines (turn breaks the generator jammed into one utterance) survive
    for `clean_dialogue` to split on."""
    decoded = _LITERAL_UNICODE_ESCAPE.sub(lambda m: chr(int(m.group(1), 16)), text)
    stripped = _strip_wrapping(decoded.replace(_CRLF, "\n").replace(_BACKSPACE, ""))
    return "\n".join(_WHITESPACE.sub(" ", line).strip() for line in stripped.split("\n")).strip()


def _strip_wrapping(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith(_WRAPPING_QUOTE) and stripped.endswith(_WRAPPING_QUOTE) and len(stripped) >= 2:
        return stripped[1:-1].strip()
    if stripped.startswith(_WRAPPING_QUOTE) and stripped.count(_WRAPPING_QUOTE) == 1:
        return stripped[1:].strip()
    if stripped.endswith(_WRAPPING_QUOTE) and stripped.count(_WRAPPING_QUOTE) == 1:
        return stripped[:-1].strip()
    return stripped


def is_corrupted(text: str) -> bool:
    """True when control characters stand where letters were: nothing `clean_text` does can
    recover the word, so the row must not be trained or evaluated on."""
    return bool(_CONTROL.search(text) or _NEWLINE_INSIDE_WORD.search(text))


def clean_dialogue(dialogue: Dialogue) -> Dialogue | None:
    """The dialogue with every utterance repaired -- an utterance the generator wrapped
    several turns into is split back into turns (speakers alternate from the original) --
    or `None` if any utterance is corrupted beyond repair; a turn that empties out (a stray
    quote line) is simply removed. Trigger spans are
    kept only when no text changed (the scrub step re-grounds them afterwards)."""
    cleaned: list[Utterance] = []
    for utterance in dialogue.utterances:
        text = clean_text(utterance.text)
        if is_corrupted(text):
            return None
        pieces = [_strip_wrapping(piece) for piece in _TURN_BREAK.split(text)]
        pieces = [piece for piece in pieces if piece]
        if not pieces:
            continue  # a stray quote or blank line the generator emitted as a turn
        start = _SPEAKERS.index(utterance.speaker) if utterance.speaker in _SPEAKERS else 0
        for offset, piece in enumerate(pieces):
            speaker = utterance.speaker if offset == 0 else _SPEAKERS[(start + offset) % len(_SPEAKERS)]
            cleaned.append(Utterance(speaker=speaker, text=piece))
    if not cleaned:
        return None
    unchanged = len(cleaned) == len(dialogue.utterances) and all(u.text == c.text for u, c in zip(dialogue.utterances, cleaned))
    label = Label(
        risk=dialogue.label.risk,
        tactic_tags=dialogue.label.tactic_tags,
        trigger_spans=dialogue.label.trigger_spans if unchanged else (),
        is_hard_negative=dialogue.label.is_hard_negative,
    )
    return dialogue.model_copy(update={"utterances": tuple(cleaned), "label": label})


def main(argv: list[str] | None = None) -> None:
    """CLI: `python -m qorgan.data.clean <split.jsonl> [...]` -- repair each JSONL in place,
    drop unrecoverable rows, print what changed. For eval files that `build_corpus` does not
    produce (the July `ood.jsonl`); idempotent."""
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Repair generation artefacts in dialogue JSONL files (ADR D34).")
    parser.add_argument("paths", nargs="+", type=Path)
    args = parser.parse_args(argv)
    for path in args.paths:
        rows = [Dialogue.model_validate_json(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        kept, repaired, dropped = [], 0, []
        for row in rows:
            cleaned = clean_dialogue(row)
            if cleaned is None:
                dropped.append(row.id)
                continue
            repaired += cleaned != row
            kept.append(cleaned)
        path.write_text("".join(d.model_dump_json() + "\n" for d in kept), encoding="utf-8")
        print(f"{path}: {len(rows)} rows, repaired {repaired}, dropped {len(dropped)}" + (f" ({', '.join(dropped)})" if dropped else ""))


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
