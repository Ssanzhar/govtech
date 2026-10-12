"""The row a generating agent writes, its validation against its grid cell, and its conversion
to the corpus `Dialogue` (ADR D63).

A generated row is the existing `Dialogue` shape plus per-utterance annotations: each
utterance carries its `stage` and the tactics it shows, each as a VERBATIM phrase of that
utterance. Validation is strict where a mistake would corrupt a label (unknown tactic, a
phrase that is not verbatim, a negative with tactics, a scam missing its required tactics)
and advisory where it is a style matter (length off target, script mismatch).
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from qorgan.data.gen.grid import GridCell
from qorgan.data.schema import HARD_NEGATIVE_RISK, Dialogue, Label, Span, TacticTag, Utterance

# Generation-time risk of a scam row, as `data/generate.py` assigns (`_SEED_POSITIVE_RISK`).
SCAM_RISK = 0.9
SPEAKERS = ("caller", "callee")
STAGES = ("hook", "escalation", "extraction", "handoff", "none")
# Length is advisory within this tolerance: max(2 utterances, 30 % of the target).
_LENGTH_TOLERANCE_ABS = 2
_LENGTH_TOLERANCE_REL = 0.30
_KK_LETTERS = re.compile(r"[әғқңөұүһіӘҒҚҢӨҰҮҺІ]")
_CYRILLIC = re.compile(r"[А-Яа-яЁё]")
_LATIN = re.compile(r"[A-Za-z]")


class TacticPhrase(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    phrase: str = Field(min_length=1)


class GenUtterance(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    speaker: Literal["caller", "callee"]
    text: str = Field(min_length=1)
    stage: Literal["hook", "escalation", "extraction", "handoff", "none"]
    tactics: tuple[TacticPhrase, ...] = ()

    @field_validator("text")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("utterance text must not be blank")
        return value


class GeneratedRow(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    cell_id: str
    language: Literal["kk", "ru", "mixed", "en"]
    kind: Literal["scam", "legit"]
    scheme: str | None = None
    negative_class: str | None = None
    role: str | None = None
    utterances: tuple[GenUtterance, ...] = Field(min_length=1)


class RowCheck(BaseModel):
    """Outcome of validating one row: the corpus dialogue (if usable), errors, warnings."""

    model_config = ConfigDict(frozen=True)

    row_id: str
    dialogue: Dialogue | None = None
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()


def _phrase_errors(row: GeneratedRow, v2: frozenset[str]) -> list[str]:
    errors: list[str] = []
    for index, utterance in enumerate(row.utterances):
        for tactic in utterance.tactics:
            if tactic.id not in v2:
                errors.append(f"utterance {index}: unknown tactic {tactic.id!r}")
            if tactic.phrase not in utterance.text:
                errors.append(f"utterance {index}: phrase for {tactic.id} is not verbatim")
    return errors


def _cell_errors(row: GeneratedRow, cell: GridCell) -> list[str]:
    errors: list[str] = []
    expected = {
        "id": cell.cell_id, "cell_id": cell.cell_id, "language": cell.language, "kind": cell.kind,
        "scheme": cell.scheme, "negative_class": cell.negative_class, "role": cell.role,
    }
    for key, want in expected.items():
        if getattr(row, key) != want:
            errors.append(f"{key} is {getattr(row, key)!r}, the cell says {want!r}")
    tags = {t.id for u in row.utterances for t in u.tactics}
    if cell.kind == "legit" and tags:
        errors.append(f"a negative carries tactics {sorted(tags)}")
    if cell.kind == "scam":
        missing = set(cell.tactics) - tags
        if missing:
            errors.append(f"required tactics not shown: {sorted(missing)}")
    return errors


def _warnings(row: GeneratedRow, cell: GridCell) -> list[str]:
    warnings: list[str] = []
    n = len(row.utterances)
    tolerance = max(_LENGTH_TOLERANCE_ABS, round(_LENGTH_TOLERANCE_REL * cell.length_target))
    if abs(n - cell.length_target) > tolerance:
        warnings.append(f"length {n} vs target {cell.length_target}")
    text = " ".join(u.text for u in row.utterances)
    kk, cyr, lat = len(_KK_LETTERS.findall(text)), len(_CYRILLIC.findall(text)), len(_LATIN.findall(text))
    if cell.language == "en" and lat < cyr:
        warnings.append("register en but mostly Cyrillic")
    if cell.language in ("kk", "mixed") and kk == 0:
        warnings.append(f"register {cell.language} but no Kazakh-specific letters")
    if cell.language == "ru" and cyr and kk / cyr > 0.02:
        warnings.append("register ru but Kazakh letters present")
    if {u.speaker for u in row.utterances} != set(SPEAKERS):
        warnings.append("only one speaker")
    return warnings


def to_dialogue(row: GeneratedRow, *, hard_negative_ids: frozenset[str]) -> Dialogue:
    """Corpus `Dialogue`: tags = tactics shown (weight 1.0, first-seen order), trigger spans =
    every phrase grounded at its own utterance's offset in the joined transcript."""
    utterances = tuple(Utterance(speaker=u.speaker, text=u.text) for u in row.utterances)
    spans: list[Span] = []
    offset = 0
    for utterance in row.utterances:
        for tactic in utterance.tactics:
            start = utterance.text.find(tactic.phrase)
            span = Span(text=tactic.phrase, start=offset + start, end=offset + start + len(tactic.phrase))
            if span not in spans:
                spans.append(span)
        offset += len(utterance.text) + 1  # UTTERANCE_JOIN
    tag_ids = tuple(dict.fromkeys(t.id for u in row.utterances for t in u.tactics))
    is_scam = row.kind == "scam"
    label = Label(
        risk=SCAM_RISK if is_scam else HARD_NEGATIVE_RISK,
        tactic_tags=tuple(TacticTag(id=t, weight=1.0) for t in tag_ids),
        trigger_spans=tuple(sorted(spans, key=lambda s: s.start)),
        is_hard_negative=(not is_scam) and row.negative_class in hard_negative_ids,
    )
    return Dialogue(id=row.id, language=row.language, utterances=utterances, label=label)


def check_row(raw: Mapping, cells: Mapping[str, GridCell], *, v2: frozenset[str], hard_negative_ids: frozenset[str]) -> RowCheck:
    """Validate one raw JSON row against the schema and its cell; never raises."""
    row_id = str(raw.get("id", "?")) if isinstance(raw, Mapping) else "?"
    try:
        row = GeneratedRow.model_validate(raw)
    except ValidationError as exc:
        return RowCheck(row_id=row_id, errors=tuple(f"schema: {e['loc']}: {e['msg']}" for e in exc.errors()))
    cell = cells.get(row.cell_id)
    if cell is None:
        return RowCheck(row_id=row_id, errors=(f"unknown cell {row.cell_id!r}",))
    errors = _phrase_errors(row, v2) + _cell_errors(row, cell)
    if errors:
        return RowCheck(row_id=row_id, errors=tuple(errors))
    return RowCheck(
        row_id=row_id,
        dialogue=to_dialogue(row, hard_negative_ids=hard_negative_ids),
        warnings=tuple(_warnings(row, cell)),
    )


def row_stage_counts(raw_rows) -> dict[str, int]:
    out: dict[str, int] = {}
    for raw in raw_rows:
        for u in raw.get("utterances", []):
            out[u.get("stage", "?")] = out.get(u.get("stage", "?"), 0) + 1
    return dict(sorted(out.items()))
