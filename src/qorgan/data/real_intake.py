"""Real-call intake: a partner batch → scrubbed `Dialogue`s + hashed number linkage
(PLAN_2026-09 A8, protocol in `docs/DATA_INTAKE.md`).

A batch is a directory the partner delivers (encrypted transfer or on-prem):

    batch.yaml          provenance + consent record + labeler declaration
    calls.csv           one row per call: id, language, label, tactic ids, caller number,
                        transcript file OR audio file, consent reference
    transcripts/*.txt   "speaker: text" per line (bare lines allowed)
    audio/*             optional; transcribed only with an injected transcriber

Everything that leaves this module is minimised: utterances pass through `scrub_text`,
caller numbers become HMAC digests in a *separate* linkage file, and nothing here prints or
logs call content — the locked set is scored, never read.
"""

from __future__ import annotations

import csv
import hashlib
from collections.abc import Callable
from datetime import date
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from qorgan.data.schema import Dialogue, Label, SupportedLanguage, TacticTag, Utterance
from qorgan.data.scrub import scrub_text
from qorgan.partners import PARTNER_ID_PATTERN
from qorgan.privacy.numbers import MissingHmacKeyError, display_prefix, hash_phone_number
from qorgan.reports.model import CONSENT_BASIS_PATTERN
from qorgan.taxonomy import get_taxonomy

BATCH_FILENAME = "batch.yaml"
CALLS_FILENAME = "calls.csv"
BATCHES_SUBDIR = "batches"
CALLS_COLUMNS = (
    "call_id", "language", "label", "tactic_ids", "caller_number", "transcript_file", "audio_file", "consent_ref",
)
BATCH_ID_PATTERN = r"^[a-z0-9][a-z0-9_-]{2,39}$"
CALL_ID_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9_-]{0,39}$"
_TACTIC_SEPARATOR = ";"
_SPEAKER_SEPARATOR = ":"
_BARE_LINE_SPEAKER = "speaker"
_SCAM_RISK, _LEGIT_RISK = 1.0, 0.0

CallLabel = Literal["scam", "legit"]
Transcriber = Callable[[Path], str]


class BatchValidationError(ValueError):
    """The batch is malformed or violates the protocol; nothing is written."""


class BatchManifest(BaseModel):
    """`batch.yaml`: who delivered what, under which basis, labelled by whom."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    batch_id: str = Field(pattern=BATCH_ID_PATTERN)
    partner_id: str = Field(pattern=PARTNER_ID_PATTERN)
    delivered_on: date
    transfer_method: str = Field(min_length=3, max_length=64)
    consent_basis: str = Field(pattern=CONSENT_BASIS_PATTERN)
    legal_reference: str | None = Field(default=None, max_length=128)
    labeler_id: str = Field(min_length=1, max_length=64)
    # The person labelling must never have edited the lexicons/anchors (PLAN A3).
    labeler_edited_lexicons: Literal[False]
    notes: str | None = Field(default=None, max_length=1000)


class CallRow(BaseModel):
    """One `calls.csv` row, validated against the taxonomy and the batch directory."""

    model_config = ConfigDict(frozen=True)

    call_id: str = Field(pattern=CALL_ID_PATTERN)
    language: SupportedLanguage
    label: CallLabel
    tactic_ids: tuple[str, ...] = ()
    caller_number: str | None = None
    transcript_file: str | None = None
    audio_file: str | None = None
    consent_ref: str | None = None

    @field_validator("tactic_ids")
    @classmethod
    def _known_tactics(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        unknown = sorted(set(value) - set(get_taxonomy().tactic_ids()))
        if unknown:
            raise ValueError(f"unknown tactic ids: {unknown}")
        return value

    @model_validator(mode="after")
    def _transcript_or_audio(self) -> "CallRow":
        if not self.transcript_file and not self.audio_file:
            raise ValueError("a call needs a transcript_file or an audio_file")
        return self


class Linkage(BaseModel):
    """Caller-number digest for one dialogue, kept apart from the text."""

    model_config = ConfigDict(frozen=True)

    dialogue_id: str
    number_hash: str
    number_prefix: str | None


class IngestedBatch(BaseModel):
    model_config = ConfigDict(frozen=True)

    manifest: BatchManifest
    dialogues: tuple[Dialogue, ...]
    linkages: tuple[Linkage, ...]
    input_sha256: dict[str, str]
    counts: dict


# --- parsing ---------------------------------------------------------------------------------


def load_batch_manifest(batch_dir: Path) -> BatchManifest:
    path = batch_dir / BATCH_FILENAME
    if not path.exists():
        raise BatchValidationError(f"missing {BATCH_FILENAME} in {batch_dir}")
    try:
        return BatchManifest.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")) or {})
    except (ValidationError, yaml.YAMLError) as exc:
        raise BatchValidationError(f"{BATCH_FILENAME}: {exc}") from exc


def load_call_rows(batch_dir: Path) -> tuple[CallRow, ...]:
    path = batch_dir / CALLS_FILENAME
    if not path.exists():
        raise BatchValidationError(f"missing {CALLS_FILENAME} in {batch_dir}")
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != CALLS_COLUMNS:
            raise BatchValidationError(f"{CALLS_FILENAME} columns must be exactly {CALLS_COLUMNS}")
        rows = tuple(_parse_row(raw, batch_dir) for raw in reader)
    ids = [row.call_id for row in rows]
    if len(set(ids)) != len(ids):
        raise BatchValidationError(f"{CALLS_FILENAME}: duplicate call ids")
    if not rows:
        raise BatchValidationError(f"{CALLS_FILENAME}: no calls")
    return rows


def _parse_row(raw: dict[str, str], batch_dir: Path) -> CallRow:
    cleaned = {key: (value or "").strip() for key, value in raw.items()}
    try:
        row = CallRow(
            call_id=cleaned["call_id"],
            language=cleaned["language"],
            label=cleaned["label"],
            tactic_ids=tuple(t.strip() for t in cleaned["tactic_ids"].split(_TACTIC_SEPARATOR) if t.strip()),
            caller_number=cleaned["caller_number"] or None,
            transcript_file=cleaned["transcript_file"] or None,
            audio_file=cleaned["audio_file"] or None,
            consent_ref=cleaned["consent_ref"] or None,
        )
    except ValidationError as exc:
        raise BatchValidationError(f"call {cleaned.get('call_id', '?')!r}: {exc}") from exc
    for relative in (row.transcript_file, row.audio_file):
        if relative and not (batch_dir / relative).is_file():
            raise BatchValidationError(f"call {row.call_id!r}: file not found: {relative}")
    return row


def parse_transcript(text: str) -> tuple[Utterance, ...]:
    """`speaker: text` per line; a line without a speaker prefix becomes `speaker`."""
    utterances: list[Utterance] = []
    for line in text.splitlines():
        if not line.strip():
            continue
        speaker, sep, rest = line.partition(_SPEAKER_SEPARATOR)
        if sep and speaker.strip() and " " not in speaker.strip():
            utterances.append(Utterance(speaker=speaker.strip(), text=rest.strip() or line.strip()))
        else:
            utterances.append(Utterance(speaker=_BARE_LINE_SPEAKER, text=line.strip()))
    if not utterances:
        raise BatchValidationError("transcript is empty")
    return tuple(utterances)


# --- conversion ------------------------------------------------------------------------------


def dialogue_id(batch_id: str, call_id: str) -> str:
    return f"real-{batch_id}-{call_id}"


def call_to_dialogue(row: CallRow, utterances: tuple[Utterance, ...], *, batch_id: str) -> Dialogue:
    """Scrub every utterance and attach the partner's label. No trigger spans: real calls
    carry human labels, not attributed phrases, and nothing is fabricated to fill them."""
    scrubbed = tuple(Utterance(speaker=u.speaker, text=scrub_text(u.text)) for u in utterances)
    is_scam = row.label == "scam"
    label = Label(
        risk=_SCAM_RISK if is_scam else _LEGIT_RISK,
        tactic_tags=tuple(TacticTag(id=t) for t in row.tactic_ids) if is_scam else (),
        is_hard_negative=not is_scam,
    )
    return Dialogue(id=dialogue_id(batch_id, row.call_id), language=row.language, utterances=scrubbed, label=label)


def ingest_batch(batch_dir: Path, *, hmac_key: bytes | None, transcriber: Transcriber | None = None) -> IngestedBatch:
    """Validate and convert a whole batch in memory (nothing is written)."""
    manifest = load_batch_manifest(batch_dir)
    rows = load_call_rows(batch_dir)
    dialogues: list[Dialogue] = []
    linkages: list[Linkage] = []
    hashes = {BATCH_FILENAME: _sha256(batch_dir / BATCH_FILENAME), CALLS_FILENAME: _sha256(batch_dir / CALLS_FILENAME)}
    for row in rows:
        utterances = _utterances_for(row, batch_dir, transcriber, hashes)
        dialogue = call_to_dialogue(row, utterances, batch_id=manifest.batch_id)
        dialogues.append(dialogue)
        if row.caller_number:
            linkages.append(_linkage(dialogue.id, row, hmac_key))
    return IngestedBatch(
        manifest=manifest,
        dialogues=tuple(dialogues),
        linkages=tuple(linkages),
        input_sha256=hashes,
        counts=_counts(dialogues),
    )


def _utterances_for(row: CallRow, batch_dir: Path, transcriber: Transcriber | None, hashes: dict[str, str]) -> tuple[Utterance, ...]:
    if row.transcript_file:
        path = batch_dir / row.transcript_file
        hashes[row.transcript_file] = _sha256(path)
        try:
            return parse_transcript(path.read_text(encoding="utf-8"))
        except BatchValidationError as exc:
            raise BatchValidationError(f"call {row.call_id!r}: {exc}") from exc
    if transcriber is None:
        raise BatchValidationError(f"call {row.call_id!r} has audio only; run with a transcriber (--transcribe)")
    path = batch_dir / str(row.audio_file)
    hashes[str(row.audio_file)] = _sha256(path)
    return parse_transcript(transcriber(path))


def _linkage(dialogue_id_: str, row: CallRow, hmac_key: bytes | None) -> Linkage:
    number = str(row.caller_number)
    try:
        digest = hash_phone_number(number, key=hmac_key)
    except MissingHmacKeyError as exc:
        raise BatchValidationError(
            f"call {row.call_id!r} carries a caller number but no QORGAN_NUMBER_HMAC_KEY is set; refusing to ingest"
        ) from exc
    except ValueError as exc:
        raise BatchValidationError(f"call {row.call_id!r}: caller number not understood") from exc
    return Linkage(dialogue_id=dialogue_id_, number_hash=digest, number_prefix=display_prefix(number))


def _counts(dialogues: list[Dialogue]) -> dict:
    by_language: dict[str, int] = {}
    for d in dialogues:
        by_language[d.language] = by_language.get(d.language, 0) + 1
    positives = sum(1 for d in dialogues if not d.label.is_hard_negative)
    return {"positives": positives, "negatives": len(dialogues) - positives, "by_language": by_language}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# --- persistence -----------------------------------------------------------------------------


def write_batch(ingested: IngestedBatch, real_dir: Path) -> tuple[Path, ...]:
    """Persist one batch under `<real_dir>/batches/`: dialogues, linkage, provenance manifest."""
    batches = real_dir / BATCHES_SUBDIR
    batches.mkdir(parents=True, exist_ok=True)
    stem = ingested.manifest.batch_id
    dialogues_path = batches / f"{stem}.jsonl"
    linkage_path = batches / f"{stem}.linkage.jsonl"
    manifest_path = batches / f"{stem}.manifest.json"
    dialogues_path.write_text("".join(d.model_dump_json() + "\n" for d in ingested.dialogues), encoding="utf-8")
    linkage_path.write_text("".join(l.model_dump_json() + "\n" for l in ingested.linkages), encoding="utf-8")
    provenance = {
        "batch": ingested.manifest.model_dump(mode="json"),
        "counts": ingested.counts,
        "input_sha256": ingested.input_sha256,
        "dialogue_ids": [d.id for d in ingested.dialogues],
    }
    manifest_path.write_text(_json_dumps(provenance), encoding="utf-8")
    return dialogues_path, linkage_path, manifest_path


def _json_dumps(payload: dict) -> str:
    import json

    return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
