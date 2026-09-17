"""TDD tests for the real-call intake protocol (PLAN_2026-09 A8): a partner batch
(`batch.yaml` + `calls.csv` + transcripts) round-trips into scrubbed `Dialogue`s, hashed
number linkage, the first-come locked/real_train allocation, materialised splits and a
hash-locked manifest. The script never prints call content; neither do these tests."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest
import yaml

from qorgan.data.build_corpus import normalize_for_dedup
from qorgan.data.real_allocation import (
    LOCKED_SPLIT,
    TRAIN_SPLIT,
    Allocation,
    LockTargets,
    LockViolation,
    allocate,
    load_allocation,
    materialize,
    save_allocation,
    verify_lock,
)
from qorgan.data.real_intake import (
    BatchValidationError,
    call_to_dialogue,
    ingest_batch,
    load_batch_manifest,
    load_call_rows,
    parse_transcript,
    write_batch,
)
from support.numbers import TEST_HMAC_KEY, hashed, prefix

TRANSCRIPTS = {
    "c1": "caller: Здравствуйте, это служба безопасности банка.\ncaller: Продиктуйте код из SMS, срочно.\nvictim: Какой код?",
    "c2": "caller: Добрый день, ваша карта готова, забрать можно в отделении.\nvictim: Спасибо, код называть не нужно?\ncaller: Нет, никаких кодов.",
    "c3": "caller: Сәлеметсіз бе, бұл банк. Сіздің ЖСН 990101300123 бойынша қарыз бар, +7 701 222 33 44 нөміріне хабарласыңыз.\nvictim: Түсінбедім.",
    "c4": "Это оператор связи, ваш тариф меняется с первого числа.\nХорошо, спасибо.",
}


def _write_batch(root: Path, *, rows=None, manifest_overrides=None) -> Path:
    batch = root / "batch-2026-09-a"
    (batch / "transcripts").mkdir(parents=True)
    manifest = {
        "batch_id": "bank-a-2026-09-a",
        "partner_id": "bank_a",
        "delivered_on": "2026-09-16",
        "transfer_method": "sftp_encrypted",
        "consent_basis": "customer_consent",
        "legal_reference": "DPA-2026-014",
        "labeler_id": "fraud-desk-analyst-2",
        "labeler_edited_lexicons": False,
        "notes": "first delivery",
    }
    manifest.update(manifest_overrides or {})
    (batch / "batch.yaml").write_text(yaml.safe_dump(manifest, allow_unicode=True), encoding="utf-8")
    for cid, text in TRANSCRIPTS.items():
        (batch / "transcripts" / f"{cid}.txt").write_text(text, encoding="utf-8")
    default_rows = [
        {"call_id": "c1", "language": "ru", "label": "scam", "tactic_ids": "impersonation_bank;otp_request", "caller_number": "+7 700 555 66 77", "transcript_file": "transcripts/c1.txt", "audio_file": "", "consent_ref": "CR-1"},
        {"call_id": "c2", "language": "ru", "label": "legit", "tactic_ids": "", "caller_number": "", "transcript_file": "transcripts/c2.txt", "audio_file": "", "consent_ref": "CR-2"},
        {"call_id": "c3", "language": "mixed", "label": "scam", "tactic_ids": "fear_threat", "caller_number": "+7 701 000 11 22", "transcript_file": "transcripts/c3.txt", "audio_file": "", "consent_ref": "CR-3"},
        {"call_id": "c4", "language": "ru", "label": "legit", "tactic_ids": "", "caller_number": "", "transcript_file": "transcripts/c4.txt", "audio_file": "", "consent_ref": "CR-4"},
    ]
    with (batch / "calls.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(default_rows[0]))
        writer.writeheader()
        writer.writerows(rows if rows is not None else default_rows)
    return batch


# --- parsing ---------------------------------------------------------------------------------


def test_manifest_and_rows_parse(tmp_path):
    batch = _write_batch(tmp_path)
    manifest = load_batch_manifest(batch)
    rows = load_call_rows(batch)
    assert manifest.batch_id == "bank-a-2026-09-a" and manifest.partner_id == "bank_a"
    assert [r.call_id for r in rows] == ["c1", "c2", "c3", "c4"]
    assert rows[0].tactic_ids == ("impersonation_bank", "otp_request") and rows[1].tactic_ids == ()


def test_labeler_must_declare_no_lexicon_authorship(tmp_path):
    batch = _write_batch(tmp_path, manifest_overrides={"labeler_edited_lexicons": True})
    with pytest.raises(BatchValidationError):
        load_batch_manifest(batch)


@pytest.mark.parametrize(
    "bad",
    [
        {"label": "maybe"},
        {"language": "en"},
        {"tactic_ids": "not_a_tactic"},
        {"transcript_file": "transcripts/missing.txt"},
        {"transcript_file": "", "audio_file": ""},
    ],
)
def test_bad_rows_fail_validation(tmp_path, bad):
    row = {"call_id": "c1", "language": "ru", "label": "scam", "tactic_ids": "", "caller_number": "", "transcript_file": "transcripts/c1.txt", "audio_file": "", "consent_ref": "CR-1"}
    batch = _write_batch(tmp_path, rows=[{**row, **bad}])
    with pytest.raises(BatchValidationError):
        load_call_rows(batch)


def test_duplicate_call_ids_are_rejected(tmp_path):
    row = {"call_id": "c1", "language": "ru", "label": "scam", "tactic_ids": "", "caller_number": "", "transcript_file": "transcripts/c1.txt", "audio_file": "", "consent_ref": ""}
    batch = _write_batch(tmp_path, rows=[row, row])
    with pytest.raises(BatchValidationError):
        load_call_rows(batch)


def test_parse_transcript_keeps_speakers_and_tolerates_bare_lines():
    utterances = parse_transcript(TRANSCRIPTS["c4"])
    assert [u.speaker for u in utterances] == ["speaker", "speaker"]
    utterances = parse_transcript(TRANSCRIPTS["c1"])
    assert [u.speaker for u in utterances] == ["caller", "caller", "victim"]
    assert utterances[0].text == "Здравствуйте, это служба безопасности банка."
    with pytest.raises(BatchValidationError):
        parse_transcript("   \n  ")


# --- conversion ------------------------------------------------------------------------------


def test_call_to_dialogue_scrubs_labels_and_grounds_nothing(tmp_path):
    batch = _write_batch(tmp_path)
    rows = {r.call_id: r for r in load_call_rows(batch)}
    scam = call_to_dialogue(rows["c3"], parse_transcript(TRANSCRIPTS["c3"]), batch_id="bank-a-2026-09-a")
    legit = call_to_dialogue(rows["c2"], parse_transcript(TRANSCRIPTS["c2"]), batch_id="bank-a-2026-09-a")

    assert scam.id == "real-bank-a-2026-09-a-c3" and scam.language == "mixed"
    assert scam.label.risk == 1.0 and not scam.label.is_hard_negative
    assert [t.id for t in scam.label.tactic_tags] == ["fear_threat"] and scam.label.trigger_spans == ()
    text = scam.transcript()
    assert "990101300123" not in text and "222 33 44" not in text and "[IIN]" in text and "[PHONE]" in text
    assert legit.label.risk == 0.0 and legit.label.is_hard_negative and legit.label.tactic_tags == ()


def test_ingest_batch_round_trip_hashes_numbers_and_never_stores_raw(tmp_path):
    batch = _write_batch(tmp_path)

    ingested = ingest_batch(batch, hmac_key=TEST_HMAC_KEY)

    assert [d.id for d in ingested.dialogues] == [f"real-bank-a-2026-09-a-{c}" for c in ("c1", "c2", "c3", "c4")]
    assert {l.dialogue_id: (l.number_hash, l.number_prefix) for l in ingested.linkages} == {
        "real-bank-a-2026-09-a-c1": (hashed("+7 700 555 66 77"), prefix("+7 700 555 66 77")),
        "real-bank-a-2026-09-a-c3": (hashed("+7 701 000 11 22"), prefix("+7 701 000 11 22")),
    }
    assert set(ingested.input_sha256) == {"batch.yaml", "calls.csv", *(f"transcripts/{c}.txt" for c in TRANSCRIPTS)}
    assert ingested.counts == {"positives": 2, "negatives": 2, "by_language": {"ru": 3, "mixed": 1}}

    real_dir = tmp_path / "real"
    written = write_batch(ingested, real_dir)
    blob = "".join(p.read_text(encoding="utf-8") for p in written)
    assert "555 66 77" not in blob and "5556677" not in blob and "000 11 22" not in blob and "990101300123" not in blob
    assert (real_dir / "batches" / "bank-a-2026-09-a.jsonl").exists()
    assert (real_dir / "batches" / "bank-a-2026-09-a.linkage.jsonl").exists()
    assert (real_dir / "batches" / "bank-a-2026-09-a.manifest.json").exists()


def test_numbers_without_a_key_are_refused(tmp_path):
    batch = _write_batch(tmp_path)
    with pytest.raises(BatchValidationError):
        ingest_batch(batch, hmac_key=None)


def test_audio_rows_use_the_injected_transcriber(tmp_path):
    row = {"call_id": "a1", "language": "kk", "label": "legit", "tactic_ids": "", "caller_number": "", "transcript_file": "", "audio_file": "audio/a1.wav", "consent_ref": ""}
    batch = _write_batch(tmp_path, rows=[row])
    (batch / "audio").mkdir()
    (batch / "audio" / "a1.wav").write_bytes(b"RIFF....WAVE")
    calls: list[Path] = []

    def fake_transcriber(path: Path) -> str:
        calls.append(path)
        return "Сәлеметсіз бе, бұл байланыс операторы."

    ingested = ingest_batch(batch, hmac_key=TEST_HMAC_KEY, transcriber=fake_transcriber)

    assert calls == [batch / "audio" / "a1.wav"]
    assert ingested.dialogues[0].transcript() == "Сәлеметсіз бе, бұл байланыс операторы."
    with pytest.raises(BatchValidationError):
        ingest_batch(batch, hmac_key=TEST_HMAC_KEY)  # audio without a transcriber


# --- allocation + lock -----------------------------------------------------------------------


def test_allocation_is_first_come_until_the_lock_targets_are_met(tmp_path):
    batch = _write_batch(tmp_path)
    dialogues = ingest_batch(batch, hmac_key=TEST_HMAC_KEY).dialogues  # scam, legit, scam, legit
    empty = Allocation(targets=LockTargets(negatives=1, positives=1))

    allocation = allocate(empty, dialogues)

    assert allocation.locked_positives == ("real-bank-a-2026-09-a-c1",)
    assert allocation.locked_negatives == ("real-bank-a-2026-09-a-c2",)
    assert allocation.real_train == ("real-bank-a-2026-09-a-c3", "real-bank-a-2026-09-a-c4")
    assert allocation.is_locked
    assert empty.real_train == ()  # immutable input
    assert allocate(allocation, dialogues) == allocation  # re-ingest is a no-op


def test_allocation_persists(tmp_path):
    allocation = Allocation(targets=LockTargets(negatives=2, positives=2), locked_negatives=("x",))
    path = tmp_path / "real" / "allocation.json"
    save_allocation(allocation, path)
    assert load_allocation(path) == allocation
    assert load_allocation(tmp_path / "nope.json").targets == LockTargets()


def test_materialize_writes_splits_and_a_hash_locked_manifest(tmp_path):
    batch = _write_batch(tmp_path)
    real_dir, processed = tmp_path / "real", tmp_path / "processed"
    ingested = ingest_batch(batch, hmac_key=TEST_HMAC_KEY)
    write_batch(ingested, real_dir)
    save_allocation(allocate(Allocation(targets=LockTargets(negatives=1, positives=1)), ingested.dialogues), real_dir / "allocation.json")

    manifest = materialize(real_dir, processed)

    locked = (processed / f"{LOCKED_SPLIT}.jsonl").read_text(encoding="utf-8").splitlines()
    train = (processed / f"{TRAIN_SPLIT}.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(locked) == 2 and len(train) == 2
    assert manifest.locked and manifest.counts[LOCKED_SPLIT] == {"positives": 1, "negatives": 1}
    assert manifest.counts[TRAIN_SPLIT] == {"positives": 1, "negatives": 1}
    assert manifest.batches == ("bank-a-2026-09-a",)
    assert len(manifest.locked_sha256) == 64
    verify_lock(real_dir / "manifest.json", processed)  # does not raise
    assert materialize(real_dir, processed) == manifest  # idempotent


def test_a_locked_set_cannot_change(tmp_path):
    batch = _write_batch(tmp_path)
    real_dir, processed = tmp_path / "real", tmp_path / "processed"
    ingested = ingest_batch(batch, hmac_key=TEST_HMAC_KEY)
    write_batch(ingested, real_dir)
    allocation = allocate(Allocation(targets=LockTargets(negatives=1, positives=1)), ingested.dialogues)
    save_allocation(allocation, real_dir / "allocation.json")
    materialize(real_dir, processed)

    # Someone edits the allocation so a different call sits in the locked set.
    tampered = allocation.model_copy(update={"locked_positives": ("real-bank-a-2026-09-a-c3",), "real_train": ("real-bank-a-2026-09-a-c1", "real-bank-a-2026-09-a-c4")})
    save_allocation(tampered, real_dir / "allocation.json")
    with pytest.raises(LockViolation):
        materialize(real_dir, processed)

    # ...or the materialised file itself is edited.
    save_allocation(allocation, real_dir / "allocation.json")
    materialize(real_dir, processed)
    path = processed / f"{LOCKED_SPLIT}.jsonl"
    path.write_text(path.read_text(encoding="utf-8").replace("Продиктуйте", "Назовите"), encoding="utf-8")
    with pytest.raises(LockViolation):
        verify_lock(real_dir / "manifest.json", processed)


def test_no_overlap_with_the_synthetic_corpus_is_checkable(tmp_path):
    batch = _write_batch(tmp_path)
    ingested = ingest_batch(batch, hmac_key=TEST_HMAC_KEY)
    normalized = {normalize_for_dedup(d.transcript()) for d in ingested.dialogues}
    assert len(normalized) == 4  # distinct, and comparable against train/val/augment by the lock test
