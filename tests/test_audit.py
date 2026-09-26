"""TDD tests for `qorgan.audit` -- the append-only, content-free audit log shared by the
partner API (PLAN_2026-09 C5) and the analyst drill-down (C4)."""

from __future__ import annotations

import json
import threading
import time
from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from qorgan.audit import (
    AuditEntry,
    AuditIntegrityError,
    MissingAuditKeyError,
    append_audit,
    audit_head,
    load_audit,
    main,
    seal_legacy,
    verify_audit,
)

NOW = datetime(2026, 9, 17, 12, 0, tzinfo=UTC)
KEY = b"tests-only-audit-chain-key-0123456789abcdef"


def _entry(**overrides) -> AuditEntry:
    base = dict(
        timestamp=NOW,
        actor_kind="partner",
        actor_id="bank_a",
        action="report.submit",
        subject="receipt:0123456789abcdef01234567",
        outcome="stored",
    )
    return AuditEntry(**{**base, **overrides})


def test_append_then_load_round_trips_in_order(tmp_path):
    path = tmp_path / "audit_log.jsonl"
    first = _entry()
    second = _entry(action="report.delete", outcome="deleted")

    append_audit(first, path, key=KEY)
    append_audit(second, path, key=KEY)

    assert load_audit(path) == [first, second]


def test_missing_log_means_no_entries(tmp_path):
    assert load_audit(tmp_path / "nope.jsonl") == []


def test_entries_are_content_free_by_construction():
    """Audit lines describe *that* something happened, never call content or numbers."""
    with pytest.raises(ValidationError):
        _entry(subject="caller +7 700 555 66 77")
    with pytest.raises(ValidationError):
        _entry(outcome="stored transcript: продиктуйте код из смс на +77001112233")
    with pytest.raises(ValidationError):
        _entry(actor_id="")


def test_entries_are_frozen():
    entry = _entry()
    with pytest.raises(ValidationError):
        entry.outcome = "changed"  # type: ignore[misc]


# --- system-generated ids are not call content (found 2026-09-24) ---------------------------

def test_a_receipt_subject_whose_hex_looks_like_a_number_is_still_accepted():
    """`subject="receipt:<32 hex>"` is a system-generated opaque id. ~0.46 % of random
    receipts contain a digit run the PII scrubber reads as a phone number, which used to make
    the audit write fail — losing the audit record for roughly 1 partner submission in 217."""
    from qorgan.audit import AuditEntry

    entry = AuditEntry(
        timestamp=datetime(2026, 9, 24, tzinfo=UTC),
        actor_kind="partner",
        actor_id="partner:acme",
        action="report.submit",
        subject="receipt:743c0ef8f88913843287eaa388fd69bc",
        outcome="stored",
    )
    assert entry.subject == "receipt:743c0ef8f88913843287eaa388fd69bc"


def test_a_subject_carrying_real_call_content_is_still_refused():
    from qorgan.audit import AuditEntry

    with pytest.raises(ValidationError):
        AuditEntry(
            timestamp=datetime(2026, 9, 24, tzinfo=UTC),
            actor_kind="partner",
            actor_id="partner:acme",
            action="report.submit",
            subject="caller +7 701 234 56 78 said to transfer",
            outcome="stored",
        )


# --- purpose limitation (PLAN C4): why a full transcript was opened is a closed code -------


def test_purpose_is_a_closed_enum_and_optional():
    from qorgan.audit import ACCESS_PURPOSES

    assert ACCESS_PURPOSES == ("pattern_review", "citizen_request", "partner_request")
    assert _entry(purpose="citizen_request").purpose == "citizen_request"
    assert _entry().purpose is None
    with pytest.raises(ValidationError):
        _entry(purpose="curiosity")


# --- tamper evidence: a keyed hash chain ----------------------------------------------------


def _chain(path, n: int) -> list[AuditEntry]:
    entries = [_entry(action="report.submit", outcome=f"stored-{i}") for i in range(n)]
    for entry in entries:
        append_audit(entry, path, key=KEY)
    return entries


def _lines(path) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines(keepends=True)


def _write_lines(path, lines) -> None:
    path.write_text("".join(lines), encoding="utf-8")


def test_each_line_carries_its_sequence_the_previous_mac_and_its_own_mac(tmp_path):
    path = tmp_path / "audit_log.jsonl"
    entries = _chain(path, 3)

    records = [json.loads(line) for line in _lines(path)]
    assert [r["seq"] for r in records] == [0, 1, 2]
    assert records[1]["prev"] == records[0]["mac"] and records[2]["prev"] == records[1]["mac"]
    assert all(len(r["mac"]) == 64 and int(r["mac"], 16) >= 0 for r in records)
    assert load_audit(path) == entries  # chain fields do not leak into the entries

    report = verify_audit(path, key=KEY)
    assert report.ok and report.first_broken is None
    assert (report.entries, report.legacy, report.chained) == (3, 0, 3)
    assert (report.head_seq, report.head_mac) == (2, records[2]["mac"])
    assert audit_head(path) == (2, records[2]["mac"])


def test_no_key_means_no_audit_line_and_no_file(tmp_path):
    path = tmp_path / "audit_log.jsonl"
    with pytest.raises(MissingAuditKeyError):
        append_audit(_entry(), path, key=None)
    with pytest.raises(MissingAuditKeyError):
        append_audit(_entry(), path, key=b"")
    assert not path.exists()


def test_an_empty_or_missing_log_verifies_as_empty(tmp_path):
    report = verify_audit(tmp_path / "nope.jsonl", key=KEY)
    assert report.ok and report.entries == 0 and report.head_seq is None


def test_an_edited_field_breaks_the_chain_at_that_entry(tmp_path):
    path = tmp_path / "audit_log.jsonl"
    _chain(path, 4)
    lines = _lines(path)
    record = json.loads(lines[1])
    record["actor_id"] = "someone_else"
    lines[1] = json.dumps(record, ensure_ascii=False) + "\n"
    _write_lines(path, lines)

    report = verify_audit(path, key=KEY)
    assert not report.ok and report.first_broken == 1
    assert "mac" in report.reason.lower()


def test_a_deleted_middle_entry_is_reported_where_the_gap_starts(tmp_path):
    path = tmp_path / "audit_log.jsonl"
    _chain(path, 4)
    lines = _lines(path)
    _write_lines(path, lines[:1] + lines[2:])

    report = verify_audit(path, key=KEY)
    assert not report.ok and report.first_broken == 1
    assert "sequence" in report.reason


def test_reordered_entries_are_detected(tmp_path):
    path = tmp_path / "audit_log.jsonl"
    _chain(path, 4)
    lines = _lines(path)
    _write_lines(path, [lines[0], lines[2], lines[1], lines[3]])

    report = verify_audit(path, key=KEY)
    assert not report.ok and report.first_broken == 1


def test_a_chain_recomputed_without_the_server_key_is_detected(tmp_path):
    """Someone with file access but not the key can rewrite an entry *and* recompute every
    later link -- with a key of their own. The verifier holds the real key."""
    path = tmp_path / "audit_log.jsonl"
    _chain(path, 2)
    forged = tmp_path / "forged.jsonl"
    for entry in (_entry(outcome="stored-0"), _entry(outcome="nothing to see")):
        append_audit(entry, forged, key=b"attacker-guessed-key-000000000000")
    _write_lines(path, _lines(forged))

    report = verify_audit(path, key=KEY)
    assert not report.ok and report.first_broken == 0


def test_verifying_with_the_wrong_key_fails(tmp_path):
    path = tmp_path / "audit_log.jsonl"
    _chain(path, 2)
    assert verify_audit(path, key=b"another-key-000000000000000000000").first_broken == 0


def test_tail_truncation_needs_an_external_anchor(tmp_path):
    """Honest limit: dropping the newest entries leaves a valid chain. An anchor (seq, mac)
    recorded elsewhere -- `python -m qorgan.audit head` -- makes it detectable."""
    path = tmp_path / "audit_log.jsonl"
    _chain(path, 5)
    anchor = audit_head(path)
    _write_lines(path, _lines(path)[:3])

    assert verify_audit(path, key=KEY).ok  # undetectable on its own -- stated, not hidden
    truncated = verify_audit(path, key=KEY, anchor=anchor)
    assert not truncated.ok and truncated.first_broken == 3 and "truncated" in truncated.reason

    _chain(path, 2)  # the log grows again past seq 4, but with different entries
    rewritten = verify_audit(path, key=KEY, anchor=anchor)
    assert not rewritten.ok and rewritten.first_broken == 4


def test_an_anchor_on_an_intact_log_verifies(tmp_path):
    path = tmp_path / "audit_log.jsonl"
    _chain(path, 3)
    anchor = audit_head(path)
    _chain(path, 2)
    assert verify_audit(path, key=KEY, anchor=anchor).ok


def test_a_routine_append_never_seals_unchained_lines(tmp_path):
    # Review finding (2026-09-26): without the key, someone could replace the log with invented
    # unchained lines naming a real analyst; the next append sealed them and `verify` said OK.
    path = tmp_path / "audit_log.jsonl"
    _write_lines(path, [_entry(actor_kind="analyst", actor_id="alice", outcome="forged").model_dump_json() + "\n"])
    with pytest.raises(AuditIntegrityError, match="seal"):
        append_audit(_entry(), path, key=KEY)
    assert len(_lines(path)) == 1


def test_a_legacy_unchained_prefix_is_sealed_only_by_the_explicit_seal_step(tmp_path):
    path = tmp_path / "audit_log.jsonl"
    legacy = [_entry(outcome="legacy-0"), _entry(outcome="legacy-1")]
    _write_lines(path, [e.model_dump_json() + "\n" for e in legacy])
    seal = seal_legacy(path, key=KEY, now=NOW)
    assert (seal.actor_kind, seal.action, seal.outcome) == ("system", "audit.seal", "sealed:2")
    assert seal_legacy(path, key=KEY, now=NOW) is None  # idempotent: nothing left to seal
    chained = _chain(path, 1)

    assert [json.loads(line).get("seq") for line in _lines(path)] == [None, None, 2, 3]
    assert load_audit(path) == legacy + [seal] + chained
    report = verify_audit(path, key=KEY)
    assert report.ok and (report.legacy, report.chained) == (2, 2)

    lines = _lines(path)
    lines[0] = _entry(outcome="legacy-rewritten").model_dump_json() + "\n"
    _write_lines(path, lines)
    broken = verify_audit(path, key=KEY)
    assert not broken.ok and broken.first_broken == 2 and "legacy" in broken.reason


def test_an_unchained_line_after_the_chain_started_is_a_break_and_blocks_appends(tmp_path):
    path = tmp_path / "audit_log.jsonl"
    _chain(path, 2)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(_entry(outcome="slipped in").model_dump_json() + "\n")

    report = verify_audit(path, key=KEY)
    assert not report.ok and report.first_broken == 2
    with pytest.raises(AuditIntegrityError):
        append_audit(_entry(), path, key=KEY)


def test_a_torn_tail_refuses_further_appends(tmp_path):
    path = tmp_path / "audit_log.jsonl"
    _chain(path, 2)
    with path.open("a", encoding="utf-8") as handle:
        handle.write('{"timestamp": "2026-09-')

    assert verify_audit(path, key=KEY).first_broken == 2
    with pytest.raises(AuditIntegrityError):
        append_audit(_entry(), path, key=KEY)


def test_concurrent_appends_never_fork_the_chain(tmp_path, monkeypatch):
    """FastAPI runs sync routes in a threadpool: two requests must never read the same head.
    The head read is slowed down to make an unlocked read-modify-append race certain."""
    import qorgan.audit as audit

    path = tmp_path / "audit_log.jsonl"
    original = audit._chain_head

    def slow_head(*args, **kwargs):
        result = original(*args, **kwargs)
        time.sleep(0.002)
        return result

    monkeypatch.setattr(audit, "_chain_head", slow_head)
    errors: list[BaseException] = []

    def worker(n: int) -> None:
        try:
            for i in range(15):
                append_audit(_entry(outcome=f"t{n}-{i}"), path, key=KEY)
        except BaseException as exc:  # pragma: no cover - surfaced below
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(n,)) for n in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert not errors
    report = verify_audit(path, key=KEY)
    assert report.ok, report.reason
    assert report.chained == 120
    assert [json.loads(line)["seq"] for line in _lines(path)] == list(range(120))


# --- the verifier CLI: `python -m qorgan.audit verify|head` -------------------------------


def test_cli_verify_reports_ok_and_the_first_broken_entry(tmp_path, monkeypatch, capsys):
    path = tmp_path / "audit_log.jsonl"
    _chain(path, 3)
    monkeypatch.setenv("QORGAN_AUDIT_CHAIN_KEY", KEY.decode())

    assert main(["verify", "--path", str(path)]) == 0
    out = capsys.readouterr().out
    assert "OK" in out and "3 entries" in out and "tail truncation" in out

    lines = _lines(path)
    _write_lines(path, [lines[0], lines[2]])
    assert main(["verify", "--path", str(path)]) == 1
    assert "BROKEN at entry 1" in capsys.readouterr().out


def test_cli_head_and_anchor_round_trip(tmp_path, monkeypatch, capsys):
    path = tmp_path / "audit_log.jsonl"
    _chain(path, 4)
    monkeypatch.setenv("QORGAN_AUDIT_CHAIN_KEY", KEY.decode())

    assert main(["head", "--path", str(path)]) == 0
    anchor = capsys.readouterr().out.strip()
    seq, mac = anchor.split(":")
    assert (int(seq), mac) == audit_head(path)

    assert main(["verify", "--path", str(path), "--anchor", anchor]) == 0
    _write_lines(path, _lines(path)[:2])
    assert main(["verify", "--path", str(path), "--anchor", anchor]) == 1


def test_cli_without_a_key_is_a_configuration_error(tmp_path, monkeypatch, capsys):
    path = tmp_path / "audit_log.jsonl"
    _chain(path, 1)
    monkeypatch.setenv("QORGAN_AUDIT_CHAIN_KEY", "")
    assert main(["verify", "--path", str(path)]) == 2
    assert "QORGAN_AUDIT_CHAIN_KEY" in capsys.readouterr().err


def test_verifier_output_never_contains_entry_fields(tmp_path, monkeypatch, capsys):
    path = tmp_path / "audit_log.jsonl"
    append_audit(_entry(actor_id="distinctive_actor", outcome="distinctive-outcome"), path, key=KEY)
    monkeypatch.setenv("QORGAN_AUDIT_CHAIN_KEY", KEY.decode())
    main(["verify", "--path", str(path)])
    out = capsys.readouterr().out
    assert "distinctive" not in out


def test_a_report_incident_subject_whose_hex_looks_like_a_number_is_accepted():
    """`incident:report-<10 hex>` is system-generated (intake.report_incident_id); ~0.25 % of
    them contain a digit run the scrubber reads as a phone number, which made `open` fail
    for those incidents (found 2026-09-25, same class as the receipt bug above)."""
    assert _entry(subject="incident:report-8882046218").subject == "incident:report-8882046218"
    with pytest.raises(ValidationError):
        _entry(subject="incident:report-8882046218 caller +7 701 234 56 78")
