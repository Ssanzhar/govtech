"""Append-only, content-free, tamper-evident audit log (PLAN_2026-09 C4/C5, ADR D19/D20).

One JSON line per action by a partner or an analyst: *who* did *what* to *which subject*
with *which outcome* (and, for opening a full transcript, *why*: a closed purpose code) --
never call content, never a number. The schema enforces that (`scrub_text` must be a fixed
point on every text field), so a careless caller cannot turn the audit log into a second
copy of the data it is supposed to account for.

**Tamper evidence.** Every line carries `seq` (its position), `prev` (the previous line's
MAC) and `mac` = HMAC-SHA256 over the line's canonical JSON with a server key
(`QORGAN_AUDIT_CHAIN_KEY`). Someone with write access to the file but not the key cannot
edit, insert, delete or reorder lines, or cut lines out of the middle, without
`verify_audit` naming the first broken entry. Lines written before chaining existed (no
`mac`) are allowed only as a leading prefix, and only an explicit, logged step seals them
(`python -m qorgan.audit seal`, after an operator has reviewed them): a routine append
refuses to chain onto unchained lines, so someone without the key cannot plant invented
"history" and have the next request vouch for it. The seal entry commits to a hash of the
prefix, so the legacy lines are frozen from then on -- but no key ever protected their content.

**What it cannot detect on its own:** removal of the newest entries (tail truncation) or
of the whole file -- the remaining chain is still valid. Record `python -m qorgan.audit head`
somewhere the server cannot rewrite (a ticket, a partner, a WORM store) and pass it back as
`--anchor`. Nor can it stop a party that holds the key (the server process itself) from
rewriting history; shipping lines to a separate trust domain is the deployment's job.

CLI: `python -m qorgan.audit verify [--path P] [--anchor SEQ:MAC]`, `... head [--path P]` and
`... seal [--path P]`.
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
import re
import sys
import threading
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import IO, Literal, get_args

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from qorgan.data.scrub import scrub_text

try:  # POSIX: also exclude other processes (a second worker, an ops script) during an append
    import fcntl
except ImportError:  # pragma: no cover - Windows dev boxes: the in-process lock still holds
    fcntl = None  # type: ignore[assignment]

AUDIT_FILENAME = "audit_log.jsonl"
_MAX_FIELD_CHARS = 200

ActorKind = Literal["partner", "analyst", "system"]
# Why a full transcript was opened (PLAN C4 purpose limitation). Deliberately small and closed:
# - pattern_review:  confirm / dismiss an organization or a model verdict -- the console's own purpose
# - citizen_request: the reporting citizen asked about their report (their data-subject rights)
# - partner_request: a connected partner (bank fraud desk, Anti-Fraud Center) asked about this case
AccessPurpose = Literal["pattern_review", "citizen_request", "partner_request"]
ACCESS_PURPOSES: tuple[str, ...] = get_args(AccessPurpose)

_CHAIN_FIELDS = ("seq", "prev", "mac")
_MAC_DOMAIN = b"qorgan-audit-v1\x00"
_GENESIS_DOMAIN = b"qorgan-audit-genesis-v1\x00"
_TAIL_WINDOW_BYTES = 64 * 1024
_APPEND_LOCK = threading.Lock()


class MissingAuditKeyError(RuntimeError):
    """No audit-chain key is configured: an audited action cannot be accounted for."""


class AuditIntegrityError(RuntimeError):
    """The log's tail is not a valid chain head; appending would extend a broken chain."""


class AuditEntry(BaseModel):
    model_config = ConfigDict(frozen=True)

    timestamp: datetime
    actor_kind: ActorKind
    actor_id: str = Field(min_length=1, max_length=_MAX_FIELD_CHARS)
    action: str = Field(min_length=1, max_length=_MAX_FIELD_CHARS)
    subject: str | None = Field(default=None, max_length=_MAX_FIELD_CHARS)
    outcome: str = Field(min_length=1, max_length=_MAX_FIELD_CHARS)
    purpose: AccessPurpose | None = None

    @field_validator("actor_id", "action", "subject", "outcome")
    @classmethod
    def _content_free(cls, value: str | None) -> str | None:
        if value is not None and not _is_system_id(value) and scrub_text(value) != value:
            raise ValueError("audit fields must not carry numbers, cards, IINs or other call content")
        return value


# A system-generated opaque id is not call content, but its random hex can contain a digit run
# the PII scrubber reads as a phone number -- measured at ~0.46 % of receipts (2026-09-24) and
# ~0.25 % of report-derived incident ids `incident:report-<hex>` (2026-09-25), each making the
# audited action fail. These exact shapes are exempt; everything else goes through the scrubber.
_SYSTEM_ID = re.compile(r"^(?:(?:receipt|report|incident|organization):|incident:report-)[0-9a-f]{8,64}$")


def _is_system_id(value: str) -> bool:
    return bool(_SYSTEM_ID.match(value))


# --- chain primitives -----------------------------------------------------------------------


def _canonical(record: dict) -> bytes:
    return json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _mac(key: bytes, record: dict) -> str:
    return hmac.new(key, _MAC_DOMAIN + _canonical(record), hashlib.sha256).hexdigest()


def _genesis(key: bytes, legacy_prefix: bytes) -> str:
    """The `prev` of the first chained line: commits to every byte before it."""
    return hmac.new(key, _GENESIS_DOMAIN + hashlib.sha256(legacy_prefix).digest(), hashlib.sha256).hexdigest()


def _require_key(key: bytes | None) -> bytes:
    if not key:
        raise MissingAuditKeyError("no audit-chain key configured (QORGAN_AUDIT_CHAIN_KEY)")
    return key


def _parse_record(line: bytes) -> dict | None:
    try:
        record = json.loads(line)
    except ValueError:
        return None
    return record if isinstance(record, dict) else None


def _is_chained(record: dict) -> bool:
    return "mac" in record


# --- append ---------------------------------------------------------------------------------


def append_audit(entry: AuditEntry, path: Path, *, key: bytes | None) -> Path:
    """Append `entry` as the next link of the chain. Serialized in-process (sync FastAPI
    routes run in a threadpool) and, on POSIX, across processes, so two concurrent appends
    can never read the same head and fork the chain. Raises `MissingAuditKeyError` without a
    key and `AuditIntegrityError` if the current tail is not a valid chain head -- including
    a log that starts with unchained lines nobody has sealed yet (`seal_legacy`)."""
    return _append(entry, path, key=key, sealing=False)


def seal_legacy(path: Path, *, key: bytes | None, now: datetime) -> AuditEntry | None:
    """Seal a log that holds only unchained (pre-chain) lines with one `system` entry, after an
    operator has reviewed them. Returns the seal entry, or None when there is nothing to seal."""
    key = _require_key(key)
    if not path.exists():
        return None
    records = [_parse_record(line) for line in path.read_bytes().splitlines() if line.strip()]
    if not records or any(record is not None and _is_chained(record) for record in records):
        return None
    entry = AuditEntry(
        timestamp=now, actor_kind="system", actor_id="qorgan", action="audit.seal", outcome=f"sealed:{len(records)}"
    )
    _append(entry, path, key=key, sealing=True)
    return entry


def _append(entry: AuditEntry, path: Path, *, key: bytes | None, sealing: bool) -> Path:
    key = _require_key(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    record = entry.model_dump(mode="json", exclude_none=True)
    with _APPEND_LOCK, path.open("a+b") as handle:
        if fcntl is not None:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            record["seq"], record["prev"] = _chain_head(handle, key, sealing=sealing)
            record["mac"] = _mac(key, record)
            handle.write(json.dumps(record, ensure_ascii=False).encode("utf-8") + b"\n")
            handle.flush()
            os.fsync(handle.fileno())  # on disk before the caller releases anything
        finally:
            if fcntl is not None:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    return path


def _chain_head(handle: IO[bytes], key: bytes, *, sealing: bool) -> tuple[int, str]:
    """`(seq, prev)` for the line about to be appended."""
    size = handle.seek(0, os.SEEK_END)
    last = _last_line(handle, size) if size else None
    if last is not None:
        if not last.endswith(b"\n"):
            raise AuditIntegrityError("the audit log ends in an unterminated line (torn write?); run `python -m qorgan.audit verify`")
        record = _parse_record(last)
        if record is None:
            raise AuditIntegrityError("the audit log's last line is not a JSON object; run `python -m qorgan.audit verify`")
        if _is_chained(record):
            seq, mac = record.get("seq"), record.get("mac")
            if type(seq) is not int or not isinstance(mac, str):
                raise AuditIntegrityError("the audit log's last line has a malformed chain link")
            return seq + 1, mac
    # Empty, blank or unchained tail: fine only if nothing is chained yet.
    handle.seek(0)
    data = handle.read()
    lines = [line for line in data.splitlines(keepends=True) if line.strip()]
    if any((record := _parse_record(line)) is not None and _is_chained(record) for line in lines):
        raise AuditIntegrityError("an unchained line follows chained entries; run `python -m qorgan.audit verify`")
    if lines and not sealing:
        raise AuditIntegrityError(
            f"the audit log starts with {len(lines)} unchained line(s) that no key protects; review them, "
            "then run `python -m qorgan.audit seal` once"
        )
    return len(lines), _genesis(key, data)


def _last_line(handle: IO[bytes], size: int) -> bytes | None:
    """The last non-blank line (with its newline, if it has one), or None if there is none."""
    start = max(0, size - _TAIL_WINDOW_BYTES)
    handle.seek(start)
    lines = [line for line in handle.read().splitlines(keepends=True) if line.strip()]
    if start > 0 and len(lines) < 2:  # the window may have cut the only line in it
        handle.seek(0)
        lines = [line for line in handle.read().splitlines(keepends=True) if line.strip()]
    return lines[-1] if lines else None


# --- read -----------------------------------------------------------------------------------


def load_audit(path: Path) -> list[AuditEntry]:
    """All entries in order (chain fields dropped); a missing file means none yet. Raises
    `ValueError` on a corrupt line. Reading does not verify -- use `verify_audit`."""
    if not path.exists():
        return []
    return [  # split on b"\n" only: str.splitlines() would also break at U+2028 inside a JSON string
        AuditEntry.model_validate_json(line)
        for line in path.read_bytes().split(b"\n")
        if line.strip()
    ]


def audit_head(path: Path) -> tuple[int, str] | None:
    """`(seq, mac)` of the newest chained entry -- the value to record externally as an anchor."""
    if not path.exists():
        return None
    head = None
    for line in path.read_bytes().splitlines():
        record = _parse_record(line) if line.strip() else None
        if record is not None and _is_chained(record):
            head = (record.get("seq"), record.get("mac"))
    return head


# --- verify ---------------------------------------------------------------------------------


@dataclass(frozen=True)
class AuditVerification:
    """Content-free result: counts, the first broken entry (0-based index among non-blank
    lines) and why, and the chain head for external anchoring."""

    ok: bool
    entries: int
    legacy: int
    chained: int
    first_broken: int | None = None
    reason: str | None = None
    head_seq: int | None = None
    head_mac: str | None = None


@dataclass
class _Walk:
    index: int = 0
    legacy: int = 0
    chained: int = 0
    prefix: bytes = b""
    prev: str | None = None
    macs: dict[int, str] = field(default_factory=dict)

    def result(self, *, ok: bool, first_broken: int | None = None, reason: str | None = None) -> AuditVerification:
        head_seq = max(self.macs) if self.macs else None
        return AuditVerification(
            ok=ok, entries=self.index, legacy=self.legacy, chained=self.chained, first_broken=first_broken,
            reason=reason, head_seq=head_seq, head_mac=self.macs.get(head_seq) if head_seq is not None else None,
        )


def verify_audit(path: Path, *, key: bytes | None, anchor: tuple[int, str] | None = None) -> AuditVerification:
    """Walk the whole log; stop at the first entry that breaks the chain. With `anchor`
    (`(seq, mac)` recorded earlier), also detect that the log was cut or rewritten after it."""
    key = _require_key(key)
    walk = _Walk()
    for raw in (path.read_bytes() if path.exists() else b"").splitlines(keepends=True):
        if not raw.strip():
            if not walk.chained:
                walk.prefix += raw  # blank lines inside the legacy prefix are sealed too
            continue
        problem = _check_line(raw, walk, key)
        if problem is not None:
            return walk.result(ok=False, first_broken=walk.index, reason=problem)
        walk.index += 1
    if anchor is not None:
        seq, mac = anchor
        if seq >= walk.index:
            return walk.result(ok=False, first_broken=walk.index, reason=f"the log ends before the anchored entry {seq} (truncated)")
        recorded = walk.macs.get(seq)
        if recorded is None or not hmac.compare_digest(recorded, mac):
            return walk.result(ok=False, first_broken=seq, reason=f"entry {seq} differs from the anchor (history rewritten)")
    return walk.result(ok=True)


def _check_line(raw: bytes, walk: _Walk, key: bytes) -> str | None:
    if not raw.endswith(b"\n"):
        return "unterminated last line (torn write, or bytes appended by hand)"
    record = _parse_record(raw)
    if record is None:
        return "not a JSON object"
    if not _is_chained(record):
        if walk.chained:
            return "an unchained line after the chain started (inserted without the key?)"
        if not _valid_entry(record):
            return "a legacy line that fails the audit schema"
        walk.legacy += 1
        walk.prefix += raw
        return None
    if walk.prev is None:
        walk.prev = _genesis(key, walk.prefix)
    seq = record.get("seq")
    if type(seq) is not int or seq != walk.index:
        return f"sequence {seq!r} where {walk.index} was expected (lines deleted, inserted or reordered)"
    if record.get("prev") != walk.prev:
        if walk.chained == 0 and walk.legacy:
            return "the first chained entry does not match the legacy prefix before it (legacy lines altered)"
        return "link to the previous entry does not match (history altered before this entry)"
    claimed = record.pop("mac")
    if not isinstance(claimed, str) or not hmac.compare_digest(claimed, _mac(key, record)):
        return "entry MAC does not match (edited, or written without the server key)"
    if not _valid_entry(record):
        return "a chained line that fails the audit schema"
    walk.prev = claimed
    walk.chained += 1
    walk.macs[seq] = claimed
    return None


def _valid_entry(record: dict) -> bool:
    try:
        AuditEntry.model_validate({k: v for k, v in record.items() if k not in _CHAIN_FIELDS})
    except ValidationError:
        return False
    return True


# --- CLI ------------------------------------------------------------------------------------


def _parse_anchor(value: str) -> tuple[int, str]:
    seq, _, mac = value.partition(":")
    if not seq.isdigit() or not re.fullmatch(r"[0-9a-f]{64}", mac):
        raise argparse.ArgumentTypeError("anchor must be SEQ:MAC as printed by `python -m qorgan.audit head`")
    return int(seq), mac


def main(argv: Sequence[str] | None = None) -> int:
    from qorgan.config import get_config

    parser = argparse.ArgumentParser(prog="python -m qorgan.audit", description="Tamper-evident audit log tools.")
    sub = parser.add_subparsers(dest="command", required=True)
    for name, help_text in (("verify", "verify the chain; exit 1 at the first broken entry"),
                            ("head", "print SEQ:MAC of the newest entry, to record as an external anchor"),
                            ("seal", "after reviewing them, seal the unchained lines a legacy log starts with")):
        command = sub.add_parser(name, help=help_text)
        command.add_argument("--path", type=Path, default=None, help="default: <QORGAN_DATA_DIR>/processed/audit_log.jsonl")
        if name == "verify":
            command.add_argument("--anchor", type=_parse_anchor, default=None, help="SEQ:MAC recorded earlier with `head`")
    args = parser.parse_args(argv)

    cfg = get_config()
    path = args.path or cfg.data_dir / "processed" / AUDIT_FILENAME
    if args.command == "head":
        head = audit_head(path)
        print(f"{head[0]}:{head[1]}" if head else "")
        return 0
    if cfg.audit_chain_key is None:
        print(f"cannot {args.command}: QORGAN_AUDIT_CHAIN_KEY is not set", file=sys.stderr)
        return 2
    if args.command == "seal":
        from datetime import UTC

        seal = seal_legacy(path, key=cfg.audit_chain_key.get_secret_value(), now=datetime.now(UTC))
        print(f"sealed {seal.outcome.split(':')[1]} legacy line(s)" if seal else "nothing to seal")
        return 0
    report = verify_audit(path, key=cfg.audit_chain_key.get_secret_value(), anchor=args.anchor)
    print(f"audit log: {path}")
    print(f"{report.entries} entries (legacy unchained: {report.legacy}, chained: {report.chained})")
    if report.head_seq is not None:
        print(f"head: {report.head_seq}:{report.head_mac}")
    if not report.ok:
        print(f"BROKEN at entry {report.first_broken}: {report.reason}")
        return 1
    print("OK: chain intact" + (", anchor matched." if args.anchor else "."))
    if report.legacy:
        print(f"Note: the first {report.legacy} line(s) predate the chain; no key protects their content.")
    print("Not detectable without an external anchor (`head`): tail truncation, or removal of the whole file.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
