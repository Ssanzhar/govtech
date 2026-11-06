"""What one report, or one damaged file, must never do to everybody else's (audit 2026-11-06, ADR D85).

The stores are JSONL files, the route is public and anonymous, and the deletion promise (invariant 6)
holds even when a derived file is broken.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

import qorgan.api_admin as api_admin
from qorgan.analytics.intake import forget_report, ingest_pending, report_incident_id
from qorgan.analytics.pipeline import load_embeddings_npz, load_organizations_jsonl, save_embeddings_npz, write_organizations_jsonl
from qorgan.api import app
from qorgan.api_reports import _LIMITER
from qorgan.data.incident_seed import load_incidents_jsonl, write_incidents_jsonl
from qorgan.data.schema import Incident, Label, Organization, spans_from_phrases
from qorgan.data.span_grounding import ground_flagged_phrases
from qorgan.reports.purge import purge
from qorgan.reports.store import load_reports, prepare_report
from support.analysts import as_analyst
from support.numbers import TEST_HMAC_KEY, hashed, prefix, stored_report

NUMBER = "+7 700 101 20 30"


class _Embedder:
    """Deterministic, offline."""

    def encode(self, texts, normalize_embeddings=True, convert_to_numpy=True):
        rows = []
        for text in texts:
            rng = np.random.default_rng(sum(text.encode("utf-8")) % (2**32))
            vector = rng.normal(size=8)
            rows.append(vector / np.linalg.norm(vector))
        return np.array(rows, dtype=np.float32)


class _BrokenEmbedder:
    def encode(self, *_args, **_kwargs):
        raise RuntimeError("the embedding model cannot load here")


@pytest.fixture()
def paths(tmp_path):
    processed = tmp_path / "processed"
    processed.mkdir()
    return {
        "reports_path": processed / "citizen_reports.jsonl",
        "incidents_path": processed / "incidents.jsonl",
        "organizations_path": processed / "organizations.jsonl",
        "embeddings_path": processed / "incident_embeddings.npz",
    }


@pytest.fixture()
def client(tmp_path, monkeypatch) -> TestClient:
    monkeypatch.setenv("QORGAN_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("QORGAN_TAXONOMY_PATH", "data/taxonomy/tactics.yaml")
    monkeypatch.setenv("QORGAN_CUE_LEXICON_PATH", "data/lexicon/hard_signal_cues.yaml")
    monkeypatch.setenv("QORGAN_REASSURANCE_PATTERNS_PATH", "data/lexicon/reassurance_patterns.yaml")
    monkeypatch.setenv("QORGAN_NUMBER_HMAC_KEY", "tests-only-key")
    monkeypatch.setattr(api_admin, "_EMBEDDER_OVERRIDE", _Embedder())
    (tmp_path / "processed").mkdir()
    _LIMITER.reset()
    return TestClient(app, raise_server_exceptions=False)


def _body(transcript: str = "Алло, это служба безопасности банка. Продиктуйте код.", **extra) -> dict:
    return {"transcript": transcript, "risk_score": 90.0, "consent": True, "consent_version": "report-v1", **extra}


def _seed_analysis(paths, embedder=None) -> list[Incident]:
    incidents = [
        Incident(
            id=f"inc_{i}", dialogue_id=f"d{i}", transcript=f"звонок номер {i}: продиктуйте код", label=Label(risk=0.9),
            number_hash=hashed(NUMBER), number_prefix=prefix(NUMBER), timestamp=datetime(2026, 10, 1, 12, i),
        )
        for i in range(3)
    ]
    write_incidents_jsonl(incidents, paths["incidents_path"])
    write_organizations_jsonl(
        [Organization(id="org_0", members=tuple(i.id for i in incidents), numbers=(hashed(NUMBER),), priority=0.5, representative_script=incidents[0].transcript)],
        paths["organizations_path"],
    )
    save_embeddings_npz([i.id for i in incidents], (embedder or _Embedder()).encode([i.transcript for i in incidents]), paths["embeddings_path"])
    return incidents


def _ingested_report(paths, transcript: str = "Отчёт гражданина: продиктуйте код из смс.", receipt: str = "a" * 24):
    report = stored_report(number=NUMBER, transcript=transcript, receipt_id=receipt)
    paths["reports_path"].write_text(report.model_dump_json() + "\n", encoding="utf-8")
    ingest_pending(**paths, embedder=_Embedder())
    return report


# --- one report must not poison the store -------------------------------------------------------


@pytest.mark.parametrize("separator", [" ", " ", "\u0085"])
def test_a_unicode_line_separator_in_a_report_does_not_break_the_store(client, tmp_path, separator):
    """str.splitlines() also breaks on these, and JSON leaves them unescaped: one anonymous POST
    used to make every later delete, partner call, ingest and purge raise."""
    good = client.post("/api/reports", json=_body("Обычное сообщение: продиктуйте код из смс."))
    poison = client.post("/api/reports", json=_body(f"Скопировано с сайта.{separator}Вторая строка: назовите код."))
    assert good.status_code == poison.status_code == 201

    stored = load_reports(tmp_path / "processed" / "citizen_reports.jsonl")
    assert len(stored) == 2 and separator in stored[1].transcript

    assert client.delete(f"/api/reports/{good.json()['receipt_id']}").status_code == 204
    ingest = client.post("/api/admin/ingest", headers=as_analyst())
    assert ingest.status_code == 200, ingest.text
    assert ingest.json()["ingested"] == 1
    incidents = load_incidents_jsonl(tmp_path / "processed" / "incidents.jsonl")
    assert separator in incidents[0].transcript


def test_a_line_separator_in_an_analyst_note_does_not_hide_every_verdict(client, tmp_path):
    from qorgan.analytics.feedback import FeedbackEvent, append_feedback, load_feedback, snapshot_for

    org = Organization(id="org_0", members=("a",), numbers=(hashed(NUMBER),))
    path = tmp_path / "processed" / "org_feedback.jsonl"
    event = FeedbackEvent(timestamp=datetime.now(UTC), analyst_id="a1", action="confirm", org=snapshot_for(org), note="см. заявку")
    append_feedback(event, path)
    assert load_feedback(path)[0].note == "см. заявку"


def test_a_corrupt_line_is_refused_without_quoting_it(tmp_path):
    """The parse error used to carry the stored text into HTTP responses and logs."""
    path = tmp_path / "citizen_reports.jsonl"
    path.write_text('{"receipt_id": 5, "transcript": "секретный текст звонка"}\n', encoding="utf-8")
    with pytest.raises(ValueError) as caught:
        load_reports(path)
    assert "секретный" not in str(caught.value)
    assert "line 1" in str(caught.value)


@pytest.mark.parametrize("blank", [" ", "\n", "\t \t", " "])
def test_a_blank_flagged_phrase_never_reaches_a_span(blank):
    assert ground_flagged_phrases([blank, "код"], f"а{blank}код", f"а{blank}код") == ("код",)
    assert [s.text for s in spans_from_phrases([blank, "код"], f"а{blank}код")] == ["код"]


def test_a_blank_flagged_phrase_does_not_block_ingest(client):
    assert client.post("/api/reports", json=_body("а б в: продиктуйте код", flagged_phrases=[" "])).status_code == 201
    second = client.post("/api/reports", json=_body("Другое сообщение: назовите код"))
    assert second.status_code == 201
    ingest = client.post("/api/admin/ingest", headers=as_analyst())
    assert ingest.status_code == 200, ingest.text
    assert ingest.json()["ingested"] == 2


def test_a_row_that_cannot_be_converted_does_not_block_the_others(paths, monkeypatch):
    """Rows already stored by an older build: the bad one stays pending, the rest go through."""
    import qorgan.analytics.intake as intake

    bad = stored_report(number=NUMBER, transcript="плохая строка", receipt_id="b" * 24)
    good = stored_report(number=NUMBER, transcript="хорошая строка: продиктуйте код", receipt_id="c" * 24)
    paths["reports_path"].write_text(bad.model_dump_json() + "\n" + good.model_dump_json() + "\n", encoding="utf-8")
    real = intake.report_to_incident

    def explode(report, *, incident_id):
        if report.receipt_id == bad.receipt_id:
            raise ValueError("this row cannot become an incident")
        return real(report, incident_id=incident_id)

    monkeypatch.setattr(intake, "report_to_incident", explode)
    summary = intake.ingest_pending(**paths, embedder=_Embedder())
    assert summary.ingested == 1 and summary.skipped == 1


# --- the embedding file is a cache: damage to it costs time, never an answer ---------------------


@pytest.mark.parametrize("damage", [b"", b"not a zip file at all", b"PK\x03\x04truncated"])
def test_a_damaged_embedding_cache_is_a_miss_on_ingest(paths, damage):
    _seed_analysis(paths)
    paths["embeddings_path"].write_bytes(damage)
    report = stored_report(number=NUMBER, transcript="новое сообщение: продиктуйте код", receipt_id="d" * 24)
    paths["reports_path"].write_text(report.model_dump_json() + "\n", encoding="utf-8")

    summary = ingest_pending(**paths, embedder=_Embedder())

    assert summary.ingested == 1
    ids, rows = load_embeddings_npz(paths["embeddings_path"])
    assert len(ids) == 4 and len(rows) == 4


def test_an_empty_width_cache_from_a_signals_only_first_ingest_is_a_miss(paths):
    """The first ingest on an empty stream was signals-only: an (n, 0) matrix that nothing can
    ever stack with, so every later ingest answered 503."""
    report = prepare_report(
        transcript="", phone_number=NUMBER, flagged_phrases=(), tactic_ids=("otp_request",), receipt_id="e" * 24,
        timestamp=datetime(2026, 9, 17, 10, 0, tzinfo=UTC), risk_score=100.0, hmac_key=TEST_HMAC_KEY,
        source="partner", consent_basis="customer_consent", partner_id="bank_a", partner_reference="CASE-1",
    )
    paths["reports_path"].write_text(report.model_dump_json() + "\n", encoding="utf-8")
    assert ingest_pending(**paths, embedder=_Embedder()).ingested == 1

    later = stored_report(number=NUMBER, transcript="сообщение с текстом: продиктуйте код", receipt_id="f" * 24)
    with paths["reports_path"].open("a", encoding="utf-8") as handle:
        handle.write(later.model_dump_json() + "\n")
    assert ingest_pending(**paths, embedder=_Embedder()).ingested == 1
    _, rows = load_embeddings_npz(paths["embeddings_path"])
    assert rows.shape[1] > 0


def test_saving_the_cache_replaces_it_atomically(paths, monkeypatch):
    """The one store that was written in place: a crash left a truncated file."""
    _seed_analysis(paths)
    before = paths["embeddings_path"].read_bytes()

    def crash(target, **_kwargs):  # half a file, then the disk fills
        if hasattr(target, "write"):
            target.write(b"PK half a file")
        else:
            Path(target).write_bytes(b"PK half a file")
        raise OSError("disk full")

    monkeypatch.setattr(np, "savez_compressed", crash)
    with pytest.raises(OSError):
        save_embeddings_npz(["x"], np.zeros((1, 8), dtype=np.float32), paths["embeddings_path"])
    assert paths["embeddings_path"].read_bytes() == before
    assert [p.name for p in paths["embeddings_path"].parent.iterdir() if p.name.endswith(".tmp")] == []


# --- a deletion finishes, or can be repeated -----------------------------------------------------


def _assert_gone(paths, report) -> None:
    incident_id = report_incident_id(report)
    assert all(r.receipt_id != report.receipt_id for r in load_reports(paths["reports_path"]))
    incidents = load_incidents_jsonl(paths["incidents_path"])
    assert incident_id not in {i.id for i in incidents}
    for org in load_organizations_jsonl(paths["organizations_path"]):
        assert incident_id not in org.members
        assert report.transcript not in (org.representative_script or "")
    if paths["embeddings_path"].exists():
        ids, rows = load_embeddings_npz(paths["embeddings_path"])
        assert incident_id not in ids and len(ids) == len(rows)


def test_a_damaged_cache_does_not_stop_a_deletion(paths):
    _seed_analysis(paths)
    report = _ingested_report(paths)
    paths["embeddings_path"].write_bytes(b"PK\x03\x04truncated")

    summary = forget_report(report.receipt_id, **paths)

    assert summary is not None and summary.incident_removed
    _assert_gone(paths, report)


def test_a_misaligned_cache_keeps_no_row_of_the_deleted_incident(paths):
    _seed_analysis(paths)
    report = _ingested_report(paths)
    ids, rows = load_embeddings_npz(paths["embeddings_path"])
    save_embeddings_npz(list(reversed(ids)), rows, paths["embeddings_path"])

    assert forget_report(report.receipt_id, **paths).incident_removed
    _assert_gone(paths, report)


def test_a_deletion_works_when_no_model_can_load(paths, monkeypatch):
    """No cache and no embedder: the promise does not wait for re-clustering."""
    import qorgan.analytics.intake as intake

    _seed_analysis(paths)
    report = _ingested_report(paths)
    paths["embeddings_path"].unlink()
    monkeypatch.setattr(intake, "analyze_incidents", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no model")))

    assert forget_report(report.receipt_id, **paths).incident_removed
    _assert_gone(paths, report)
    assert load_organizations_jsonl(paths["organizations_path"]), "the other incidents keep their organization"


def test_a_failed_deletion_can_be_repeated(paths, monkeypatch):
    """The report row used to go first: any later failure left an incident no receipt could reach."""
    import qorgan.analytics.intake as intake

    _seed_analysis(paths)
    report = _ingested_report(paths)
    real = intake.write_incidents_jsonl
    monkeypatch.setattr(intake, "write_incidents_jsonl", lambda *a, **k: (_ for _ in ()).throw(OSError("disk full")))
    with pytest.raises(OSError):
        forget_report(report.receipt_id, **paths)
    assert any(r.receipt_id == report.receipt_id for r in load_reports(paths["reports_path"])), "the receipt still works"

    monkeypatch.setattr(intake, "write_incidents_jsonl", real)
    assert forget_report(report.receipt_id, **paths).incident_removed
    _assert_gone(paths, report)


def test_a_retried_post_is_forgotten_with_the_receipt_the_citizen_kept(client, tmp_path):
    """A lost response and a retry store the same report twice; only the second receipt was seen."""
    stamp = datetime.now(UTC).isoformat()
    first = client.post("/api/reports", json=_body(timestamp=stamp))
    second = client.post("/api/reports", json=_body(timestamp=stamp))
    assert first.json()["report_id"] == second.json()["report_id"]

    assert client.delete(f"/api/reports/{second.json()['receipt_id']}").status_code == 204

    assert load_reports(tmp_path / "processed" / "citizen_reports.jsonl") == []
    assert client.post("/api/admin/ingest", headers=as_analyst()).json()["ingested"] == 0


def test_purge_removes_a_report_incident_with_no_report_behind_it(paths):
    """Orphans an older build left behind: no receipt reaches them, so retention must."""
    _seed_analysis(paths)
    report = _ingested_report(paths)
    orphan_id = report_incident_id(report)
    paths["reports_path"].write_text("", encoding="utf-8")

    purge(retention_days=180, now=datetime.now(UTC) + timedelta(days=1), **paths)

    assert orphan_id not in {i.id for i in load_incidents_jsonl(paths["incidents_path"])}
    assert all(orphan_id not in org.members for org in load_organizations_jsonl(paths["organizations_path"]))
    assert {"inc_0", "inc_1", "inc_2"} <= {i.id for i in load_incidents_jsonl(paths["incidents_path"])}, "seeds stay"


# --- a number is a number, however it is written ---------------------------------------------------


@pytest.mark.parametrize(
    "written",
    [
        "7011234567",
        "(701) 123-45-67",
        "(701)1234567",
        "701.123.45.67",
        "+7.701.123.45.67",
        "+1 202 555 0123",
        "+12025550123",
        "+49 151 23456789",
        "+86 138 0013 8000",
    ],
)
def test_every_common_way_of_writing_a_phone_number_is_masked(written):
    from qorgan.data.scrub import scrub_text

    masked = scrub_text(f"Перезвоните на {written} сегодня.")
    assert masked == "Перезвоните на [PHONE] сегодня.", masked


@pytest.mark.parametrize(
    "kept",
    [
        "Переведите 7000000000 тенге на этот счёт.",  # an amount, not a number to call back
        "Встреча +5 минут.",
        "Рост 10.5.2026 и 12.03.",
        "ул. Абая 7, кв. 12",
        "Сумма +1 500 тенге.",
    ],
)
def test_amounts_and_dates_are_not_phone_numbers(kept):
    from qorgan.data.scrub import scrub_text

    assert "[PHONE]" not in scrub_text(kept), scrub_text(kept)


def test_a_partner_reference_that_is_a_phone_number_is_refused():
    """The reference is stored and audited in clear; it must not smuggle a number in."""
    from pydantic import ValidationError

    from qorgan.api_partner import PartnerReportIn

    fields = {"consent_basis": "customer_consent", "tactic_ids": ["otp_request"], "risk_score": 80.0}
    assert PartnerReportIn(**fields, partner_reference="CASE-2026-17").partner_reference == "CASE-2026-17"
    with pytest.raises(ValidationError):
        PartnerReportIn(**fields, partner_reference="77011234567")


def test_a_scrub_that_needs_two_passes_is_stored_not_refused():
    """A mask can expose a number to the next rule; that used to read as "caller number not understood"."""
    from support.numbers import stored_report as store

    from qorgan.data.scrub import scrub_text

    report = store(number=None, transcript="меня зовут KZ12 300123 9010", flagged_phrases=())
    assert scrub_text(report.transcript) == report.transcript


def test_a_partner_id_that_looks_like_a_phone_number_is_refused_at_start_up():
    """It passed the config check, and then every partner action failed in the audit log."""
    from qorgan.partners import _parse_entry

    with pytest.raises(ValueError, match="personal data"):
        _parse_entry("bank-77011234567:0123456789abcdef0123", 100)


# --- the lock holds across processes --------------------------------------------------------------


def test_the_reports_lock_excludes_another_process(tmp_path, monkeypatch):
    """A cron purge rewrites the whole file: appends from the server landing meanwhile were lost."""
    import subprocess
    import sys

    from qorgan.reports.store import REPORTS_LOCK

    monkeypatch.setenv("QORGAN_DATA_DIR", str(tmp_path))
    probe = (
        "import fcntl, sys\n"
        f"handle = open({str(tmp_path / 'processed' / '.reports.lock')!r}, 'a')\n"
        "try:\n"
        "    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)\n"
        "except BlockingIOError:\n"
        "    sys.exit(3)\n"
    )
    with REPORTS_LOCK, REPORTS_LOCK:  # re-entrant in this thread
        assert subprocess.run([sys.executable, "-c", probe], check=False).returncode == 3
    assert subprocess.run([sys.executable, "-c", probe], check=False).returncode == 0


# --- the anonymous scoring route, and the console's stale view -------------------------------------


def test_analyze_is_rate_limited_per_client(client):
    from qorgan.api import _ANALYZE_LIMITER

    _ANALYZE_LIMITER.reset()
    payload = {"transcript": "Назовите код из смс", "backend": "mock"}
    statuses = [client.post("/api/analyze", json=payload).status_code for _ in range(_ANALYZE_LIMITER.max_requests + 1)]
    assert statuses[:-1] == [200] * _ANALYZE_LIMITER.max_requests
    assert statuses[-1] == 429
    _ANALYZE_LIMITER.reset()


def test_analyze_refuses_a_transcript_of_thousands_of_lines(client):
    from qorgan.api import _ANALYZE_LIMITER, _MAX_TRANSCRIPT_LINES

    _ANALYZE_LIMITER.reset()
    res = client.post("/api/analyze", json={"transcript": "а\n" * (_MAX_TRANSCRIPT_LINES + 1), "backend": "mock"})
    assert res.status_code == 422
    assert client.post("/api/analyze", json={"transcript": "а\n" * 10, "backend": "mock"}).status_code == 200


def test_a_chunked_body_cannot_hide_behind_a_small_content_length(client):
    res = client.post(
        "/api/analyze",
        content=b'{"transcript": "a"}',
        headers={"Content-Type": "application/json", "Content-Length": "10", "Transfer-Encoding": "chunked"},
    )
    assert res.status_code == 411


def _seed_two_orgs(tmp_path) -> None:
    processed = tmp_path / "processed"
    other = "+7 701 202 30 40"
    incidents = [
        Incident(id=f"a{i}", dialogue_id=f"a{i}", transcript=f"звонок а {i}: продиктуйте код", label=Label(risk=0.9), number_hash=hashed(NUMBER), number_prefix=prefix(NUMBER), timestamp=datetime(2026, 10, 1, 12, i))
        for i in range(2)
    ] + [
        Incident(id=f"b{i}", dialogue_id=f"b{i}", transcript=f"звонок б {i}: назовите код", label=Label(risk=0.9), number_hash=hashed(other), number_prefix=prefix(other), timestamp=datetime(2026, 10, 1, 13, i))
        for i in range(2)
    ]
    write_incidents_jsonl(incidents, processed / "incidents.jsonl")
    write_organizations_jsonl(
        [
            Organization(id="org_0", members=("a0", "a1"), numbers=(hashed(NUMBER),), priority=0.9),
            Organization(id="org_1", members=("b0", "b1"), numbers=(hashed(other),), priority=0.8),
        ],
        processed / "organizations.jsonl",
    )


def test_a_verdict_from_a_stale_view_is_refused_not_misapplied(client, tmp_path):
    """`org_<n>` is positional: after a deletion or an ingest it can name another operation."""
    _seed_two_orgs(tmp_path)
    overview = client.get("/api/admin/overview", headers=as_analyst()).json()["organizations"]
    shown = {o["id"]: o for o in overview}
    assert shown["org_0"]["fingerprint"] != shown["org_1"]["fingerprint"]

    stale = client.post(
        "/api/admin/organizations/org_0/feedback", headers=as_analyst(),
        json={"action": "dismiss", "fingerprint": shown["org_1"]["fingerprint"]},
    )
    assert stale.status_code == 409
    fresh = client.post(
        "/api/admin/organizations/org_0/feedback", headers=as_analyst(),
        json={"action": "dismiss", "fingerprint": shown["org_0"]["fingerprint"]},
    )
    assert fresh.status_code == 200 and fresh.json()["feedback"] == "dismissed"


def test_the_partner_export_follows_the_analyst(client, tmp_path, monkeypatch):
    from qorgan.api_partner_export import _load_analysis

    _seed_two_orgs(tmp_path)
    assert client.post("/api/admin/organizations/org_0/feedback", headers=as_analyst(), json={"action": "dismiss"}).status_code == 200
    organizations, _ = _load_analysis()
    assert next(o for o in organizations if o.id == "org_0").priority < 0.9


def test_a_short_call_is_never_shown_whole_without_opening_the_case(client, tmp_path):
    """An excerpt of a short call used to be the whole call, past the investigator's gate."""
    _seed_two_orgs(tmp_path)
    samples = client.get("/api/admin/organizations/org_0", headers=as_analyst()).json()["sample_incidents"]
    for sample in samples:
        assert sample["excerpt"].rstrip("…") != "" and "звонок а 0: продиктуйте код" not in sample["excerpt"]


def test_ingest_is_audited_before_it_changes_anything(client, tmp_path):
    """With the audit log unavailable, an ingest used to change the analysis and then answer 503."""
    _seed_two_orgs(tmp_path)
    assert client.post("/api/reports", json=_body()).status_code == 201
    before = load_incidents_jsonl(tmp_path / "processed" / "incidents.jsonl")
    (tmp_path / "processed" / "audit_log.jsonl").write_bytes(b'{"torn": ')  # a torn tail: nothing can be appended

    assert client.post("/api/admin/ingest", headers=as_analyst()).status_code == 503

    assert load_incidents_jsonl(tmp_path / "processed" / "incidents.jsonl") == before


def test_ingest_waits_while_first_start_seeding_runs(client, tmp_path):
    processed = tmp_path / "processed"
    write_incidents_jsonl([], processed / "incidents.jsonl")
    (processed / "incidents.jsonl").write_text("", encoding="utf-8")
    res = client.post("/api/admin/ingest", headers=as_analyst())
    assert res.status_code == 409


def test_an_ingest_failure_does_not_put_stored_text_in_the_response(client, tmp_path):
    (tmp_path / "processed" / "citizen_reports.jsonl").write_text('{"receipt_id": 5, "transcript": "секретный текст"}\n', encoding="utf-8")
    res = client.post("/api/admin/ingest", headers=as_analyst())
    assert res.status_code == 503
    assert "секретный" not in res.text


# --- review of this change, 2026-11-06 --------------------------------------------------------------


def test_a_fingerprint_can_never_read_as_a_phone_number():
    """A hex digest holds digit runs; the audit log's scrubber then refused the verdict for good."""
    from qorgan.analytics.presentation import org_fingerprint
    from qorgan.data.scrub import scrub_text

    for i in range(5000):
        fingerprint = org_fingerprint(Organization(id="org_3", members=(f"inc_{i}", f"inc_{i + 1}")))
        subject = f"org:org_3#{fingerprint}"
        assert scrub_text(subject) == subject, subject
        assert not any(character.isdigit() for character in fingerprint)


@pytest.mark.parametrize("store", ["incidents.jsonl", "organizations.jsonl", "org_feedback.jsonl"])
def test_a_corrupt_derived_store_does_not_stop_a_deletion(paths, store):
    """The report row now goes last, so a line nothing can parse must not block it for ever: a
    deletion that cannot read a line drops it, which is the safe direction."""
    _seed_analysis(paths)
    report = _ingested_report(paths)
    target = paths["reports_path"].parent / store
    with target.open("a", encoding="utf-8") as handle:
        handle.write('{"not": "a record", "transcript": "секретный текст"}\n')

    assert forget_report(report.receipt_id, **paths) is not None

    assert all(r.receipt_id != report.receipt_id for r in load_reports(paths["reports_path"]))
    assert report_incident_id(report) not in {i.id for i in load_incidents_jsonl(paths["incidents_path"], skip_corrupt=True)}
    assert "секретный" not in target.read_text(encoding="utf-8")


def test_a_corrupt_report_line_does_not_stop_a_deletion(paths):
    _seed_analysis(paths)
    report = _ingested_report(paths)
    with paths["reports_path"].open("a", encoding="utf-8") as handle:
        handle.write('{"receipt_id": 5}\n')

    assert forget_report(report.receipt_id, **paths) is not None
    assert load_reports(paths["reports_path"]) == []


def test_one_report_that_cannot_be_forgotten_does_not_stop_the_purge(paths, monkeypatch):
    import qorgan.reports.purge as purge_module

    _seed_analysis(paths)
    old = datetime(2026, 1, 1, tzinfo=UTC)
    rows = [stored_report(number=NUMBER, transcript=f"старое сообщение {i}: продиктуйте код", receipt_id=str(i) * 24, received_at=old, timestamp=old) for i in range(1, 4)]
    paths["reports_path"].write_text("".join(r.model_dump_json() + "\n" for r in rows), encoding="utf-8")
    real = purge_module.forget_report

    def flaky(receipt_id, **kwargs):
        if receipt_id == "1" * 24:
            raise OSError("disk full")
        return real(receipt_id, **kwargs)

    monkeypatch.setattr(purge_module, "forget_report", flaky)
    removed = purge(retention_days=180, now=datetime(2026, 11, 6, tzinfo=UTC), **paths)
    assert removed == ["2" * 24, "3" * 24]


def test_every_copy_of_a_retried_report_gives_up_its_number(paths):
    """Retried with an edited number: the citizen holds only the second receipt."""
    from qorgan.analytics.feedback import FeedbackEvent, OrgSnapshot, append_feedback, load_feedback

    _seed_analysis(paths)
    other = "+7 705 404 50 60"
    stamp = datetime(2026, 10, 2, 9, 0, tzinfo=UTC)
    text = "Повторно отправленное сообщение: продиктуйте код"
    first = stored_report(number=NUMBER + "1"[:0], transcript=text, receipt_id="a" * 24, timestamp=stamp)
    second = stored_report(number=other, transcript=text, receipt_id="b" * 24, timestamp=stamp)
    first = first.model_copy(update={"number_hash": hashed("+7 708 111 22 33")})
    paths["reports_path"].write_text(first.model_dump_json() + "\n" + second.model_dump_json() + "\n", encoding="utf-8")
    incident_id = report_incident_id(second)
    feedback = paths["reports_path"].parent / "org_feedback.jsonl"
    append_feedback(
        FeedbackEvent(
            timestamp=stamp, analyst_id="a1", action="confirm",
            org=OrgSnapshot(numbers=(hashed("+7 708 111 22 33"), hashed(other)), members=(incident_id, "inc_0")),
        ),
        feedback,
    )

    assert forget_report(second.receipt_id, **paths) is not None

    numbers = {n for event in load_feedback(feedback) for n in event.org.numbers}
    assert hashed("+7 708 111 22 33") not in numbers and hashed(other) not in numbers


def test_a_failing_file_lock_leaves_the_store_usable(tmp_path, monkeypatch):
    """`__exit__` does not run when `__enter__` raises: every other thread would have hung."""
    import threading

    import qorgan.reports.store as store

    monkeypatch.setenv("QORGAN_DATA_DIR", str(tmp_path))

    def no_locks(*_args, **_kwargs):
        raise OSError("ENOLCK: no locks on this volume")

    monkeypatch.setattr(store.fcntl, "flock", no_locks)
    with store.REPORTS_LOCK:
        pass
    entered = threading.Event()

    def other_thread():
        with store.REPORTS_LOCK:
            entered.set()

    thread = threading.Thread(target=other_thread, daemon=True)
    thread.start()
    thread.join(timeout=5)
    assert entered.is_set(), "the lock must not stay held"


def test_a_config_failure_does_not_leave_the_lock_held(monkeypatch):
    import threading

    import qorgan.reports.store as store

    def broken():
        raise ValueError("bad config")

    monkeypatch.setattr(store, "get_config", broken)
    with pytest.raises(ValueError), store.REPORTS_LOCK:
        pass
    monkeypatch.undo()
    entered = threading.Event()

    def other_thread():
        monkeypatch.setenv("QORGAN_DATA_DIR", "/tmp")
        with store.REPORTS_LOCK:
            entered.set()

    thread = threading.Thread(target=other_thread, daemon=True)
    thread.start()
    thread.join(timeout=5)
    assert entered.is_set()


def test_ingest_says_how_many_reports_it_could_not_take(client, tmp_path, monkeypatch):
    import qorgan.analytics.intake as intake

    assert client.post("/api/reports", json=_body("Сообщение: продиктуйте код из смс")).status_code == 201
    monkeypatch.setattr(intake, "report_to_incident", lambda *a, **k: (_ for _ in ()).throw(ValueError("no")))
    res = client.post("/api/admin/ingest", headers=as_analyst())
    assert res.status_code == 200
    assert res.json()["ingested"] == 0 and res.json()["skipped"] == 1


def test_the_client_address_behind_a_known_number_of_proxies():
    """The nearest proxy appends what it saw, so counting from the right cannot be spoofed; the
    leftmost entry is whatever the client sent."""
    from types import SimpleNamespace

    from qorgan.api_ratelimit import client_key

    def request(forwarded=None):
        headers = {"x-forwarded-for": forwarded} if forwarded else {}
        return SimpleNamespace(client=SimpleNamespace(host="10.0.0.1"), headers=headers)

    assert client_key(request("1.1.1.1"), hops=0) == "10.0.0.1", "no proxy declared: the header means nothing"
    assert client_key(request("1.1.1.1"), hops=1) == "1.1.1.1"
    assert client_key(request("6.6.6.6, 1.1.1.1"), hops=1) == "1.1.1.1", "a forged entry stays on the left"
    assert client_key(request("6.6.6.6, 1.1.1.1, 2.2.2.2"), hops=2) == "1.1.1.1"
    assert client_key(request(None), hops=1) == "10.0.0.1"
    assert client_key(request("1.1.1.1"), hops=3) == "1.1.1.1", "fewer entries than hops: the leftmost is the best known"
    assert client_key(SimpleNamespace(client=None, headers={}), hops=0) == "unknown"


def test_a_limiter_forgets_clients_it_no_longer_needs():
    """One new key per request (spoofed or just many visitors) must not grow memory for ever."""
    from qorgan.api_ratelimit import SlidingWindowLimiter

    now = [0.0]
    limiter = SlidingWindowLimiter(max_requests=2, window_seconds=60.0, clock=lambda: now[0], max_keys=100)
    for i in range(100):
        limiter.allow(f"client-{i}")
    now[0] = 61.0
    limiter.allow("one-more")
    assert len(limiter._hits) == 1
    # A full table of live clients is not dropped for a newcomer: the budget still holds.
    now[0] = 200.0
    for i in range(150):
        limiter.allow(f"live-{i}")
    assert limiter.allow("live-0") is True and limiter.allow("live-0") is False


def test_a_store_that_is_mostly_unreadable_stops_the_deletion_instead_of_being_wiped(paths):
    """A torn line is left out. A schema mismatch reads as every line corrupt, and rewriting without
    them would erase the whole store: that must fail, and the receipt keeps working."""
    _seed_analysis(paths)
    report = _ingested_report(paths)
    before = paths["incidents_path"].read_text(encoding="utf-8")
    paths["incidents_path"].write_text("".join('{"not": "an incident"}\n' for _ in range(5)) + before, encoding="utf-8")

    with pytest.raises(ValueError, match="corrupt"):
        forget_report(report.receipt_id, **paths)

    assert any(r.receipt_id == report.receipt_id for r in load_reports(paths["reports_path"]))
    assert '{"not": "an incident"}' in paths["incidents_path"].read_text(encoding="utf-8"), "nothing was rewritten"
