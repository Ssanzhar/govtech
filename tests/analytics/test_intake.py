"""TDD tests for `qorgan.analytics.intake` — citizen reports → incidents → re-analysis.

Fully offline: a local deterministic fake embedder, tmp_path jsonl fixtures. The key
behaviors under test: idempotent deterministic ids, number-graph placement (a report
whose caller number matches a known organization joins it), unknown numbers forming new
organizations, and the embeddings cache growing by exactly the new rows.
"""

from datetime import UTC, datetime

import numpy as np
import pytest

from qorgan.analytics.intake import (
    IngestSummary,
    ingest_pending,
    pending_reports,
    report_incident_id,
)
from qorgan.analytics.pipeline import (
    analyze_incidents,
    load_embeddings_npz,
    load_organizations_jsonl,
    save_embeddings_npz,
    write_organizations_jsonl,
)
from qorgan.data.incident_seed import load_incidents_jsonl, write_incidents_jsonl
from support.numbers import hashed, prefix, stored_report

from qorgan.data.schema import Incident, Label

KNOWN_NUMBER = "+7 700 101 20 30"
OTHER_NUMBER = "+7 701 202 30 40"


class _FakeEmbedder:
    """Deterministic offline embedder (hash-seeded unit vectors)."""

    dim = 8

    def encode(self, texts, normalize_embeddings=True, convert_to_numpy=True):
        rows = []
        for text in texts:
            rng = np.random.default_rng(abs(hash(text)) % (2**32))
            vector = rng.normal(size=self.dim)
            rows.append(vector / np.linalg.norm(vector))
        return np.array(rows, dtype=np.float32)


def _incident(iid, number):
    # Naive timestamp: the seeded incident store's convention. Live report drafts are
    # tz-aware UTC -- intake must normalize the mix (regression: ranking crashed on it).
    return Incident(
        id=iid,
        dialogue_id=iid,
        transcript="это служба безопасности банка продиктуйте код",
        label=Label(risk=0.9),
        number_hash=hashed(number), number_prefix=prefix(number),
        timestamp=datetime(2026, 7, 10, 12, 0),
    )


def _draft(number=KNOWN_NUMBER, transcript="алло переведите деньги на безопасный счёт"):
    # Reports reach intake already minimised (number hashed, transcript scrubbed).
    return stored_report(number=number, transcript=transcript, receipt_id="c" * 24)


@pytest.fixture
def seeded(tmp_path):
    """Two seeded organizations (by number), analysis + embedding cache on disk."""
    embedder = _FakeEmbedder()
    incidents = [
        _incident("a0", KNOWN_NUMBER),
        _incident("a1", KNOWN_NUMBER),
        _incident("b0", OTHER_NUMBER),
    ]
    paths = {
        "reports": tmp_path / "citizen_reports.jsonl",
        "incidents": tmp_path / "incidents.jsonl",
        "organizations": tmp_path / "organizations.jsonl",
        "embeddings": tmp_path / "incident_embeddings.npz",
    }
    write_incidents_jsonl(incidents, paths["incidents"])
    from qorgan.analytics.embed import embed_incidents

    matrix = embed_incidents(incidents, embedder=embedder)
    organizations = analyze_incidents(incidents, embeddings=matrix)
    write_organizations_jsonl(organizations, paths["organizations"])
    save_embeddings_npz([i.id for i in incidents], matrix, paths["embeddings"])
    return paths, embedder


def _write_reports(path, drafts):
    path.write_text("\n".join(d.model_dump_json() for d in drafts) + "\n", encoding="utf-8")


def _ingest(paths, embedder):
    return ingest_pending(
        reports_path=paths["reports"],
        incidents_path=paths["incidents"],
        organizations_path=paths["organizations"],
        embeddings_path=paths["embeddings"],
        embedder=embedder,
        now=datetime(2026, 7, 15, 12, 0),
    )


# --- ids & pending ----------------------------------------------------------------------------


def test_report_incident_id_is_deterministic_and_content_addressed():
    draft = _draft()
    assert report_incident_id(draft) == report_incident_id(draft)
    assert report_incident_id(draft).startswith("report-")
    assert report_incident_id(draft) != report_incident_id(_draft(transcript="другой текст"))


def test_pending_reports_excludes_already_ingested(seeded):
    paths, _ = seeded
    draft = _draft()
    _write_reports(paths["reports"], [draft])
    ingested = _incident(report_incident_id(draft), KNOWN_NUMBER)

    existing = load_incidents_jsonl(paths["incidents"])
    assert len(pending_reports(paths["reports"], existing)) == 1
    assert pending_reports(paths["reports"], [*existing, ingested]) == []


def test_missing_reports_file_yields_empty_summary(seeded):
    paths, embedder = seeded

    summary = _ingest(paths, embedder)

    assert summary == IngestSummary(ingested=0, placements=(), organizations_total=2)


# --- placement ---------------------------------------------------------------------------------


def test_report_with_known_number_joins_that_organization(seeded):
    paths, embedder = seeded
    draft = _draft(number=KNOWN_NUMBER)
    _write_reports(paths["reports"], [draft])

    summary = _ingest(paths, embedder)

    assert summary.ingested == 1
    placement = summary.placements[0]
    organizations = load_organizations_jsonl(paths["organizations"])
    org = next(o for o in organizations if o.id == placement.org_id)
    assert placement.incident_id in org.members
    assert {"a0", "a1"} <= set(org.members)  # joined the seeded family, not a new org
    assert summary.organizations_total == 2


def test_report_with_unknown_number_forms_new_organization(seeded):
    paths, embedder = seeded
    draft = _draft(number="+7 777 000 00 99")
    _write_reports(paths["reports"], [draft])

    summary = _ingest(paths, embedder)

    organizations = load_organizations_jsonl(paths["organizations"])
    org = next(o for o in organizations if summary.placements[0].incident_id in o.members)
    assert set(org.members) == {summary.placements[0].incident_id}
    assert summary.organizations_total == 3


# --- idempotency & cache -----------------------------------------------------------------------


def test_ingest_is_idempotent(seeded):
    paths, embedder = seeded
    _write_reports(paths["reports"], [_draft()])

    first = _ingest(paths, embedder)
    second = _ingest(paths, embedder)

    assert first.ingested == 1
    assert second.ingested == 0
    assert len(load_incidents_jsonl(paths["incidents"])) == 4  # 3 seeded + 1, not 5


def test_duplicate_drafts_in_one_batch_ingest_once(seeded):
    paths, embedder = seeded
    _write_reports(paths["reports"], [_draft(), _draft()])

    summary = _ingest(paths, embedder)

    assert summary.ingested == 1


def test_embeddings_cache_grows_by_exactly_the_new_rows(seeded):
    paths, embedder = seeded
    _write_reports(paths["reports"], [_draft()])

    _ingest(paths, embedder)

    ids, matrix = load_embeddings_npz(paths["embeddings"])
    assert len(ids) == 4
    assert matrix.shape == (4, _FakeEmbedder.dim)
    assert ids[-1].startswith("report-")


def test_missing_embeddings_cache_falls_back_to_full_embed(seeded):
    paths, embedder = seeded
    paths["embeddings"].unlink()
    _write_reports(paths["reports"], [_draft()])

    summary = _ingest(paths, embedder)

    assert summary.ingested == 1
    ids, matrix = load_embeddings_npz(paths["embeddings"])  # cache rebuilt in full
    assert len(ids) == 4 and matrix.shape[0] == 4


def test_summary_is_frozen(seeded):
    paths, embedder = seeded
    summary = _ingest(paths, embedder)
    with pytest.raises(Exception):
        summary.ingested = 99  # type: ignore[misc]


# --- forget_report: a citizen's deletion reaches the analysis too (PLAN_2026-09 C3) ----------


def _forget(paths, receipt_id):
    from qorgan.analytics.intake import forget_report

    return forget_report(
        receipt_id,
        reports_path=paths["reports"],
        incidents_path=paths["incidents"],
        organizations_path=paths["organizations"],
        embeddings_path=paths["embeddings"],
        now=datetime(2026, 7, 15, 12, 0),
    )


def test_forget_removes_an_ingested_report_from_reports_incidents_and_cache(seeded):
    from qorgan.analytics.pipeline import load_embeddings_npz, load_organizations_jsonl
    from qorgan.data.incident_seed import load_incidents_jsonl
    from qorgan.reports.store import load_reports

    paths, embedder = seeded
    draft = _draft()
    _write_reports(paths["reports"], [draft])
    _ingest(paths, embedder)
    incident_id = report_incident_id(draft)
    assert incident_id in {i.id for i in load_incidents_jsonl(paths["incidents"])}

    summary = _forget(paths, draft.receipt_id)

    assert summary is not None and summary.incident_removed
    assert load_reports(paths["reports"]) == []
    remaining = load_incidents_jsonl(paths["incidents"])
    assert incident_id not in {i.id for i in remaining}
    cached_ids, matrix = load_embeddings_npz(paths["embeddings"])
    assert cached_ids == [i.id for i in remaining] and len(matrix) == len(remaining)
    members = {m for org in load_organizations_jsonl(paths["organizations"]) for m in org.members}
    assert incident_id not in members and members == {i.id for i in remaining}


def test_forget_a_pending_report_only_touches_the_reports_file(seeded):
    from qorgan.data.incident_seed import load_incidents_jsonl

    paths, _ = seeded
    draft = _draft()
    _write_reports(paths["reports"], [draft])
    before = load_incidents_jsonl(paths["incidents"])

    summary = _forget(paths, draft.receipt_id)

    assert summary is not None and not summary.incident_removed
    assert load_incidents_jsonl(paths["incidents"]) == before


def test_forget_unknown_receipt_returns_none(seeded):
    paths, _ = seeded
    assert _forget(paths, "f" * 24) is None


# --- signals-only partner reports (PLAN C9) -----------------------------------------------------


def _signals_only_draft(number, reference="CASE-1"):
    from qorgan.reports.store import prepare_report
    from support.numbers import TEST_HMAC_KEY

    return prepare_report(
        transcript="", phone_number=number, flagged_phrases=(), tactic_ids=("otp_request", "safe_account"),
        timestamp=datetime(2026, 9, 17, 10, 0, tzinfo=UTC), risk_score=100.0, hmac_key=TEST_HMAC_KEY,
        source="partner", consent_basis="customer_consent", partner_id="bank_a", partner_reference=reference,
    )


def test_signals_only_report_with_known_number_joins_that_organization(seeded):
    """No transcript to embed -- the number graph alone places it (a zero embedding row)."""
    paths, embedder = seeded
    _write_reports(paths["reports"], [_signals_only_draft(KNOWN_NUMBER)])

    summary = _ingest(paths, embedder)

    assert summary.ingested == 1
    placement = summary.placements[0]
    org = next(o for o in load_organizations_jsonl(paths["organizations"]) if o.id == placement.org_id)
    assert {"a0", "a1", placement.incident_id} <= set(org.members) and not org.is_novel
    ids, matrix = load_embeddings_npz(paths["embeddings"])
    assert placement.incident_id in ids
    assert not matrix[ids.index(placement.incident_id)].any()  # zero row, nothing fabricated
    incident = next(i for i in load_incidents_jsonl(paths["incidents"]) if i.id == placement.incident_id)
    assert incident.transcript == "" and {t.id for t in incident.label.tactic_tags} == {"otp_request", "safe_account"}


def test_signals_only_report_with_unknown_number_is_a_singleton_but_never_novel(seeded):
    paths, embedder = seeded
    _write_reports(paths["reports"], [_signals_only_draft("+7 777 000 00 99")])

    summary = _ingest(paths, embedder)

    org = next(o for o in load_organizations_jsonl(paths["organizations"]) if summary.placements[0].incident_id in o.members)
    assert set(org.members) == {summary.placements[0].incident_id}
    assert org.is_novel is False  # no text evidence: a zero vector is not "far from everything"
    assert org.representative_script is None


def test_signals_only_reports_are_pending_like_any_other(seeded):
    paths, _ = seeded
    _write_reports(paths["reports"], [_signals_only_draft(KNOWN_NUMBER)])
    assert len(pending_reports(paths["reports"], load_incidents_jsonl(paths["incidents"]))) == 1
