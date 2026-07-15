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
from qorgan.data.schema import Incident, Label
from qorgan.live.summary import ReportDraft

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
        phone_number=number,
        timestamp=datetime(2026, 7, 10, 12, 0),
    )


def _draft(number=KNOWN_NUMBER, transcript="алло переведите деньги на безопасный счёт"):
    return ReportDraft(
        phone_number=number,
        transcript=transcript,
        flagged_phrases=("переведите деньги на безопасный счёт",),
        tactic_ids=("safe_account",),
        timestamp=datetime(2026, 7, 15, 10, 0, tzinfo=UTC),
        risk_score=84.0,
    )


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
