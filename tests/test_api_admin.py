"""TDD tests for `qorgan.api_admin` — the analyst-dashboard HTTP API (Level 2).

Degrades to `available: false` whenever the precomputed analysis is missing or corrupt —
the demo must never 500 just because `scripts/demo_seed.py` +
`python -m qorgan.analytics.pipeline` haven't been run yet (CLAUDE.md SS9).
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from qorgan.analytics.pipeline import write_organizations_jsonl
from qorgan.api import app
from qorgan.data.incident_seed import write_incidents_jsonl
from support.numbers import hashed, prefix, stored_report

from qorgan.data.schema import Incident, Label, Organization, TacticTag


@pytest.fixture()
def client() -> TestClient:
    return TestClient(app)


def _incident(
    iid: str,
    *,
    tags: tuple[str, ...] = (),
    transcript: str = "это служба безопасности банка, продиктуйте код",
    number: str | None = None,
    ts=None,
) -> Incident:
    return Incident(
        id=iid,
        dialogue_id=iid,
        transcript=transcript,
        label=Label(risk=0.9, tactic_tags=tuple(TacticTag(id=t) for t in tags)),
        number_hash=hashed(number), number_prefix=prefix(number),
        timestamp=ts,
    )


def _seed_analysis(tmp_path, monkeypatch, *, organizations, incidents) -> None:
    monkeypatch.setenv("QORGAN_DATA_DIR", str(tmp_path))
    # The taxonomy stays the real one (needed for display names); only the incident/org
    # data files move under tmp_path.
    monkeypatch.setenv("QORGAN_TAXONOMY_PATH", "data/taxonomy/tactics.yaml")
    processed = tmp_path / "processed"
    write_organizations_jsonl(organizations, processed / "organizations.jsonl")
    write_incidents_jsonl(incidents, processed / "incidents.jsonl")


# --- degrade-ready ------------------------------------------------------------------------


def test_overview_degrades_when_no_analysis(client: TestClient, tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("QORGAN_DATA_DIR", str(tmp_path))

    body = client.get("/api/admin/overview").json()

    assert body["available"] is False
    assert body["kpis"] == {
        "incidents": 0,
        "organizations": 0,
        "novel_schemes": 0,
        "pending_reports": 0,
    }
    assert body["organizations"] == []


def test_overview_never_500s_on_corrupt_jsonl(client: TestClient, tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("QORGAN_DATA_DIR", str(tmp_path))
    processed = tmp_path / "processed"
    processed.mkdir(parents=True)
    (processed / "organizations.jsonl").write_text("{not valid json\n", encoding="utf-8")

    res = client.get("/api/admin/overview")

    assert res.status_code == 200
    assert res.json()["available"] is False


# --- overview -------------------------------------------------------------------------------


def test_overview_lists_organizations_with_tactic_derived_names(
    client: TestClient, tmp_path, monkeypatch
) -> None:
    incidents = [
        _incident("i1", tags=("impersonation_bank", "otp_request")),
        _incident("i2", tags=("impersonation_bank",)),
    ]
    org = Organization(
        id="org_0", members=("i1", "i2"), numbers=("+7 700 101 20 30",), priority=0.8
    )
    _seed_analysis(tmp_path, monkeypatch, organizations=[org], incidents=incidents)

    body = client.get("/api/admin/overview").json()

    assert body["available"] is True
    assert body["kpis"]["incidents"] == 2
    assert body["kpis"]["organizations"] == 1
    assert len(body["organizations"]) == 1
    out = body["organizations"][0]
    assert out["id"] == "org_0"
    assert out["name"] != "org_0"  # tactic-derived, never the raw cluster id
    assert out["incidents"] == 2
    assert out["numbers"] == ["+7 700 101 20 30"]  # full list — the queue is number-searchable
    assert out["is_novel"] is False


def test_overview_locale_changes_display_name(client: TestClient, tmp_path, monkeypatch) -> None:
    incidents = [_incident("i1", tags=("impersonation_bank",))]
    org = Organization(id="org_0", members=("i1",))
    _seed_analysis(tmp_path, monkeypatch, organizations=[org], incidents=incidents)

    ru = client.get("/api/admin/overview", params={"locale": "ru"}).json()
    kk = client.get("/api/admin/overview", params={"locale": "kk"}).json()

    assert ru["organizations"][0]["name"] != kk["organizations"][0]["name"]


# --- drilldown ------------------------------------------------------------------------------


def test_drilldown_returns_tactics_and_representative_script_and_excerpt(
    client: TestClient, tmp_path, monkeypatch
) -> None:
    long_transcript = "это служба безопасности банка " + "продиктуйте код из смс " * 20
    incidents = [
        _incident(
            "i1",
            tags=("impersonation_bank", "otp_request"),
            transcript=long_transcript,
            number="+7 700 101 20 30",
        ),
        _incident("i2", tags=("impersonation_bank",)),
    ]
    org = Organization(
        id="org_0",
        members=("i1", "i2"),
        numbers=("+7 700 101 20 30",),
        representative_script=long_transcript,
        priority=0.5,
    )
    _seed_analysis(tmp_path, monkeypatch, organizations=[org], incidents=incidents)

    body = client.get("/api/admin/organizations/org_0").json()

    assert body["id"] == "org_0"
    assert body["representative_script"] == long_transcript
    tactic_ids = {t["id"] for t in body["tactics"]}
    assert "impersonation_bank" in tactic_ids
    assert body["sample_incidents"]
    sample = body["sample_incidents"][0]
    assert long_transcript.startswith(sample["excerpt"])
    assert len(sample["excerpt"]) <= 200


def test_drilldown_unknown_org_returns_404(client: TestClient, tmp_path, monkeypatch) -> None:
    incidents = [_incident("i1", tags=("impersonation_bank",))]
    org = Organization(id="org_0", members=("i1",))
    _seed_analysis(tmp_path, monkeypatch, organizations=[org], incidents=incidents)

    res = client.get("/api/admin/organizations/nope")

    assert res.status_code == 404


def test_drilldown_404_when_analysis_unavailable(
    client: TestClient, tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("QORGAN_DATA_DIR", str(tmp_path))

    res = client.get("/api/admin/organizations/org_0")

    assert res.status_code == 404


# --- per-call on-demand model analysis ----------------------------------------------------------


SCAM_TRANSCRIPT = (
    "это служба безопасности банка, переведите деньги на безопасный счёт "
    "и продиктуйте код из смс"
)


def test_incident_analysis_returns_ranked_tags_and_verbatim_spans(
    client: TestClient, tmp_path, monkeypatch
) -> None:
    incidents = [_incident("i1", transcript=SCAM_TRANSCRIPT, number="+7 700 101 20 30")]
    org = Organization(id="org_0", members=("i1",))
    _seed_analysis(tmp_path, monkeypatch, organizations=[org], incidents=incidents)

    res = client.get("/api/admin/incidents/i1/analysis", params={"backend": "mock"})

    assert res.status_code == 200, res.text
    body = res.json()
    assert body["incident_id"] == "i1"
    assert body["backend"] == "mock"
    assert body["transcript"] == SCAM_TRANSCRIPT
    assert 0.0 <= body["risk"] <= 1.0
    assert isinstance(body["flagged"], bool)
    assert body["threshold"] > 0.0
    assert body["tags"], "the scam transcript must produce at least one tactic tag"
    weights = [t["weight"] for t in body["tags"]]
    assert weights == sorted(weights, reverse=True)  # ranked, heaviest first
    for tag in body["tags"]:
        assert tag["id"] and tag["name"]
    for span in body["spans"]:  # grounded: every span is verbatim at its offsets
        assert SCAM_TRANSCRIPT[span["start"] : span["end"]] == span["text"]
    assert body["reason"]


def test_incident_analysis_unknown_incident_is_404(
    client: TestClient, tmp_path, monkeypatch
) -> None:
    incidents = [_incident("i1", tags=("impersonation_bank",))]
    org = Organization(id="org_0", members=("i1",))
    _seed_analysis(tmp_path, monkeypatch, organizations=[org], incidents=incidents)

    res = client.get("/api/admin/incidents/nope/analysis", params={"backend": "mock"})

    assert res.status_code == 404


def test_incident_analysis_404_when_analysis_unavailable(
    client: TestClient, tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("QORGAN_DATA_DIR", str(tmp_path))

    res = client.get("/api/admin/incidents/i1/analysis", params={"backend": "mock"})

    assert res.status_code == 404


def test_incident_analysis_unknown_backend_is_422(
    client: TestClient, tmp_path, monkeypatch
) -> None:
    incidents = [_incident("i1", tags=("impersonation_bank",))]
    org = Organization(id="org_0", members=("i1",))
    _seed_analysis(tmp_path, monkeypatch, organizations=[org], incidents=incidents)

    res = client.get("/api/admin/incidents/i1/analysis", params={"backend": "not_a_backend"})

    assert res.status_code == 422


def test_drilldown_sample_incidents_carry_ids(
    client: TestClient, tmp_path, monkeypatch
) -> None:
    incidents = [_incident("i1", tags=("impersonation_bank",), number="+7 700 101 20 30")]
    org = Organization(id="org_0", members=("i1",))
    _seed_analysis(tmp_path, monkeypatch, organizations=[org], incidents=incidents)

    body = client.get("/api/admin/organizations/org_0").json()

    assert body["sample_incidents"][0]["id"] == "i1"


# --- ingest (citizen reports → refreshed analysis) ----------------------------------------------


class _FakeEmbedder:
    """Deterministic offline embedder (hash-seeded unit vectors), as in test_intake."""

    dim = 8

    def encode(self, texts, normalize_embeddings=True, convert_to_numpy=True):
        import numpy as np

        rows = []
        for text in texts:
            rng = np.random.default_rng(abs(hash(text)) % (2**32))
            vector = rng.normal(size=self.dim)
            rows.append(vector / np.linalg.norm(vector))
        return np.array(rows, dtype=np.float32)


def _write_report(tmp_path, *, number: str) -> None:
    from datetime import UTC, datetime

    draft = stored_report(number=number, timestamp=datetime(2026, 7, 17, 10, 0, tzinfo=UTC))
    path = tmp_path / "processed" / "citizen_reports.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(draft.model_dump_json() + "\n", encoding="utf-8")


def test_ingest_with_no_pending_reports_is_a_noop(
    client: TestClient, tmp_path, monkeypatch
) -> None:
    import qorgan.api_admin as api_admin

    incidents = [_incident("i1", tags=("impersonation_bank",), number="+7 700 101 20 30")]
    org = Organization(id="org_0", members=("i1",), numbers=("+7 700 101 20 30",))
    _seed_analysis(tmp_path, monkeypatch, organizations=[org], incidents=incidents)
    monkeypatch.setattr(api_admin, "_EMBEDDER_OVERRIDE", _FakeEmbedder())

    body = client.post("/api/admin/ingest").json()

    assert body["ingested"] == 0
    assert body["placements"] == []


def test_ingest_places_report_and_clears_pending_kpi(
    client: TestClient, tmp_path, monkeypatch
) -> None:
    import qorgan.api_admin as api_admin

    known = "+7 700 101 20 30"
    incidents = [_incident("i1", tags=("impersonation_bank",), number=known)]
    org = Organization(id="org_0", members=("i1",), numbers=(known,))
    _seed_analysis(tmp_path, monkeypatch, organizations=[org], incidents=incidents)
    _write_report(tmp_path, number=known)
    monkeypatch.setattr(api_admin, "_EMBEDDER_OVERRIDE", _FakeEmbedder())

    before = client.get("/api/admin/overview").json()
    body = client.post("/api/admin/ingest").json()
    after = client.get("/api/admin/overview").json()

    assert before["kpis"]["pending_reports"] == 1
    assert body["ingested"] == 1
    assert len(body["placements"]) == 1
    placement = body["placements"][0]
    assert placement["incident_id"].startswith("report-")
    assert placement["org_name"]  # display name resolved for the UI
    assert after["kpis"]["pending_reports"] == 0
    assert after["kpis"]["incidents"] == before["kpis"]["incidents"] + 1


def test_ingest_failure_is_503_not_500(client: TestClient, tmp_path, monkeypatch) -> None:
    import qorgan.api_admin as api_admin

    class _BrokenEmbedder:
        def encode(self, *args, **kwargs):
            raise RuntimeError("no embedding model here")

    known = "+7 700 101 20 30"
    incidents = [_incident("i1", tags=("impersonation_bank",), number=known)]
    org = Organization(id="org_0", members=("i1",), numbers=(known,))
    _seed_analysis(tmp_path, monkeypatch, organizations=[org], incidents=incidents)
    _write_report(tmp_path, number=known)
    monkeypatch.setattr(api_admin, "_EMBEDDER_OVERRIDE", _BrokenEmbedder())

    res = client.post("/api/admin/ingest")

    assert res.status_code == 503
