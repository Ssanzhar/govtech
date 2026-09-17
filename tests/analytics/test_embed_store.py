"""SMOKE tests for `qorgan.analytics.embed` + `store` — offline, no model download."""

import numpy as np

from qorgan.analytics.embed import embed_incidents
from qorgan.analytics.store import IncidentStore
from qorgan.data.schema import Incident, Label


class _FakeEmbedder:
    def encode(self, texts, normalize_embeddings=True, convert_to_numpy=True):
        return np.array([[float(len(t)), 1.0, 0.0] for t in texts], dtype=np.float32)


def _incident(iid, text):
    return Incident(id=iid, dialogue_id=iid, transcript=text, label=Label(risk=0.9))


def test_embed_incidents_returns_matrix():
    incidents = [_incident("a", "раз"), _incident("b", "два слова здесь")]
    vecs = embed_incidents(incidents, embedder=_FakeEmbedder())
    assert vecs.shape == (2, 3)


def test_store_nearest_returns_self_as_top_match():
    ids = ["a", "b", "c"]
    embeddings = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    store = IncidentStore(ids, embeddings)
    hits = store.nearest(np.array([1.0, 0.0, 0.0], dtype=np.float32), k=2)
    assert hits[0][0] == "a"
    assert hits[0][1] > 0.99


def test_store_length_mismatch_raises():
    import pytest

    with pytest.raises(ValueError):
        IncidentStore(["a"], np.zeros((2, 3), dtype=np.float32))


def test_embed_incidents_gives_transcript_less_incidents_a_zero_row():
    """PLAN C9: signals-only incidents must not be embedded as an empty string (which
    would give every one of them the same constant vector and cluster them together)."""
    import numpy as np

    from qorgan.analytics.embed import embed_incidents
    from qorgan.data.schema import Incident, Label

    class _Embedder:
        def encode(self, texts, normalize_embeddings=True, convert_to_numpy=True):
            assert all(t.strip() != "query:" for t in texts), "blank transcript reached the embedder"
            return np.ones((len(texts), 4), dtype=np.float32) / 2.0

    incidents = [
        Incident(id="t1", dialogue_id="t1", transcript="алло это банк", label=Label(risk=0.9)),
        Incident(id="s1", dialogue_id="s1", transcript="", label=Label(risk=0.9)),
        Incident(id="t2", dialogue_id="t2", transcript="назовите код", label=Label(risk=0.9)),
    ]
    matrix = embed_incidents(incidents, embedder=_Embedder())
    assert matrix.shape == (3, 4)
    assert matrix[0].any() and matrix[2].any() and not matrix[1].any()
    assert embed_incidents([incidents[1]], embedder=_Embedder()).shape == (1, 0) or not embed_incidents([incidents[1]], embedder=_Embedder()).any()
