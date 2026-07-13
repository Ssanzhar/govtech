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
