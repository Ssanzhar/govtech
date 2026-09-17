"""Embed incident transcripts for Level-2 clustering (reuses the classifier's frozen e5
embedder -- same multilingual model, one download, shared across L1 and L2)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from qorgan.classifier.embed import embed_texts
from qorgan.data.schema import Incident


def embed_incidents(
    incidents: Sequence[Incident], *, embedder: Any = None, model_name: str | None = None
) -> np.ndarray:
    """Embed each incident's transcript into a `(n, dim)` matrix (L2-normalized)."""
    return embed_transcripts([incident.transcript for incident in incidents], embedder=embedder, model_name=model_name)


def embed_transcripts(texts: Sequence[str], *, embedder: Any = None, model_name: str | None = None) -> np.ndarray:
    """`embed_texts` with a **zero row** for every blank transcript (signals-only partner
    reports, PLAN C9): embedding "" would give all of them one constant vector and cluster
    them together; a zero vector carries no text evidence and is ignored by novelty."""
    present = [i for i, text in enumerate(texts) if text.strip()]
    if not present:
        return np.zeros((len(texts), 0), dtype=np.float32)
    vectors = embed_texts([texts[i] for i in present], embedder=embedder, model_name=model_name)
    matrix = np.zeros((len(texts), vectors.shape[1]), dtype=np.float32)
    matrix[present] = vectors
    return matrix
