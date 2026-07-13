"""In-memory cosine store over incident embeddings for L2 drill-down / retrieval (D5-2).

A light nearest-neighbor index keyed to incident ids -- enough for "show me incidents
similar to this one" in the analyst panel without a database. Implemented with numpy cosine
similarity (over ~500 incidents it is instant) rather than FAISS, which shares an OpenMP
runtime with torch and segfaults when both are loaded in one process on macOS.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

_EPS = 1e-12


class IncidentStore:
    """Cosine nearest-neighbor index mapping vectors back to incident ids."""

    def __init__(self, incident_ids: Sequence[str], embeddings: np.ndarray) -> None:
        if len(incident_ids) != len(embeddings):
            raise ValueError(
                f"incident_ids and embeddings length mismatch: {len(incident_ids)} vs {len(embeddings)}"
            )
        self.incident_ids = list(incident_ids)
        vectors = np.asarray(embeddings, dtype=np.float32)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True) if len(vectors) else vectors
        self._vectors = vectors / np.clip(norms, _EPS, None) if len(vectors) else vectors

    def nearest(self, query: np.ndarray, k: int = 5) -> list[tuple[str, float]]:
        """Return the `k` most similar incidents to `query` as `(incident_id, score)` pairs."""
        if not self.incident_ids:
            return []
        vector = np.asarray(query, dtype=np.float32).reshape(-1)
        vector = vector / max(float(np.linalg.norm(vector)), _EPS)
        similarities = self._vectors @ vector
        top = np.argsort(-similarities)[: min(k, len(self.incident_ids))]
        return [(self.incident_ids[i], float(similarities[i])) for i in top]
