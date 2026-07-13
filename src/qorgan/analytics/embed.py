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
    return embed_texts([incident.transcript for incident in incidents], embedder=embedder, model_name=model_name)
