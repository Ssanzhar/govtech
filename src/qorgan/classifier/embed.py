"""Sentence-embedding feature extraction (multilingual-e5 / BGE-M3) for the `linear`
scam classifier -- and, later, Level-2 clustering.

We use a strong *frozen* multilingual embedder (pretrained on billions of tokens, incl.
Kazakh/Russian) as the feature extractor and learn only a light classifier on top
(`linear_train.py`). This keeps the transformer's context understanding while staying
trainable on a few hundred examples -- the opposite of the end-to-end XLM-R fine-tune that
collapsed (`docs/eval_report.md`).

The embedder is injectable so tests run offline with a fake; production lazily loads and
caches the real SentenceTransformer.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from qorgan.config import get_config

# e5-family models expect a task prefix on each input; "query: " is the standard prefix.
_E5_PREFIX = "query: "

_MODEL_CACHE: dict[str, Any] = {}


def get_embedder(model_name: str | None = None) -> Any:  # pragma: no cover - loads the real model
    """Lazily load and cache a SentenceTransformer by name (defaults to `config.embed_model_name`)."""
    from sentence_transformers import SentenceTransformer

    name = model_name or get_config().embed_model_name
    if name not in _MODEL_CACHE:
        _MODEL_CACHE[name] = SentenceTransformer(name)
    return _MODEL_CACHE[name]


def embed_texts(
    texts: list[str],
    *,
    embedder: Any = None,
    model_name: str | None = None,
    prefix: str = _E5_PREFIX,
    normalize: bool = True,
) -> np.ndarray:
    """Embed `texts` into a `(n, dim)` float32 matrix, applying the e5 task prefix.

    `embedder` (any object with a compatible `.encode`) is injected in tests; otherwise the
    configured model is loaded lazily. Embeddings are L2-normalized by default so a linear
    classifier sees unit vectors.
    """
    active = embedder if embedder is not None else get_embedder(model_name)
    prefixed = [f"{prefix}{text}" for text in texts]
    vectors = active.encode(prefixed, normalize_embeddings=normalize, convert_to_numpy=True)
    return np.asarray(vectors, dtype=np.float32)
