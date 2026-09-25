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

from pathlib import Path
from typing import Any

import numpy as np

from qorgan.config import get_config

# e5-family models expect a task prefix on each input; "query: " is the standard prefix.
_E5_PREFIX = "query: "

_MODEL_CACHE: dict[str, Any] = {}


# The int8 ONNX export the browser ships (PLAN_2026-09 A4/B3). Embedding with the same
# graph server-side makes on-device and server scores agree by construction.
ONNX_MAX_LENGTH = 512
# ONE text per run, deliberately: the int8 graph is dynamically quantised, and
# `DynamicQuantizeLinear` derives activation scales over the whole batched tensor (padding
# included), so a text's embedding would depend on what it was batched with (measured:
# cosine down to 0.98 between batch compositions). Batch size 1 makes every embedding a
# pure function of its text -- the property the server/device parity rests on.
ONNX_BATCH_SIZE = 1
ONNX_MODEL_FILE = "onnx/model_quantized.onnx"
ONNX_TOKENIZER_FILE = "tokenizer.json"
_PAD_TOKEN_ID = 1  # XLM-R / e5 vocabulary: <pad> = 1


class OnnxEmbedder:
    """`encode()`-compatible embedder over an ONNX Runtime session: tokenizer -> last
    hidden state -> attention-masked mean pooling -> optional L2 normalisation. The same
    recipe transformers.js applies in `site/core/embed-worker.js`."""

    def __init__(self, *, session: Any, tokenizer: Any) -> None:
        self._session = session
        self._tokenizer = tokenizer
        self._input_names = {i.name for i in session.get_inputs()}
        tokenizer.enable_truncation(ONNX_MAX_LENGTH)
        tokenizer.enable_padding(pad_id=_PAD_TOKEN_ID, pad_token="<pad>")

    @classmethod
    def from_dir(cls, model_dir: Path) -> "OnnxEmbedder":  # pragma: no cover - loads real files
        import onnxruntime as ort
        from tokenizers import Tokenizer

        session = ort.InferenceSession(str(model_dir / ONNX_MODEL_FILE), providers=["CPUExecutionProvider"])
        return cls(session=session, tokenizer=Tokenizer.from_file(str(model_dir / ONNX_TOKENIZER_FILE)))

    def encode(self, texts, normalize_embeddings: bool = True, convert_to_numpy: bool = True) -> np.ndarray:
        """Embed one text per graph run (see `ONNX_BATCH_SIZE`): deterministic per text and
        memory-bounded (a padded whole-corpus batch would need gigabytes of attention)."""
        text_list = list(texts)
        if not text_list:
            return np.zeros((0, 0), dtype=np.float32)
        chunks = [
            self._encode_batch(text_list[i : i + ONNX_BATCH_SIZE], normalize_embeddings)
            for i in range(0, len(text_list), ONNX_BATCH_SIZE)
        ]
        return np.vstack(chunks)

    def _encode_batch(self, texts: list[str], normalize_embeddings: bool) -> np.ndarray:
        encodings = self._tokenizer.encode_batch(texts)
        ids = np.asarray([e.ids for e in encodings], dtype=np.int64)
        mask = np.asarray([e.attention_mask for e in encodings], dtype=np.int64)
        feed = {"input_ids": ids, "attention_mask": mask}
        if "token_type_ids" in self._input_names:
            feed["token_type_ids"] = np.zeros_like(ids)
        hidden = self._session.run(None, feed)[0]
        weights = mask[..., None].astype(np.float32)
        pooled = (hidden * weights).sum(axis=1) / np.maximum(weights.sum(axis=1), 1.0)
        if normalize_embeddings:
            pooled = pooled / np.maximum(np.linalg.norm(pooled, axis=1, keepdims=True), 1e-12)
        return np.asarray(pooled, dtype=np.float32)


def get_embedder(model_name: str | None = None) -> Any:  # pragma: no cover - loads the real model
    """Lazily load and cache the configured embedder: the browser's own vectors through the
    device bridge (`"device"`, ADR D33), the int8 ONNX graph (`"onnx"`), else a
    SentenceTransformer by name."""
    cfg = get_config()
    name = model_name or cfg.embed_model_name
    if cfg.embed_backend == "device":
        key = f"device:{cfg.device_embed_url}"
    elif cfg.embed_backend == "onnx":
        key = f"onnx:{cfg.embed_onnx_dir}"
    else:
        key = name
    if key not in _MODEL_CACHE:
        if cfg.embed_backend == "device":
            from qorgan.classifier.device_embed import DeviceEmbedder, EmbeddingCache

            _MODEL_CACHE[key] = DeviceEmbedder(cfg.device_embed_url, cache=EmbeddingCache(cfg.device_embed_cache))
        elif cfg.embed_backend == "onnx":
            _MODEL_CACHE[key] = OnnxEmbedder.from_dir(cfg.embed_onnx_dir)
        else:
            from sentence_transformers import SentenceTransformer

            _MODEL_CACHE[key] = SentenceTransformer(name)
    return _MODEL_CACHE[key]


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
