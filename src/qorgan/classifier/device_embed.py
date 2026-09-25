"""The `device` embed backend (PLAN_2026-09 B10, ADR D33): embeddings computed by the
citizen's runtime itself -- the site's embedding worker in headless Chromium (WASM) behind
the bridge `npm run device:serve` -- so the heads can be fitted and the tables computed on
what the browser actually produces. The server's native ONNX Runtime is a cosine-0.98 proxy
of that (ADR D32); training on the proxy left 2.5 % of gate decisions runtime-dependent.

The bridge is slow (~0.2-1 s per text) so every vector is cached on disk keyed by
(model, runtime build, exact text); a new browser build or model is a cache miss by
construction. Texts arrive already prefixed by `embed_texts`; the bridge's worker runs with an
empty prefix so the embedded string is byte-identical to the site's.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import urllib.error
import urllib.request
from collections.abc import Sequence
from pathlib import Path

import numpy as np

MAX_TEXTS_PER_REQUEST = 256  # the bridge's own cap
_REQUEST_TIMEOUT_S = 600.0    # a 256-text chunk on a loaded laptop
_INFO_TIMEOUT_S = 10.0
_START_HINT = "start the device bridge with `npm run device:serve` (needs `npx playwright install chromium` once)"


class EmbeddingCache:
    """sqlite-backed `(runtime_id, text) -> float32 vector` store; one file, append-only."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path))
        self._conn.execute("CREATE TABLE IF NOT EXISTS embeddings (key TEXT PRIMARY KEY, dim INTEGER NOT NULL, vec BLOB NOT NULL)")
        self._conn.execute("CREATE TABLE IF NOT EXISTS runtime (id INTEGER PRIMARY KEY CHECK (id = 1), info TEXT NOT NULL)")
        self._conn.commit()

    def remember_runtime(self, info: dict) -> None:
        """Keep the bridge's `/info` so a warm cache can be read with the bridge down."""
        self._conn.execute("INSERT OR REPLACE INTO runtime (id, info) VALUES (1, ?)", (json.dumps(info),))
        self._conn.commit()

    def remembered_runtime(self) -> dict | None:
        row = self._conn.execute("SELECT info FROM runtime WHERE id = 1").fetchone()
        return json.loads(row[0]) if row else None

    @staticmethod
    def key(runtime_id: str, text: str) -> str:
        return hashlib.sha256(f"{runtime_id}\n{text}".encode("utf-8")).hexdigest()

    def get(self, runtime_id: str, text: str) -> np.ndarray | None:
        row = self._conn.execute("SELECT dim, vec FROM embeddings WHERE key = ?", (self.key(runtime_id, text),)).fetchone()
        if row is None:
            return None
        return np.frombuffer(row[1], dtype=np.float32).reshape(row[0]).copy()

    def put(self, runtime_id: str, text: str, vector: np.ndarray) -> None:
        arr = np.ascontiguousarray(vector, dtype=np.float32)
        self._conn.execute("INSERT OR REPLACE INTO embeddings (key, dim, vec) VALUES (?, ?, ?)", (self.key(runtime_id, text), arr.shape[0], arr.tobytes()))

    def commit(self) -> None:
        self._conn.commit()


class DeviceEmbedder:
    """`encode()`-compatible embedder over the device bridge. `runtime_id` names the exact
    browser build the vectors come from (model | transformers.js | onnxruntime-web | device)."""

    def __init__(self, url: str, *, cache: EmbeddingCache) -> None:
        self._url = url.rstrip("/")
        self._cache = cache
        try:
            info = self._get_info()
            self.offline = False
            cache.remember_runtime(info)
        except ConnectionError:
            # A warm cache is usable without the bridge (evaluation reruns, tests): only a
            # miss needs the browser, and that miss raises with the same hint.
            info = cache.remembered_runtime()
            if info is None:
                raise
            self.offline = True
        self.runtime_id = "|".join((info["embedder"]["model_id"], info["transformers_js"], info["onnxruntime_web"], info["device"]))
        self.info = info

    def encode(self, texts: Sequence[str], normalize_embeddings: bool = True, convert_to_numpy: bool = True) -> np.ndarray:
        if not normalize_embeddings:
            raise ValueError("the device worker always L2-normalises; un-normalised embeddings are not available")
        text_list = list(texts)
        vectors: dict[str, np.ndarray] = {}
        missing: list[str] = []
        for text in dict.fromkeys(text_list):  # unique, ordered
            cached = self._cache.get(self.runtime_id, text)
            if cached is None:
                missing.append(text)
            else:
                vectors[text] = cached
        if missing and self.offline:
            raise ConnectionError(f"{len(missing)} text(s) are not in the device cache and the bridge at {self._url} is down; {_START_HINT}")
        for start in range(0, len(missing), MAX_TEXTS_PER_REQUEST):
            chunk = missing[start : start + MAX_TEXTS_PER_REQUEST]
            for text, row in zip(chunk, self._post_embed(chunk)):
                vector = np.asarray(row, dtype=np.float32)
                self._cache.put(self.runtime_id, text, vector)
                vectors[text] = vector
            self._cache.commit()
        return np.stack([vectors[t] for t in text_list]).astype(np.float32) if text_list else np.zeros((0, 0), dtype=np.float32)

    def _get_info(self) -> dict:
        try:
            with urllib.request.urlopen(f"{self._url}/info", timeout=_INFO_TIMEOUT_S) as response:
                return json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, OSError) as exc:
            raise ConnectionError(f"device bridge at {self._url} is not answering ({exc}); {_START_HINT}") from exc

    def _post_embed(self, texts: list[str]) -> list[list[float]]:
        payload = json.dumps({"texts": texts}, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(f"{self._url}/embed", data=payload, headers={"content-type": "application/json"}, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=_REQUEST_TIMEOUT_S) as response:
                rows = json.loads(response.read().decode("utf-8"))["rows"]
        except (urllib.error.URLError, OSError) as exc:
            raise ConnectionError(f"device bridge at {self._url} failed ({exc}); {_START_HINT}") from exc
        if len(rows) != len(texts):
            raise RuntimeError(f"device bridge returned {len(rows)} rows for {len(texts)} texts")
        return rows
