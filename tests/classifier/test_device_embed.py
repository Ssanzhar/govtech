"""TDD tests for `qorgan.classifier.device_embed` (PLAN B10, ADR D33): an `encode()`-compatible
embedder that asks the device bridge (`npm run device:serve`) for the browser's own embeddings
and caches them on disk keyed by (model, runtime, text)."""

from __future__ import annotations

import hashlib
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np
import pytest

from qorgan.classifier.device_embed import DeviceEmbedder, EmbeddingCache, MAX_TEXTS_PER_REQUEST

DIM = 8
INFO = {"embedder": {"model_id": "Xenova/test-model", "prefix": ""}, "transformers_js": "3.8.1", "onnxruntime_web": "1.22.0-dev", "device": "wasm"}


def _vector(text: str) -> list[float]:
    seed = int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)
    v = np.random.default_rng(seed).normal(size=DIM)
    return (v / np.linalg.norm(v)).tolist()


class _Bridge(BaseHTTPRequestHandler):
    requests: list[list[str]] = []

    def do_GET(self):
        self._send(200, INFO if self.path == "/info" else {"error": "not found"})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["content-length"])))
        type(self).requests.append(body["texts"])
        self._send(200, {"rows": [_vector(t) for t in body["texts"]]})

    def _send(self, status, payload):
        data = json.dumps(payload).encode()
        self.send_response(status); self.send_header("content-type", "application/json"); self.send_header("content-length", str(len(data))); self.end_headers(); self.wfile.write(data)

    def log_message(self, *_):
        pass


@pytest.fixture
def bridge():
    _Bridge.requests = []
    server = HTTPServer(("127.0.0.1", 0), _Bridge)
    thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()


def test_encode_returns_the_bridge_rows_and_reports_the_runtime(bridge, tmp_path):
    emb = DeviceEmbedder(bridge, cache=EmbeddingCache(tmp_path / "cache.sqlite"))
    rows = emb.encode(["query: a", "query: b"], normalize_embeddings=True, convert_to_numpy=True)
    assert rows.shape == (2, DIM) and rows.dtype == np.float32
    assert np.allclose(rows[0], _vector("query: a"), atol=1e-6)
    assert emb.runtime_id == "Xenova/test-model|3.8.1|1.22.0-dev|wasm"


def test_cache_serves_repeats_without_asking_the_bridge(bridge, tmp_path):
    cache = EmbeddingCache(tmp_path / "cache.sqlite")
    emb = DeviceEmbedder(bridge, cache=cache)
    first = emb.encode(["query: a", "query: b"])
    again = DeviceEmbedder(bridge, cache=EmbeddingCache(tmp_path / "cache.sqlite")).encode(["query: b", "query: a", "query: c"])
    assert np.array_equal(again[1], first[0]) and np.array_equal(again[0], first[1])
    assert _Bridge.requests == [["query: a", "query: b"], ["query: c"]]  # only the miss went over the wire


def test_cache_key_includes_the_runtime_so_a_new_browser_build_is_a_miss(bridge, tmp_path):
    cache = EmbeddingCache(tmp_path / "cache.sqlite")
    cache.put("other-runtime", "query: a", np.ones(DIM, dtype=np.float32))
    rows = DeviceEmbedder(bridge, cache=cache).encode(["query: a"])
    assert not np.array_equal(rows[0], np.ones(DIM)) and _Bridge.requests == [["query: a"]]


def test_requests_are_chunked(bridge, tmp_path):
    texts = [f"query: t{i}" for i in range(MAX_TEXTS_PER_REQUEST + 3)]
    DeviceEmbedder(bridge, cache=EmbeddingCache(tmp_path / "c.sqlite")).encode(texts)
    assert [len(r) for r in _Bridge.requests] == [MAX_TEXTS_PER_REQUEST, 3]


def test_a_dead_bridge_with_a_cold_cache_is_a_clear_error(tmp_path):
    with pytest.raises(ConnectionError, match="npm run device:serve"):
        DeviceEmbedder("http://127.0.0.1:1", cache=EmbeddingCache(tmp_path / "c.sqlite"))


def test_normalize_false_is_refused(bridge, tmp_path):
    emb = DeviceEmbedder(bridge, cache=EmbeddingCache(tmp_path / "c.sqlite"))
    with pytest.raises(ValueError, match="normalis"):
        emb.encode(["query: a"], normalize_embeddings=False)


def test_config_accepts_the_device_backend_and_its_knobs():
    from qorgan.config import load_config

    cfg = load_config({"QORGAN_EMBED_BACKEND": "device", "QORGAN_DEVICE_EMBED_URL": "http://127.0.0.1:9999"})
    assert cfg.embed_backend == "device" and cfg.device_embed_url == "http://127.0.0.1:9999"
    assert cfg.device_embed_cache.name == "device_embeddings.sqlite"


def test_a_warm_cache_works_without_the_bridge_and_a_miss_says_how_to_start_it(bridge, tmp_path):
    cache_path = tmp_path / "cache.sqlite"
    online = DeviceEmbedder(bridge, cache=EmbeddingCache(cache_path))
    warm = online.encode(["query: a"])
    offline = DeviceEmbedder("http://127.0.0.1:1", cache=EmbeddingCache(cache_path))  # bridge down, cache remembers the runtime
    assert offline.offline and offline.runtime_id == online.runtime_id
    assert np.array_equal(offline.encode(["query: a"])[0], warm[0])
    with pytest.raises(ConnectionError, match="npm run device:serve"):
        offline.encode(["query: never seen"])
