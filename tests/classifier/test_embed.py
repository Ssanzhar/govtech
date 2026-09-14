"""SMOKE tests for `qorgan.classifier.embed` — injected fake embedder, no download."""

import pytest

from qorgan.classifier.embed import embed_texts


def test_embed_texts_returns_matrix_of_expected_shape(fake_embedder):
    vectors = embed_texts(["раз", "два", "три"], embedder=fake_embedder)
    assert vectors.shape == (3, fake_embedder.dim)


def test_embed_texts_applies_e5_query_prefix(fake_embedder):
    embed_texts(["привет мир"], embedder=fake_embedder)
    assert fake_embedder.seen[0].startswith("query: ")


def test_embed_texts_separates_scam_from_legit(fake_embedder):
    vectors = embed_texts(["Продиктуйте код из SMS", "Обычный разговор"], embedder=fake_embedder)
    # the fake maps scam-marked text to a different region -> vectors differ
    assert not (vectors[0] == vectors[1]).all()


# --- ONNX embedder (PLAN_2026-09 A4/B3): the server can embed with the exact int8 graph the
# browser ships, so on-device and server scores agree by construction. ---------------------


class _FakeSession:
    """Stands in for onnxruntime: returns a last_hidden_state whose mean over real tokens is
    the token-id pattern, so pooling/normalisation can be checked exactly."""

    def __init__(self, dim: int = 4) -> None:
        self.dim = dim
        self.calls: list[dict] = []

    def get_inputs(self):
        class _I:  # noqa: D401 - minimal stand-in
            def __init__(self, name):
                self.name = name

        return [_I("input_ids"), _I("attention_mask"), _I("token_type_ids")]

    def run(self, _outputs, feed):
        import numpy as np

        self.calls.append(feed)
        ids = feed["input_ids"]
        n, t = ids.shape
        hidden = np.zeros((n, t, self.dim), dtype=np.float32)
        hidden[:, :, 0] = ids  # first channel carries the token id
        hidden[:, :, 1] = 1.0
        return [hidden]


class _FakeTokenizer:
    def __init__(self):
        self.enabled = {}

    def enable_truncation(self, max_length):
        self.enabled["trunc"] = max_length

    def enable_padding(self, **kwargs):
        self.enabled["pad"] = kwargs

    def encode_batch(self, texts):
        class _Enc:
            def __init__(self, ids, mask):
                self.ids, self.attention_mask = ids, mask

        longest = max(len(t.split()) for t in texts)
        out = []
        for t in texts:
            ids = [len(w) for w in t.split()]
            mask = [1] * len(ids) + [0] * (longest - len(ids))
            out.append(_Enc(ids + [1] * (longest - len(ids)), mask))
        return out


def test_onnx_embedder_mean_pools_over_real_tokens_and_normalises():
    import numpy as np

    from qorgan.classifier.embed import OnnxEmbedder

    session, tokenizer = _FakeSession(), _FakeTokenizer()
    embedder = OnnxEmbedder(session=session, tokenizer=tokenizer)
    vectors = embedder.encode(["query: ab c", "query: abcd"], normalize_embeddings=True, convert_to_numpy=True)

    assert vectors.shape == (2, 4) and vectors.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-6)
    # row 0: tokens "query:"(6), "ab"(2), "c"(1) -> mean id 3, channel1 = 1 -> direction (3, 1, 0, 0)
    expected = np.array([3.0, 1.0, 0.0, 0.0]); expected /= np.linalg.norm(expected)
    np.testing.assert_allclose(vectors[0], expected, atol=1e-6)
    feed = session.calls[0]
    assert set(feed) == {"input_ids", "attention_mask", "token_type_ids"}
    assert feed["token_type_ids"].sum() == 0 and feed["input_ids"].dtype == np.int64


def test_onnx_embedder_unnormalised_and_truncation_setup():
    from qorgan.classifier.embed import ONNX_MAX_LENGTH, OnnxEmbedder

    tokenizer = _FakeTokenizer()
    embedder = OnnxEmbedder(session=_FakeSession(), tokenizer=tokenizer)
    assert tokenizer.enabled["trunc"] == ONNX_MAX_LENGTH == 512
    raw = embedder.encode(["query: ab"], normalize_embeddings=False, convert_to_numpy=True)
    assert abs(float(raw[0][1]) - 1.0) < 1e-6


def test_embed_backend_config_defaults_and_validation():
    from qorgan.config import ConfigError, load_config

    assert load_config({}).embed_backend == "onnx"
    cfg = load_config({"QORGAN_EMBED_BACKEND": "onnx", "QORGAN_EMBED_ONNX_DIR": "/tmp/m"})
    assert cfg.embed_backend == "onnx" and str(cfg.embed_onnx_dir) == "/tmp/m"
    with pytest.raises(ConfigError):
        load_config({"QORGAN_EMBED_BACKEND": "tflite"})


def test_onnx_embedder_runs_one_text_per_graph_call():
    from qorgan.classifier.embed import ONNX_BATCH_SIZE, OnnxEmbedder

    session = _FakeSession()
    embedder = OnnxEmbedder(session=session, tokenizer=_FakeTokenizer())
    out = embedder.encode([f"query: t{i}" for i in range(ONNX_BATCH_SIZE * 2 + 3)])
    assert out.shape[0] == ONNX_BATCH_SIZE * 2 + 3
    assert len(session.calls) == ONNX_BATCH_SIZE * 2 + 3 and max(len(c["input_ids"]) for c in session.calls) == ONNX_BATCH_SIZE
