"""SMOKE tests for `qorgan.classifier.embed` — injected fake embedder, no download."""

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
