"""SMOKE test for `integrated_gradient_spans` — tiny model + fake tokenizer, offline."""

import pytest

torch = pytest.importorskip("torch")

from qorgan.classifier.attribution import integrated_gradient_spans
from qorgan.classifier.model import ScamClassifierModel
from qorgan.data.schema import Span

_TRANSCRIPT = "Продиктуйте код из SMS и переведите деньги на безопасный счёт"


def test_ig_returns_verbatim_spans(tiny_encoder, fake_tokenizer):
    torch.manual_seed(0)
    model = ScamClassifierModel(tiny_encoder, num_tactics=3)

    spans = integrated_gradient_spans(
        model,
        fake_tokenizer,
        _TRANSCRIPT,
        device=torch.device("cpu"),
        max_length=32,
        top_k=4,
        n_steps=8,
        # very low threshold forces at least some tokens through, regardless of the
        # random tiny model's attribution signs
        min_score=-1e9,
    )

    assert isinstance(spans, tuple)
    assert len(spans) >= 1
    for span in spans:
        assert isinstance(span, Span)
        assert _TRANSCRIPT[span.start : span.end] == span.text  # verbatim


def test_ig_default_threshold_returns_valid_spans(tiny_encoder, fake_tokenizer):
    torch.manual_seed(0)
    model = ScamClassifierModel(tiny_encoder, num_tactics=3)

    spans = integrated_gradient_spans(
        model, fake_tokenizer, _TRANSCRIPT, device=torch.device("cpu"), max_length=32, top_k=4, n_steps=8
    )

    assert isinstance(spans, tuple)
    assert len(spans) <= 4
    for span in spans:
        assert _TRANSCRIPT[span.start : span.end] == span.text
