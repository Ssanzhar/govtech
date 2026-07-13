"""SMOKE tests for `qorgan.classifier.model` — tiny random-config encoder, no download."""

import pytest
import torch

from qorgan.classifier.model import ScamClassifierModel


def test_forward_returns_risk_and_tactic_logits_of_expected_shape(tiny_encoder):
    model = ScamClassifierModel(tiny_encoder, num_tactics=5)
    input_ids = torch.randint(0, 200, (2, 8))
    attention_mask = torch.ones(2, 8, dtype=torch.long)

    out = model(input_ids=input_ids, attention_mask=attention_mask)

    assert out["risk_logit"].shape == (2,)
    assert out["tactic_logits"].shape == (2, 5)


def test_forward_accepts_inputs_embeds_for_attribution(tiny_encoder):
    model = ScamClassifierModel(tiny_encoder, num_tactics=3)
    embeds = torch.randn(1, 6, tiny_encoder.config.hidden_size, requires_grad=True)
    attention_mask = torch.ones(1, 6, dtype=torch.long)

    out = model(inputs_embeds=embeds, attention_mask=attention_mask)

    assert out["risk_logit"].shape == (1,)
    # gradient flows back to the embeddings (Captum needs this).
    out["risk_logit"].sum().backward()
    assert embeds.grad is not None


def test_num_tactics_must_be_positive(tiny_encoder):
    with pytest.raises(ValueError):
        ScamClassifierModel(tiny_encoder, num_tactics=0)
