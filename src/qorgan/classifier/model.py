"""XLM-R multi-task scam classifier (D3-1): a shared XLM-RoBERTa encoder feeding a binary
**risk head** (is this a scam?) and a multi-label **tactic head** (which tactics), per
CLAUDE.md §4.

Kept deliberately thin and injectable: `ScamClassifierModel` wraps *any* pre-built encoder,
so tests construct a tiny random-config XLM-R (no 1GB download) while `build_model` loads the
real `xlm-roberta-base`. Only imported on the `xlmr` code path, so torch never loads for the
`llm`/`mock` backends.
"""

from __future__ import annotations

import torch
from torch import nn

_DROPOUT = 0.1
# Guard for the mean-pool denominator (a fully-masked row should never divide by zero).
_MIN_TOKENS = 1e-9


class ScamClassifierModel(nn.Module):
    """Encoder + risk head (1 logit) + tactic head (`num_tactics` logits)."""

    def __init__(self, encoder: nn.Module, num_tactics: int, dropout: float = _DROPOUT) -> None:
        super().__init__()
        if num_tactics < 1:
            raise ValueError(f"num_tactics must be >= 1, got {num_tactics}")
        self.encoder = encoder
        self.num_tactics = num_tactics
        hidden_size = encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.risk_head = nn.Linear(hidden_size, 1)
        self.tactic_head = nn.Linear(hidden_size, num_tactics)

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Return `{"risk_logit": [B], "tactic_logits": [B, num_tactics]}`.

        Accepts `inputs_embeds` (instead of `input_ids`) so Captum can attribute through
        the embedding layer (`attribution.py`).
        """
        encoder_kwargs: dict[str, torch.Tensor] = {}
        if attention_mask is not None:
            encoder_kwargs["attention_mask"] = attention_mask
        if inputs_embeds is not None:
            encoder_kwargs["inputs_embeds"] = inputs_embeds
        else:
            encoder_kwargs["input_ids"] = input_ids

        outputs = self.encoder(**encoder_kwargs)
        pooled = self.dropout(_mean_pool(outputs.last_hidden_state, attention_mask))
        return {
            "risk_logit": self.risk_head(pooled).squeeze(-1),
            "tactic_logits": self.tactic_head(pooled),
        }


def _mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor | None) -> torch.Tensor:
    """Attention-masked mean of token embeddings -- a far more stable sequence
    representation for XLM-R than the raw <s> token (which collapses without heavy
    fine-tuning)."""
    if attention_mask is None:
        return last_hidden_state.mean(dim=1)
    mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=_MIN_TOKENS)
    return summed / counts


def build_model(base_model_name: str, num_tactics: int) -> ScamClassifierModel:  # pragma: no cover - loads real weights
    """Build a `ScamClassifierModel` around a pretrained `base_model_name` encoder."""
    from transformers import AutoModel

    encoder = AutoModel.from_pretrained(base_model_name)
    return ScamClassifierModel(encoder, num_tactics)


def build_encoder_from_config(base_model_name: str) -> nn.Module:  # pragma: no cover - light config-only load
    """Build a randomly-initialised encoder with `base_model_name`'s architecture but WITHOUT
    downloading its weights -- used at inference to reconstruct the graph before loading a
    fine-tuned `state_dict` (avoids re-downloading ~1GB of base weights)."""
    from transformers import AutoConfig, AutoModel

    config = AutoConfig.from_pretrained(base_model_name)
    return AutoModel.from_config(config)
