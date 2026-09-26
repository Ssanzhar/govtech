"""Offline test scaffolding for the `xlmr` backend: a tiny random-config XLM-R model and a
fake offset-returning tokenizer, so train/attribution/predict smoke tests never download
`xlm-roberta-base`. The fake tokenizer mimics the slice of the HuggingFace tokenizer API our
code actually uses (`return_offsets_mapping`, `return_tensors`, `truncation`, `padding`),
so code written against it also works with the real tokenizer at runtime."""

from __future__ import annotations

import re

import numpy as np
import pytest

_VOCAB_SIZE = 200
_BOS, _PAD, _EOS = 0, 1, 2
_WORD = re.compile(r"\S+")


class FakeTokenizer:
    """Whitespace tokenizer producing deterministic ids + char offset mappings."""

    def __init__(self, vocab_size: int = _VOCAB_SIZE) -> None:
        self.vocab_size = vocab_size
        self.pad_token_id = _PAD

    def _encode_one(self, text: str, max_length: int) -> tuple[list[int], list[tuple[int, int]]]:
        ids = [_BOS]
        offsets: list[tuple[int, int]] = [(0, 0)]
        for match in _WORD.finditer(text):
            token = match.group()
            ids.append(hash(token) % (self.vocab_size - 3) + 3)
            offsets.append((match.start(), match.end()))
            if len(ids) >= max_length - 1:
                break
        ids.append(_EOS)
        offsets.append((0, 0))
        return ids, offsets

    def __call__(
        self,
        text,
        *,
        return_offsets_mapping: bool = False,
        return_tensors: str | None = None,
        truncation: bool = True,
        padding: bool = False,
        max_length: int = 64,
    ):
        texts = [text] if isinstance(text, str) else list(text)
        encoded = [self._encode_one(t, max_length) for t in texts]
        width = max(len(ids) for ids, _ in encoded)

        id_rows, mask_rows, offset_rows = [], [], []
        for ids, offsets in encoded:
            pad = width - len(ids) if padding else 0
            id_rows.append(ids + [_PAD] * pad)
            mask_rows.append([1] * len(ids) + [0] * pad)
            offset_rows.append(offsets + [(0, 0)] * pad)

        out: dict = {"input_ids": id_rows, "attention_mask": mask_rows}
        if return_offsets_mapping:
            out["offset_mapping"] = offset_rows
        if return_tensors == "pt":
            import torch

            out["input_ids"] = torch.tensor(out["input_ids"], dtype=torch.long)
            out["attention_mask"] = torch.tensor(out["attention_mask"], dtype=torch.long)
        elif isinstance(text, str):
            # HF returns un-nested lists for a single string input.
            out = {k: (v[0] if k != "offset_mapping" else v[0]) for k, v in out.items()}
        return out


def build_tiny_encoder():
    # The xlmr backend is abandoned (eval_report); its smoke tests need the optional
    # torch/transformers stack and skip without it so the lean install stays testable.
    pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    XLMRobertaConfig, XLMRobertaModel = transformers.XLMRobertaConfig, transformers.XLMRobertaModel

    config = XLMRobertaConfig(
        vocab_size=_VOCAB_SIZE,
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=64,
        max_position_embeddings=64,
        type_vocab_size=1,
    )
    return XLMRobertaModel(config)


@pytest.fixture
def tiny_encoder():
    return build_tiny_encoder()


@pytest.fixture
def fake_tokenizer():
    return FakeTokenizer()


class FakeEmbedder:
    """Deterministic offline embedder. Texts containing `scam_marker` map to a distinct
    region so scam/legit are linearly separable -- lets LR smoke tests actually learn."""

    def __init__(self, dim: int = 16, scam_marker: str = "SMS") -> None:
        self.dim = dim
        self.scam_marker = scam_marker
        self.seen: list[str] = []

    def encode(self, texts, normalize_embeddings: bool = True, convert_to_numpy: bool = True):
        self.seen = list(texts)
        rows = []
        for text in texts:
            vec = [0.0] * self.dim
            if self.scam_marker.lower() in text.lower():
                vec[0], vec[1] = 1.0, 0.5
            else:
                vec[2], vec[3] = 1.0, 0.5
            vec[4] = (hash(text) % 100) / 1000.0  # tiny deterministic jitter
            if normalize_embeddings:
                norm = sum(v * v for v in vec) ** 0.5 or 1.0
                vec = [v / norm for v in vec]
            rows.append(vec)
        return np.array(rows, dtype=np.float32)


@pytest.fixture
def fake_embedder():
    return FakeEmbedder()
