"""Token-attribution to character-span alignment for grounded explanations (CLAUDE.md
SS6 -- every trigger phrase shown to a user must be a real attributed span from the
transcript).

`align_token_attributions_to_spans` (pure) turns per-token attribution scores + tokenizer
offset mappings into merged, verbatim `Span`s. `integrated_gradient_spans` produces those
token scores via Captum Layer Integrated Gradients over the encoder's embedding layer,
attributing the risk logit. Torch/Captum are imported lazily inside that function, so the
pure aligner (and anything importing it) never pulls in torch.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from qorgan.data.schema import UTTERANCE_JOIN, Span

_ZERO_WIDTH = 0
# Captum IG defaults: enough integration steps for a stable attribution, and a modest
# cap on how many trigger spans we surface to a user.
_IG_STEPS = 32
_IG_TOP_K = 8


def align_token_attributions_to_spans(
    token_scores: Sequence[float],
    offset_mapping: Sequence[tuple[int, int]],
    transcript: str,
    *,
    top_k: int,
    min_score: float = 0.0,
) -> tuple[Span, ...]:
    """Select the `top_k` highest-scoring real tokens (special tokens with zero-width
    offsets skipped) at or above `min_score`, then merge contiguous/overlapping token
    ranges into verbatim `Span`s sorted by start.

    Raises `ValueError` if `token_scores` and `offset_mapping` differ in length, or if
    `top_k <= 0`.
    """
    if len(token_scores) != len(offset_mapping):
        raise ValueError(
            "token_scores and offset_mapping must have the same length, "
            f"got {len(token_scores)} and {len(offset_mapping)}"
        )
    if top_k <= 0:
        raise ValueError(f"top_k must be > 0, got {top_k}")

    candidates = _real_candidates(token_scores, offset_mapping, min_score)
    selected = _select_top_k(candidates, top_k)
    ranges = _merge_ranges(selected)
    return _build_spans(ranges, transcript)


def select_top_utterance_spans(
    utterances: Sequence[str],
    scores: Sequence[float],
    *,
    top_k: int,
    min_score: float = 0.0,
) -> tuple[Span, ...]:
    """Grounded attribution for the embeddings/linear backend: highlight the `top_k`
    highest-risk *utterances* (score >= `min_score`) as verbatim `Span`s.

    Offsets are computed against `UTTERANCE_JOIN.join(utterances)` (the same transcript the
    classifier scored), so duplicate utterances resolve to their correct distinct positions
    rather than a naive first-match. Raises `ValueError` on length mismatch or `top_k <= 0`.
    """
    if len(utterances) != len(scores):
        raise ValueError(
            f"utterances and scores must have the same length, got {len(utterances)} and {len(scores)}"
        )
    if top_k <= 0:
        raise ValueError(f"top_k must be > 0, got {top_k}")

    candidates: list[tuple[float, int, int, int]] = []  # (score, index, start, end)
    cursor = 0
    for index, (utterance, score) in enumerate(zip(utterances, scores)):
        start, end = cursor, cursor + len(utterance)
        cursor = end + len(UTTERANCE_JOIN)
        if score >= min_score and utterance.strip():
            candidates.append((score, index, start, end))

    top = sorted(candidates, key=lambda c: (-c[0], c[1]))[:top_k]
    spans = [Span(text=utterances[index], start=start, end=end) for _score, index, start, end in top]
    return tuple(sorted(spans, key=lambda span: span.start))


def _real_candidates(
    token_scores: Sequence[float],
    offset_mapping: Sequence[tuple[int, int]],
    min_score: float,
) -> list[tuple[float, int, int]]:
    """Real (non-special) tokens meeting `min_score`, as `(score, start, end)` triples."""
    candidates: list[tuple[float, int, int]] = []
    for score, (start, end) in zip(token_scores, offset_mapping):
        if end <= start:
            continue
        if end - start == _ZERO_WIDTH:
            continue
        if score < min_score:
            continue
        candidates.append((score, start, end))
    return candidates


def _select_top_k(
    candidates: Sequence[tuple[float, int, int]], top_k: int
) -> list[tuple[int, int]]:
    """Highest-scoring `top_k` candidates (ties broken by earlier start), as `(start, end)`."""
    ordered = sorted(candidates, key=lambda triple: (-triple[0], triple[1]))
    top = ordered[:top_k]
    return [(start, end) for _score, start, end in top]


def _merge_ranges(ranges: Sequence[tuple[int, int]]) -> list[tuple[int, int]]:
    """Merge char ranges that are contiguous or overlapping, sorted by start."""
    if not ranges:
        return []

    ordered = sorted(ranges, key=lambda pair: pair[0])
    merged: list[tuple[int, int]] = [ordered[0]]
    for start, end in ordered[1:]:
        current_start, current_end = merged[-1]
        if start <= current_end:
            merged[-1] = (current_start, max(current_end, end))
        else:
            merged.append((start, end))
    return merged


def _build_spans(ranges: Sequence[tuple[int, int]], transcript: str) -> tuple[Span, ...]:
    spans: list[Span] = []
    for start, end in ranges:
        text = transcript[start:end]
        if not text.strip():
            continue
        spans.append(Span(text=text, start=start, end=end))
    return tuple(spans)


def _normalize_offsets(raw: Any) -> list[tuple[int, int]]:
    """Coerce a tokenizer `offset_mapping` (tensor or nested list, batched) into a flat
    list of `(start, end)` for the single encoded example."""
    if hasattr(raw, "tolist"):
        raw = raw.tolist()
    rows = raw[0] if raw and isinstance(raw[0][0], (list, tuple)) else raw
    return [(int(start), int(end)) for start, end in rows]


def integrated_gradient_spans(
    model: Any,
    tokenizer: Any,
    transcript: str,
    *,
    device: Any = None,
    max_length: int = 256,
    top_k: int = _IG_TOP_K,
    n_steps: int = _IG_STEPS,
    min_score: float = 0.0,
) -> tuple[Span, ...]:
    """Attribute the model's risk logit back to input tokens via Layer Integrated
    Gradients, then align the highest-contributing tokens to verbatim `Span`s.

    Runs on CPU by default (single-example inference is cheap and avoids MPS/Captum
    quirks). `min_score=0.0` keeps only tokens that push risk *up* (real trigger phrases).
    """
    import torch
    from captum.attr import LayerIntegratedGradients

    active_device = device or torch.device("cpu")
    model.to(active_device)
    model.eval()

    enc = tokenizer(
        transcript, return_offsets_mapping=True, return_tensors="pt", truncation=True, max_length=max_length
    )
    input_ids = enc["input_ids"].to(active_device)
    attention_mask = enc["attention_mask"].to(active_device)
    offsets = _normalize_offsets(enc["offset_mapping"])

    pad_id = getattr(tokenizer, "pad_token_id", 0) or 0
    baselines = torch.full_like(input_ids, pad_id)

    def forward_risk(ids: Any) -> Any:
        return model(input_ids=ids, attention_mask=attention_mask)["risk_logit"]

    lig = LayerIntegratedGradients(forward_risk, model.encoder.embeddings.word_embeddings)
    attributions = lig.attribute(input_ids, baselines=baselines, n_steps=n_steps)
    token_scores = attributions.sum(dim=-1).squeeze(0).detach().cpu().tolist()

    return align_token_attributions_to_spans(
        token_scores, offsets, transcript, top_k=top_k, min_score=min_score
    )
