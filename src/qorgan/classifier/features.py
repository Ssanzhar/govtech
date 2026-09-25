"""Hybrid feature pipeline for the `linear` classifier: frozen e5 embedding ⊕ deterministic
hard-signal cue features.

ONE function (`compute_feature_blocks`) builds the feature blocks, so training and both
inference paths (whole-transcript + per-utterance) share an identical layout -- drift here
would silently corrupt predictions. The 5 cue features (one per hard-signal tactic, in
`taxonomy.hard_signal_ids()` order) fire when the transcript *requests* that thing -- an ask a
legitimate bank/gov call never makes. They give the risk head a linearly-separable axis the
embedding alone does not expose, and are strictly additive: a cue-free scam keeps its
embedding-only behavior (cues are features, never a veto).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from qorgan.classifier import embed as embed_mod
from qorgan.classifier.cue_lexicon import CueLexicon
from qorgan.classifier.cue_match import find_cue
from qorgan.data.schema import Span
from qorgan.taxonomy import get_taxonomy


@dataclass(frozen=True)
class CueMatch:
    """A hard-signal cue that fired for a tactic, grounded to a verbatim transcript span."""

    tactic_id: str
    span: Span


@dataclass(frozen=True)
class FeatureBlocks:
    """The feature blocks for a batch of texts, plus the per-text grounded cue matches.

    `hard_signal` (scam-request cues) and `reassurance` (anti-scam "we never ask" signal) are
    the two interpretable blocks appended to the embedding by `hybrid_matrix`.
    """

    embedding: np.ndarray  # (n, embed_dim)
    hard_signal: np.ndarray  # (n, K) request-cue presence, columns in hard-signal-id order
    reassurance: np.ndarray  # (n, 1) anti-scam reassurance signal
    matches: tuple[tuple[CueMatch, ...], ...]  # one tuple of cue matches per input text


def hard_signal_feature_ids() -> tuple[str, ...]:
    """The hard-signal tactic ids, in the fixed column order of the cue-feature block."""
    return get_taxonomy().hard_signal_ids()


def match_cues(text: str, lexicon: CueLexicon) -> tuple[CueMatch, ...]:
    """Locate hard-signal request cues in `text` (one grounded match per tactic that fired).

    Matching is `cue_match.find_cue`: verbatim first, then a bounded-edit search over the
    de-spaced text, so a phrase the recogniser broke across a word boundary («на обороте» →
    «наоборот де») is still found (ADR D39). The span is always sliced from the ORIGINAL text
    (so `text[start:end] == span.text`) -- for a fuzzy hit that is what was actually said,
    which is the right thing to highlight.
    """
    matches: list[CueMatch] = []
    for tactic_id in hard_signal_feature_ids():
        span = _first_grounded_span(text, lexicon.entries.get(tactic_id, ()))
        if span is not None:
            matches.append(CueMatch(tactic_id=tactic_id, span=span))
    return tuple(matches)


def _first_grounded_span(text: str, cues: Sequence[str]) -> Span | None:
    for cue in cues:
        found = find_cue(text, cue)
        if found is None:
            continue
        start, end = found
        candidate = text[start:end]
        # An exact hit must slice back to the cue (a length-changing casefold would mean the
        # offsets lie); a fuzzy hit is the recogniser's wording and never equals the cue.
        if candidate and (len(candidate) != len(cue) or candidate.lower() == cue.lower()):
            return Span(text=candidate, start=start, end=end)
    return None


def hard_signal_features(texts: Sequence[str], lexicon: CueLexicon) -> np.ndarray:
    """Presence matrix `(n, K)`: 1.0 if the text contains any cue for a hard-signal tactic.

    Presence uses the same `cue_match.find_cue` as the span search, so the feature and the
    highlight can never disagree.
    """
    ids = hard_signal_feature_ids()
    text_list = list(texts)
    if not text_list:
        return np.zeros((0, len(ids)), dtype=np.float32)
    rows = [
        [1.0 if _tactic_present(text, lexicon.entries.get(tid, ())) else 0.0 for tid in ids]
        for text in text_list
    ]
    return np.array(rows, dtype=np.float32)


def _tactic_present(text: str, cues: Sequence[str]) -> bool:
    return any(find_cue(text, cue) is not None for cue in cues)


def compute_feature_blocks(
    texts: Sequence[str],
    *,
    embedder: Any = None,
    model_name: str | None = None,
    lexicon: CueLexicon,
    reassurance_patterns: Any = None,
) -> FeatureBlocks:
    """Embed `texts` once and compute the hard-signal cue block, reassurance block, and
    grounded cue matches.

    The single entry point shared by training and both inference paths, so the feature
    layout can never drift between them. `reassurance_patterns` defaults to the configured
    pattern file when not injected.
    """
    from qorgan.classifier.reassurance import load_reassurance_patterns, reassurance_scores

    text_list = list(texts)
    embedding = embed_mod.embed_texts(text_list, embedder=embedder, model_name=model_name)
    hard_signal = hard_signal_features(text_list, lexicon)
    patterns = reassurance_patterns or load_reassurance_patterns()
    reassurance = (
        reassurance_scores(text_list, patterns) if text_list else np.zeros((0, 1), dtype=np.float32)
    )
    matches = tuple(match_cues(text, lexicon) for text in text_list)
    return FeatureBlocks(
        embedding=embedding, hard_signal=hard_signal, reassurance=reassurance, matches=matches
    )


def hybrid_matrix(blocks: FeatureBlocks) -> np.ndarray:
    """Concatenate embedding + hard-signal + reassurance blocks into `(n, embed_dim + K + 1)`."""
    n = len(blocks.embedding)
    if len(blocks.hard_signal) != n or len(blocks.reassurance) != n:
        raise ValueError(
            f"feature-block row mismatch: embedding {n}, hard_signal {len(blocks.hard_signal)}, "
            f"reassurance {len(blocks.reassurance)}"
        )
    return np.hstack([blocks.embedding, blocks.hard_signal, blocks.reassurance]).astype(np.float32)
