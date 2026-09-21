"""Reassurance feature: a NEGATIVE (anti-scam) signal for the hybrid `linear` classifier.

Detects a legitimate institution PROACTIVELY telling the customer that sensitive data is NOT
needed ("we will never ask for your code/card"). This is the mirror of the hard-signal
request cues and the signal that separates real bank fraud-alert / "card ready" calls (which
reassure) from scams (which demand) -- the specific failure mode of the embedding-only model
(`docs/eval_report.md`). Deterministic, offline, loaded from a committed pattern vocabulary.

A reassurance fires when a `sensitive_term` co-occurs within `window_chars` of a
`reassurance_term` (either order) in `text`. `reassurance_term`s are negation-of-need phrases
ONLY -- never scam-secrecy phrasing -- so the feature never fires on a scam's demands.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import yaml
from pydantic import BaseModel, ConfigDict, ValidationError, model_validator

_SENTENCE_STOP = r"[^.!?\n]"


class ReassuranceError(ValueError):
    """Raised when the reassurance pattern file is missing or malformed."""


class ReassurancePatterns(BaseModel):
    """Validated reassurance vocabulary (immutable)."""

    model_config = ConfigDict(frozen=True)

    version: int
    sensitive_terms: tuple[str, ...]
    reassurance_terms: tuple[str, ...]
    window_chars: int = 70

    @model_validator(mode="after")
    def _non_empty(self) -> "ReassurancePatterns":
        if not self.sensitive_terms:
            raise ValueError("reassurance patterns need at least one sensitive_term")
        if not self.reassurance_terms:
            raise ValueError("reassurance patterns need at least one reassurance_term")
        if self.window_chars <= 0:
            raise ValueError(f"window_chars must be > 0, got {self.window_chars}")
        for term in (*self.sensitive_terms, *self.reassurance_terms):
            if not term or not term.strip():
                raise ValueError("reassurance terms must not be blank")
        return self


def load_reassurance_patterns(path: Path | None = None) -> ReassurancePatterns:
    """Load + validate the reassurance pattern YAML (defaults to config). Raises `ReassuranceError`."""
    from qorgan.config import get_config

    resolved = path or get_config().reassurance_patterns_path
    if not resolved.exists():
        raise ReassuranceError(f"Reassurance pattern file not found: {resolved}")
    try:
        raw = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ReassuranceError(f"Invalid YAML in {resolved}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ReassuranceError(f"Reassurance patterns {resolved} must be a mapping")
    try:
        return ReassurancePatterns(
            version=int(raw.get("version", 1)),
            sensitive_terms=tuple(raw.get("sensitive_terms") or ()),
            reassurance_terms=tuple(raw.get("reassurance_terms") or ()),
            window_chars=int(raw.get("window_chars", 70)),
        )
    except (ValidationError, ValueError) as exc:
        raise ReassuranceError(f"Reassurance pattern validation failed for {resolved}: {exc}") from exc


def _term(term: str) -> str:
    """A term as a regex: its tokens joined by optional whitespace, so ASR output that glues
    a negation to the next word ("ненужно") or doubles a space still matches (PLAN A10 --
    the browser recogniser produced exactly that on the demo's real-bank-call scene)."""
    return r"\s*".join(re.escape(token) for token in term.split())


def _compile(patterns: ReassurancePatterns) -> re.Pattern[str]:
    sensitive = "(?:" + "|".join(_term(t) for t in patterns.sensitive_terms) + ")"
    reassure = "(?:" + "|".join(_term(t) for t in patterns.reassurance_terms) + ")"
    gap = _SENTENCE_STOP + "{0," + str(patterns.window_chars) + "}?"
    return re.compile(f"{sensitive}{gap}{reassure}|{reassure}{gap}{sensitive}", re.IGNORECASE)


def reassurance_scores(texts: Sequence[str], patterns: ReassurancePatterns) -> np.ndarray:
    """`(n, 1)` feature: 1.0 if `text` contains a sensitive term within the window of a
    reassurance term (either order), else 0.0. Compiles the pattern once for the batch."""
    matcher = _compile(patterns)
    return np.array([[1.0 if matcher.search(text) else 0.0] for text in texts], dtype=np.float32)


def reassures(text: str, patterns: ReassurancePatterns) -> bool:
    """Whether `text` contains a reassurance (single-text convenience)."""
    return bool(_compile(patterns).search(text))


def reassurance_hash(patterns: ReassurancePatterns) -> str:
    """Order-independent content fingerprint (sha256), so a trained bundle can detect a
    reassurance-pattern drift at load time."""
    canonical = json.dumps(
        {
            "sensitive": sorted(patterns.sensitive_terms),
            "reassurance": sorted(patterns.reassurance_terms),
            "window": patterns.window_chars,
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
