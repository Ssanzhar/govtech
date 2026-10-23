"""The span vote and its two baselines.

span_vote: cut the timeline at every word boundary of the three hypotheses; in each elementary
span, language l scores  dur * (logit(p_l) + bias_l)  where p_l is the calibrated probability that
l's overlapping word is right and in the right language (`missing` replaces the logit when l has no
word there); a 3-state Viterbi with a switch penalty (free after a pause >= `pause` s with no word
from any model) picks a language per span; each language's words whose midpoint lies in one of
its spans are emitted in time order (a word covering several spans is emitted once).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from common import LANGS, logit

NEG = -1e9
EPS = 1e-6


@dataclass(frozen=True)
class Params:
    switch: float = 1.0
    missing: float = -4.0
    pause: float | None = 0.15
    bias: dict = field(default_factory=lambda: dict.fromkeys(LANGS, 0.0))
    raw: bool = False          # True: emission = raw conf (minus `handicap`), no HMM (baseline b)
    handicap: dict = field(default_factory=dict)


def _word_at(words: list[list], x: float):
    for i, w in enumerate(words):
        if w[1] - EPS <= x < w[2] + EPS:
            return i
    return None


def elementary_spans(hyps: dict, cuts=()) -> list[tuple[float, float]]:
    points = sorted({p for h in hyps.values() for w in h for p in (w[1], w[2])} | set(cuts))
    return [(a, b) for a, b in zip(points, points[1:]) if b - a > EPS]


def span_vote(hyps: dict, probs: dict, params: Params, frozen: list | None = None):
    """-> (output words [(word, start, end, lang)], labelled spans [(a, b, lang)])."""
    frozen = frozen or []
    spans = elementary_spans(hyps, cuts=[c for a, b, _ in frozen for c in (a, b)])
    rows, gaps, keep = [], [], []
    gap = math.inf
    for a, b in spans:
        x = (a + b) / 2
        idx = {l: _word_at(hyps[l], x) for l in LANGS}
        if all(i is None for i in idx.values()):
            gap = (gap if gap != math.inf else 0.0) + (b - a)
            continue
        dur = b - a
        em = []
        for l in LANGS:
            i = idx[l]
            if params.raw:
                em.append((hyps[l][i][3] - params.handicap.get(l, 0.0)) if i is not None else -1.0)
            else:
                score = logit(probs[l][i]) if i is not None else params.missing
                em.append(dur * (score + params.bias.get(l, 0.0)))
        lock = next((fl for fa, fb, fl in frozen if fa - EPS <= x < fb + EPS), None)
        if lock is not None:
            em = [e if l == lock else NEG for e, l in zip(em, LANGS)]
        rows.append(em)
        gaps.append(gap)
        keep.append((a, b))
        gap = 0.0
    if not rows:
        return [], []
    states = _viterbi(np.array(rows), gaps, params)
    labelled = [(a, b, LANGS[s]) for (a, b), s in zip(keep, states)]
    out = []
    for l in LANGS:
        for w in hyps[l]:
            if state_at(labelled, (w[1] + w[2]) / 2) == l:
                out.append((w[0], w[1], w[2], l))
    return sorted(out, key=lambda w: (w[1], w[2])), labelled


def _viterbi(em: np.ndarray, gaps: list, params: Params) -> list[int]:
    if params.raw:
        return list(np.argmax(em, axis=1))
    n, k = em.shape
    score = em[0].copy()
    back = np.zeros((n, k), dtype=int)
    for t in range(1, n):
        free = params.pause is not None and gaps[t] >= params.pause
        pen = 0.0 if free else params.switch
        trans = score[:, None] - pen * (1 - np.eye(k))
        back[t] = np.argmax(trans, axis=0)
        score = trans[back[t], np.arange(k)] + em[t]
    path = [int(np.argmax(score))]
    for t in range(n - 1, 0, -1):
        path.append(int(back[t][path[-1]]))
    return path[::-1]


def state_at(labelled: list, x: float):
    for a, b, l in labelled:
        if a - EPS <= x < b + EPS:
            return l
    return None


EN_HANDICAP, TIE_EPSILON, KK_SHARE = 0.15, 0.02, 0.08


def utterance_vote(hyps: dict) -> str | None:
    """Python replica of site/core/asr.js voteFinal(hyps, 'kk', {en: 0.15}) on mean word conf."""
    ranked = sorted(((float(np.mean([w[3] for w in hyps[l]])) - (EN_HANDICAP if l == "en" else 0.0), l)
                     for l in LANGS if hyps[l]), reverse=True)
    if not ranked:
        return None
    best_rank, best = ranked[0]
    tied = [l for r, l in ranked if best_rank - r <= TIE_EPSILON]
    if "kk" in tied and "ru" in tied:
        text = "".join(w[0] for w in hyps["kk"])
        letters = [c for c in text if c.isalpha()]
        share = sum(c in "әғқңөұүһі" for c in letters) / len(letters) if letters else 0.0
        return "kk" if share >= KK_SHARE else "ru"
    if len(tied) > 1:
        return next(l for l in LANGS if l in tied)
    return best


def utterance_output(hyps: dict):
    lang = utterance_vote(hyps)
    if lang is None:
        return [], []
    words = [(w[0], w[1], w[2], lang) for w in hyps[lang]]
    return words, [(w[1], w[2], lang) for w in hyps[lang]]
