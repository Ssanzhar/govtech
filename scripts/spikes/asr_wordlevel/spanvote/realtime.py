"""Replay the logged partials/finals to simulate what the live screen shows.

At every arrival time t (a partial or final from any recogniser, 0.25 s grid in audio seconds)
each method labels the current timeline. For each word of the method's last (end-of-audio)
output we follow the tag shown at the word's midpoint over time:
- latency  = (first t after which the shown tag never changes again) - word end, audio seconds;
- flicker  = number of tag changes while shown (appearance is not a change).
Freeze rule (span vote only): once a stretch of timeline is N s behind the newest decoded word end,
its language is locked; later Viterbi passes are constrained to it.
"""

from __future__ import annotations

import numpy as np

from calibrate import predict
from common import LANGS, clean_words, hyp_features, hypothesis_at
from vote import Params, span_vote, state_at, utterance_output

SHIPPED_PREFERENCE = LANGS  # initialAsrState(languages): first language is preferred


def arrival_times(row: dict) -> list[float]:
    ts = {e["t"] for l in LANGS for k in ("finals", "partials") for e in row["models"][l][k]}
    return sorted(ts | {round(row["duration"], 3)})


def hyps_at(row: dict, t: float) -> dict:
    return {l: clean_words(hypothesis_at(row["models"][l], t)) for l in LANGS}


def label_span(hyps, cal, params, lid, frozen):
    probs = predict(cal, hyp_features(hyps, lid)) if not params.raw else {l: None for l in LANGS}
    return span_vote(hyps, probs, params, frozen)


def label_shipped(hyps, final: bool):
    if final:
        return utterance_output(hyps)
    lang = next((l for l in SHIPPED_PREFERENCE if hyps[l]), None)  # bestPartial: preferred language's partial
    if lang is None:
        return [], []
    return [(w[0], w[1], w[2], lang) for w in hyps[lang]], [(w[1], w[2], lang) for w in hyps[lang]]


def simulate(row: dict, method: str, cal=None, params: Params | None = None, lid=None, freeze: float | None = None):
    """-> (final output words, per-word records [{lat, first, flicker}])."""
    times = arrival_times(row)
    history, frozen, frozen_until = [], [], 0.0
    out = []
    for i, t in enumerate(times):
        hyps = hyps_at(row, t)
        last = i == len(times) - 1
        if method == "shipped":
            out, labelled = label_shipped(hyps, final=last)
        else:
            out, labelled = label_span(hyps, cal, params, lid, frozen)
        history.append((t, labelled))
        if freeze is not None and method == "span":
            newest = max((w[2] for h in hyps.values() for w in h), default=0.0)
            until = max(frozen_until, newest - freeze)
            for a, b, l in labelled:
                lo, hi = max(a, frozen_until), min(b, until)
                if hi > lo:
                    frozen.append((lo, hi, l))
            frozen_until = until
    records = []
    for w in out:
        x = (w[1] + w[2]) / 2
        tags = [(t, state_at(lab, x)) for t, lab in history]
        shown = [(t, g) for t, g in tags if g is not None]
        flicker = sum(1 for (_, a), (_, b) in zip(shown, shown[1:]) if a != b)
        stable = times[-1]
        for t, g in reversed(tags):
            if g != w[3]:
                break
            stable = t
        records.append({"lat": stable - w[2], "first": (shown[0][0] - w[2]) if shown else np.nan, "flicker": flicker})
    return out, records


def summarize_rt(per_row: list[list[dict]]) -> dict:
    """Word-weighted means with a cluster (utterance) bootstrap."""
    from metrics import BOOTSTRAP, SEED

    rows = [r for r in per_row if r]
    out = {"n_words": sum(len(r) for r in rows)}
    rng = np.random.default_rng(SEED)
    picks = rng.integers(0, len(rows), size=(BOOTSTRAP, len(rows)))
    n = np.array([len(r) for r in rows], dtype=float)
    stats = {
        "latency_mean_s": [sum(w["lat"] for w in r) for r in rows],
        "first_shown_mean_s": [sum(w["first"] for w in r) for r in rows],
        "flicker_per_word": [sum(w["flicker"] for w in r) for r in rows],
        "share_words_flipped": [sum(w["flicker"] > 0 for w in r) for r in rows],
    }
    for name, vals in stats.items():
        v = np.array(vals, dtype=float)
        out[name] = round(float(v.sum() / n.sum()), 4)
        boot = v[picks].sum(axis=1) / n[picks].sum(axis=1)
        out[name + "_ci"] = [round(float(np.percentile(boot, 2.5)), 4), round(float(np.percentile(boot, 97.5)), 4)]
    lats = np.array([w["lat"] for r in rows for w in r])
    out["latency_p50_s"] = round(float(np.percentile(lats, 50)), 3)
    out["latency_p90_s"] = round(float(np.percentile(lats, 90)), 3)
    return out
