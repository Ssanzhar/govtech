"""Shared loading, normalisation, alignment and per-word features for the span-vote spike."""

from __future__ import annotations

import json
import math
import re
import sys
import unicodedata
from glob import glob
from pathlib import Path

import jiwer
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
sys.path.insert(0, str(HERE.parent))

LANGS = ("kk", "ru", "en")
KAZAKH_ONLY = set("әғқңөұүһі")
MANIFESTS = {"dev": REPO / "data/asr_capture/spanvote_dev_set.jsonl", "test": REPO / "data/asr_capture/cs_set.jsonl"}
DECODE_GLOBS = {"dev": "data/asr_capture/spanvote_dev_decodes.*.jsonl", "test": "data/asr_capture/spanvote_test_decodes.*.jsonl"}
CONF_CLIP = 1e-3
MIN_DUR = 0.03


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFC", text).lower().replace("ё", "е")
    text = re.sub(r"[^\w\s']", " ", text)
    return re.sub(r"\s+", " ", text.replace("_", " ")).strip()


def key_of(row: dict) -> tuple:
    return row["id"], row["condition"], row["channel"]


def load(split: str) -> list[dict]:
    """Manifest rows joined with their partial-logging decodes, in manifest order."""
    manifest = [json.loads(l) for l in MANIFESTS[split].read_text(encoding="utf-8").splitlines() if l.strip()]
    decodes = {}
    for path in sorted(glob(str(REPO / DECODE_GLOBS[split]))):
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if line.strip():
                d = json.loads(line)
                decodes[key_of(d)] = d
    rows = []
    for m in manifest:
        d = decodes.get(key_of(m))
        if d is None:
            continue
        ref = [(normalize(w["word"]), w["lang"]) for w in m["words"]]
        ref = [(w, l) for w, l in ref if w]
        rows.append({**m, "ref_words": [w for w, _ in ref], "ref_langs": [l for _, l in ref], "models": d["models"]})
    return rows


def final_words(model_decode: dict) -> list[list]:
    return [w for f in model_decode["finals"] for w in f["words"]]


def hypothesis_at(model_decode: dict, t: float) -> list[list]:
    """What one recogniser shows at audio time t: finals so far + the latest partial after them."""
    words, last_final = [], -1.0
    for f in model_decode["finals"]:
        if f["t"] <= t:
            words += f["words"]
            last_final = f["t"]
    partial = None
    for p in model_decode["partials"]:
        if last_final < p["t"] <= t:
            partial = p["words"]
    return words + (partial or [])


def align(ref_words: list[str], hyp_words: list[str]):
    """(op, ref_index | None, hyp_index | None) triples from jiwer's alignment."""
    if not hyp_words:
        return [("delete", i, None) for i in range(len(ref_words))]
    if not ref_words:
        return [("insert", None, j) for j in range(len(hyp_words))]
    out = jiwer.process_words(" ".join(ref_words), " ".join(hyp_words))
    pairs = []
    for c in out.alignments[0]:
        if c.type in ("equal", "substitute"):
            for k in range(c.ref_end_idx - c.ref_start_idx):
                pairs.append(("hit" if c.type == "equal" else "sub", c.ref_start_idx + k, c.hyp_start_idx + k))
        elif c.type == "delete":
            pairs += [("delete", i, None) for i in range(c.ref_start_idx, c.ref_end_idx)]
        else:
            pairs += [("insert", None, j) for j in range(c.hyp_start_idx, c.hyp_end_idx)]
    return pairs


def clean_words(words: list[list]) -> list[list]:
    """Drop words that normalise to nothing; return [norm_word, start, end, conf]."""
    out = []
    for w, s, e, c in words:
        for piece in normalize(w).split()[:1]:
            out.append([piece, s, e, c])
    return out


def logit(p: float) -> float:
    p = min(1 - CONF_CLIP, max(CONF_CLIP, p))
    return math.log(p / (1 - p))


FEATURES_BASE = ("logit_conf", "conf_is_one", "duration", "log_cps", "log_chars", "has_kk", "has_latin",
                 "utt_mean_conf", "utt_mean_conf_best_other", "utt_words_per_s",
                 "overlap_conf_best_other", "overlap_words_per_s_other")
FEATURES_LID = FEATURES_BASE + ("charlm_kk_minus_ru",)


def overlap_stats(word: list, others: list[list[list]]) -> tuple[float, float]:
    """Best duration-weighted mean conf of another recogniser over this word's time, and how many
    other-model words per second overlap it (competitors confident on the same audio)."""
    s, e = word[1], word[2]
    dur = max(e - s, MIN_DUR)
    best, count = 0.0, 0
    for hyp in others:
        num = den = 0.0
        for w in hyp:
            ov = min(e, w[2]) - max(s, w[1])
            if ov > 0:
                num, den, count = num + ov * w[3], den + ov, count + 1
        if den > 0:
            best = max(best, num / den)
    return best, count / dur


def word_features(word: list, hyp: list[list], others_mean: float, lid=None, others=()) -> list[float]:
    """Cheap per-word features from one recogniser's word, its hypothesis and the competitors."""
    w, s, e, c = word
    dur = max(e - s, MIN_DUR)
    n = max(len(w), 1)
    span = max(hyp[-1][2] - hyp[0][1], MIN_DUR) if hyp else MIN_DUR
    feats = [logit(c), float(c >= 0.999), dur, math.log(n / dur), math.log(n),
             float(any(ch in KAZAKH_ONLY for ch in w)), float(bool(re.search(r"[a-z]", w))),
             float(np.mean([x[3] for x in hyp])) if hyp else 0.0, others_mean, len(hyp) / span,
             *overlap_stats(word, list(others))]
    if lid is not None:
        cyr = not re.search(r"[a-z]", w)
        feats.append((lid.char_lms["kk"].logprob(w) - lid.char_lms["ru"].logprob(w)) / n if cyr else 0.0)
    return feats


def hyp_features(hyps: dict[str, list[list]], lid=None) -> dict[str, np.ndarray]:
    """Feature matrix per language for the (cleaned) hypotheses of all three recognisers."""
    means = {l: (float(np.mean([x[3] for x in h])) if h else 0.0) for l, h in hyps.items()}
    out = {}
    for l, h in hyps.items():
        others = max(means[o] for o in hyps if o != l)
        rivals = [hyps[o] for o in hyps if o != l]
        out[l] = np.array([word_features(w, h, others, lid, rivals) for w in h]) if h else np.zeros((0, 0))
    return out
