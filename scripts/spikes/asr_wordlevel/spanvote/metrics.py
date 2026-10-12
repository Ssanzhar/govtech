"""Word-level metrics of a stitched, language-tagged transcript against the reference."""

from __future__ import annotations

import numpy as np

from common import LANGS, align

BOOTSTRAP = 1000
SEED = 20261008
KEYS = ("ref", "err", "tag_ok", "aligned", "good", "en_ref", "en_as_en", "kkru_ref", "kkru_as_en",
        *(f"tag_ok_{l}" for l in LANGS), *(f"aligned_{l}" for l in LANGS))
RATIOS = {
    "word_lang_acc": ("tag_ok", "aligned"), "wer": ("err", "ref"), "right_word_right_tag": ("good", "ref"),
    "en_recall": ("en_as_en", "en_ref"), "kkru_to_en_rate": ("kkru_as_en", "kkru_ref"),
    **{f"word_lang_acc_{l}": (f"tag_ok_{l}", f"aligned_{l}") for l in LANGS},
}


def row_counts(row: dict, out_words: list) -> dict:
    """Counts for one utterance. out_words: [(word, start, end, lang)] in order."""
    c = dict.fromkeys(KEYS, 0)
    refl = row["ref_langs"]
    c["ref"] = len(row["ref_words"])
    c["en_ref"] = sum(l == "en" for l in refl)
    c["kkru_ref"] = c["ref"] - c["en_ref"]
    for op, r, h in align(row["ref_words"], [w[0] for w in out_words]):
        c["err"] += op != "hit"
        if r is None or h is None:
            continue
        lang, tag = refl[r], out_words[h][3]
        c["aligned"] += 1
        c[f"aligned_{lang}"] += 1
        c["tag_ok"] += tag == lang
        c[f"tag_ok_{lang}"] += tag == lang
        c["good"] += op == "hit" and tag == lang
        if tag == "en":
            c["en_as_en" if lang == "en" else "kkru_as_en"] += 1
    return c


def summarize(counts: list[dict], ci: bool = True) -> dict:
    arr = {k: np.array([c[k] for c in counts], dtype=float) for k in KEYS}
    out = {"n_utts": len(counts), "n_ref_words": int(arr["ref"].sum())}
    rng = np.random.default_rng(SEED)
    picks = rng.integers(0, len(counts), size=(BOOTSTRAP, len(counts))) if ci and counts else None
    for name, (num, den) in RATIOS.items():
        d = arr[den].sum()
        if d == 0:
            continue
        out[name] = round(float(arr[num].sum() / d), 4)
        if picks is not None:
            boot = arr[num][picks].sum(axis=1) / np.maximum(arr[den][picks].sum(axis=1), 1)
            out[name + "_ci"] = [round(float(np.percentile(boot, 2.5)), 4), round(float(np.percentile(boot, 97.5)), 4)]
    return out


def bootstrap_mean(values: list[float]) -> list[float]:
    if not values:
        return [float("nan")] * 2
    v = np.array(values, dtype=float)
    rng = np.random.default_rng(SEED)
    boot = v[rng.integers(0, len(v), size=(BOOTSTRAP, len(v)))].mean(axis=1)
    return [round(float(np.percentile(boot, 2.5)), 4), round(float(np.percentile(boot, 97.5)), 4)]
