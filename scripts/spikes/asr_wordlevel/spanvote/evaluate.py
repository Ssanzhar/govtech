"""Span vote vs the shipped utterance vote and an uncalibrated per-span max-conf vote.

1. Fit per-model calibrators on DEV finals (two feature sets: base, base + train-only char-LM
   kk-vs-ru score); choose by out-of-fold DEV log loss.
2. Tune switch penalty / missing-word logit / pause reset / language biases on DEV with
   out-of-fold probabilities. Objective: share of reference words output correctly AND tagged with
   the right language, subject to KK/RU->EN false switches <= 1 % on DEV.
3. Tune the freeze delay N on DEV (real-time replay of partials).
4. Score everything once on TEST (440 rows), per condition/channel, bootstrap 95 % CIs.

    python scripts/spikes/asr_wordlevel/spanvote/evaluate.py   # -> data/asr_capture/spanvote_results.json
"""

from __future__ import annotations

import itertools
import json
import os
import time
from collections import defaultdict
from multiprocessing import get_context

import numpy as np

import calibrate as cb
from common import FEATURES_BASE, FEATURES_LID, LANGS, REPO, load
from metrics import row_counts, summarize
from realtime import simulate, summarize_rt
from vote import Params, span_vote, utterance_output

WITH_EN = "en" in LANGS
# The D64 three-model run keeps its file; the kk/ru run (ADR D70) writes its own.
OUT = REPO / ("data/asr_capture/spanvote_results.json" if WITH_EN else "data/asr_capture/spanvote_results_kkru.json")
MAX_FALSE_EN = 0.01
GRID = {"switch": (0.0, 0.25, 0.5, 1.0, 2.0, 4.0), "missing": (-1.0, -2.0, -4.0, -6.0), "pause": (None, 0.15),
        "en_bias": (0.0, 0.5, 1.0, 1.5, 2.0, 3.0) if WITH_EN else (0.0,), "kk_bias": (-0.5, 0.0, 0.5)}
FREEZE_GRID = (0.5, 1.0, 1.5, 2.0, 3.0)
FREEZE_MAX_ACC_DROP = 0.005
RAW = Params(raw=True)
RAW_HANDICAP = Params(raw=True, handicap={"en": 0.15})
_STATE = {}


def offline_output(row, hyps, probs, method, params=None):
    if method == "shipped":
        return utterance_output(hyps)[0]
    if method == "shipped_retagged":
        # what the D64 page really showed in the fallback: the vote's text, re-tagged by the word LID
        words = utterance_output(hyps)[0]
        tags = _STATE["lid_tagger"].tag([w[0] for w in words])
        return [(w[0], w[1], w[2], t) for w, t in zip(words, tags)]
    return span_vote(hyps, probs, params)[0]


def score_rows(rows, hyps, probs, method, params=None):
    return [row_counts(r, offline_output(r, h, p, method, params)) for r, h, p in zip(rows, hyps, probs)]


def _grid_point(values):
    switch, missing, pause, en_bias, kk_bias = values
    params = Params(switch=switch, missing=missing, pause=pause, bias=_bias(kk_bias, en_bias))
    s = summarize(score_rows(_STATE["rows"], _STATE["hyps"], _STATE["probs"], "span", params), ci=False)
    return values, s


def tune(rows, hyps, probs):
    _STATE.update(rows=rows, hyps=hyps, probs=probs)
    points = list(itertools.product(*GRID.values()))
    with get_context("fork").Pool(2) as pool:
        results = pool.map(_grid_point, points, chunksize=8)
    ok = [(v, s) for v, s in results if not WITH_EN or s["kkru_to_en_rate"] <= MAX_FALSE_EN]
    best_v, best_s = max(ok, key=lambda vs: (vs[1]["right_word_right_tag"], vs[1]["word_lang_acc"]))
    switch, missing, pause, en_bias, kk_bias = best_v
    return Params(switch=switch, missing=missing, pause=pause, bias=_bias(kk_bias, en_bias)), best_s, len(ok)


def _bias(kk_bias, en_bias):
    return {"kk": kk_bias, "ru": 0.0, **({"en": en_bias} if WITH_EN else {})}


def oof_by_row(tab, oof, rows, hyps):
    probs = [{l: np.zeros(len(h[l])) for l in LANGS} for h in hyps]
    for l in LANGS:
        for (ri, wi), p in zip(tab[l]["where"], oof[l]):
            probs[ri][l][wi] = p
    return probs


def groups_of(rows):
    g = defaultdict(list)
    for i, r in enumerate(rows):
        g["all"].append(i)
        g[f"{r['condition']}/{r['channel']}"].append(i)
        if "en" not in r["ref_langs"]:
            g["no_en"].append(i)  # kk/ru speech only: the product's calls since ADR D70
    return dict(sorted(g.items()))


def calibration_step(dev, lid):
    tabs = {"base": cb.table(dev), "lid": cb.table(dev, lid)}
    report, oofs = {}, {}
    for name, tab in tabs.items():
        oofs[name] = cb.oof_predictions(tab)
        report[name] = {l: round(cb.log_loss(tab[l]["y"], oofs[name][l]), 4) for l in LANGS}
    report["raw_conf"] = {l: round(cb.log_loss(tabs["base"][l]["y"], 1 / (1 + np.exp(-tabs["base"][l]["X"][:, 0]))), 4) for l in LANGS}
    report["positive_rate"] = {l: round(float(tabs["base"][l]["y"].mean()), 4) for l in LANGS}
    choice = min(("base", "lid"), key=lambda n: sum(report[n].values()))
    print("calibration DEV out-of-fold log loss:", json.dumps(report), "->", choice, flush=True)
    return tabs[choice], oofs[choice], choice, report


def main() -> None:
    from word_lid import fitted

    started = time.time()
    dev, test = load("dev"), load("test")
    if not WITH_EN:
        dev = [r for r in dev if "en" not in r["ref_langs"]]  # tune for kk/ru speech (ADR D70); TEST stays whole
    print(f"dev rows {len(dev)}, test rows {len(test)}", flush=True)
    lid = fitted()
    _STATE["lid_tagger"] = lid
    tab, oof, feat_choice, cal_report = calibration_step(dev, lid)
    use_lid = lid if feat_choice == "lid" else None
    names = FEATURES_LID if use_lid else FEATURES_BASE
    dev_hyps = [cb.final_hyps(r) for r in dev]
    dev_probs = oof_by_row(tab, oof, dev, dev_hyps)
    params, dev_best, n_ok = tune(dev, dev_hyps, dev_probs)
    print(f"tuned on DEV ({n_ok} grid points within the EN false-switch cap): {params} -> {dev_best}", flush=True)
    cal = cb.fit(tab)

    dev_rt = {}
    base_dev = [simulate(r, "span", cal, params, use_lid) for r in dev]
    base_acc = summarize([row_counts(r, o) for r, (o, _) in zip(dev, base_dev)], ci=False)
    dev_rt["none"] = {**summarize_rt([rec for _, rec in base_dev]), "word_lang_acc": base_acc["word_lang_acc"]}
    for n in FREEZE_GRID:
        sims = [simulate(r, "span", cal, params, use_lid, freeze=n) for r in dev]
        acc = summarize([row_counts(r, o) for r, (o, _) in zip(dev, sims)], ci=False)
        dev_rt[str(n)] = {**summarize_rt([rec for _, rec in sims]), "word_lang_acc": acc["word_lang_acc"]}
        print(f"DEV freeze {n}: {dev_rt[str(n)]}", flush=True)
    ok_n = [n for n in FREEZE_GRID if dev_rt[str(n)]["word_lang_acc"] >= base_acc["word_lang_acc"] - FREEZE_MAX_ACC_DROP]
    freeze_n = min(ok_n, key=lambda n: dev_rt[str(n)]["flicker_per_word"]) if ok_n else None
    print(f"freeze chosen on DEV: {freeze_n}", flush=True)

    test_hyps = [cb.final_hyps(r) for r in test]
    from common import hyp_features

    test_probs = [cb.predict(cal, hyp_features(h, use_lid)) for h in test_hyps]
    methods = {
        "shipped_utterance_vote": score_rows(test, test_hyps, test_probs, "shipped"),
        "shipped_vote_retagged_by_lid": score_rows(test, test_hyps, test_probs, "shipped_retagged"),
        "raw_span_maxconf": score_rows(test, test_hyps, test_probs, "span", RAW),
        **({"raw_span_maxconf_en_handicap": score_rows(test, test_hyps, test_probs, "span", RAW_HANDICAP)} if WITH_EN else {}),
        "span_vote_calibrated_hmm": score_rows(test, test_hyps, test_probs, "span", params),
        "span_vote_calibrated_no_hmm": score_rows(test, test_hyps, test_probs, "span", Params(switch=0.0, missing=params.missing, pause=None, bias=params.bias)),
    }
    groups = groups_of(test)
    offline = {m: {g: summarize([c[i] for i in idx]) for g, idx in groups.items()} for m, c in methods.items()}
    for m, res in offline.items():
        a = res["all"]
        print(f"TEST {m:32} lang-acc {a['word_lang_acc']} {a['word_lang_acc_ci']} WER {a['wer']} {a['wer_ci']} "
              f"kk {a.get('word_lang_acc_kk')} ru {a.get('word_lang_acc_ru')} EN recall {a.get('en_recall')} "
              f"no-EN acc {res['no_en']['word_lang_acc']} {res['no_en']['word_lang_acc_ci']} WER {res['no_en']['wer']}", flush=True)

    rt_methods = {"shipped_live": ("shipped", None, None), "raw_span_maxconf_live": ("span", RAW, None),
                  "span_vote_live": ("span", params, None), f"span_vote_live_freeze_{freeze_n}": ("span", params, freeze_n)}
    realtime = {}
    for name, (method, p, fz) in rt_methods.items():
        sims = [simulate(r, method, cal, p, use_lid, freeze=fz) for r in test]
        realtime[name] = {}
        for g, idx in groups.items():
            acc = summarize([row_counts(test[i], sims[i][0]) for i in idx])
            realtime[name][g] = {**summarize_rt([sims[i][1] for i in idx]),
                                 **{k: acc[k] for k in ("word_lang_acc", "word_lang_acc_ci", "word_lang_acc_kk", "word_lang_acc_ru", "wer", "en_recall", "kkru_to_en_rate") if k in acc}}
        print(f"TEST realtime {name}: {realtime[name]['all']}", flush=True)

    result = {
        "langs": list(LANGS),
        "dev": {"rows": len(dev), "calibration_oof_log_loss": cal_report, "features": feat_choice,
                "tuned": {"switch": params.switch, "missing": params.missing, "pause": params.pause, "bias": params.bias},
                "tuned_dev_metrics": dev_best, "grid": {k: list(v) for k, v in GRID.items()}, "freeze": dev_rt,
                "freeze_chosen_s": freeze_n},
        "js_port": {"features": list(names), "calibrators": cb.raw_space(cal, names), "switch_penalty": params.switch,
                    "missing_logit": params.missing, "pause_reset_s": params.pause, "bias_logit": params.bias,
                    "freeze_s": freeze_n, "conf_clip": 1e-3, "min_duration_s": 0.03},
        "test": {"rows": len(test), "offline": offline, "realtime": realtime},
        "seconds": round(time.time() - started, 1),
    }
    OUT.write_text(json.dumps(result, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {OUT}", flush=True)


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    main()
