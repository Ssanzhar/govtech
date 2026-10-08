"""Per-model calibrators: P(word is right AND in the right language) from cheap word features.

Target for recogniser L's word: it aligns as a hit to a reference word whose language is L.
Fitted on DEV finals only; out-of-fold DEV predictions (GroupKFold by sequence id) feed tuning so
the switch penalty is not tuned on in-sample probabilities.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from common import LANGS, align, clean_words, final_words, hyp_features

FOLDS = 5
C_REG = 1.0


def final_hyps(row: dict) -> dict:
    return {l: clean_words(final_words(row["models"][l])) for l in LANGS}


def table(rows: list[dict], lid=None):
    """Per language: X, y, group, (row index, word index)."""
    out = {l: {"X": [], "y": [], "g": [], "where": []} for l in LANGS}
    for ri, row in enumerate(rows):
        hyps = final_hyps(row)
        feats = hyp_features(hyps, lid)
        for l in LANGS:
            if not hyps[l]:
                continue
            labels = np.zeros(len(hyps[l]))
            for op, r, h in align(row["ref_words"], [w[0] for w in hyps[l]]):
                if h is not None and op == "hit" and row["ref_langs"][r] == l:
                    labels[h] = 1.0
            out[l]["X"].append(feats[l])
            out[l]["y"].append(labels)
            out[l]["g"] += [row["id"]] * len(labels)
            out[l]["where"] += [(ri, i) for i in range(len(labels))]
    return {l: {"X": np.vstack(v["X"]), "y": np.concatenate(v["y"]), "g": np.array(v["g"]), "where": v["where"]}
            for l, v in out.items()}


def _model():
    return StandardScaler(), LogisticRegression(C=C_REG, max_iter=2000)


def fit(tab: dict) -> dict:
    models = {}
    for l in LANGS:
        scaler, lr = _model()
        lr.fit(scaler.fit_transform(tab[l]["X"]), tab[l]["y"])
        models[l] = (scaler, lr)
    return models


def oof_predictions(tab: dict) -> dict:
    """Out-of-fold probabilities per language, aligned with tab[l]['where']."""
    out = {}
    for l in LANGS:
        X, y, g = tab[l]["X"], tab[l]["y"], tab[l]["g"]
        p = np.zeros(len(y))
        for tr, te in GroupKFold(n_splits=FOLDS).split(X, y, g):
            scaler, lr = _model()
            lr.fit(scaler.fit_transform(X[tr]), y[tr])
            p[te] = lr.predict_proba(scaler.transform(X[te]))[:, 1]
        out[l] = p
    return out


def predict(models: dict, feats: dict) -> dict:
    return {l: (models[l][1].predict_proba(models[l][0].transform(feats[l]))[:, 1] if len(feats[l]) else np.zeros(0))
            for l in LANGS}


def raw_space(models: dict, names: tuple) -> dict:
    """Coefficients folded through the scaler: p = sigmoid(intercept + sum coef_i * x_i)."""
    out = {}
    for l, (scaler, lr) in models.items():
        scale = np.where(scaler.scale_ == 0, 1.0, scaler.scale_)
        coef = lr.coef_[0] / scale
        out[l] = {"intercept": round(float(lr.intercept_[0] - np.sum(coef * scaler.mean_)), 5),
                  "coef": {n: round(float(c), 5) for n, c in zip(names, coef)}}
    return out


def log_loss(y, p) -> float:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
