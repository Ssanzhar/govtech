"""Dry-run evaluation of a taxonomy-v2 retrain on VALIDATION ONLY (ADR D63 phase 2).

    QORGAN_EMBED_BACKEND=device QORGAN_LABEL_SPACE=v2 \\
      python scripts/taxonomy_v2_dryrun_eval.py --processed-dir data/processed_v2_dryrun \\
        --model-dir models/linear_v2_dryrun [--baseline site/models/weights.json]

Reads `val.jsonl` and `provenance.json` of the dry-run corpus and nothing else: no held-out
split is opened (held-out evaluation waits for the new independent held-out set). Reports
FPR first, then recall, each with a Clopper-Pearson 95 % interval, overall, by generator,
by generator x language, and per-tactic F1 on labelled rows. Whole-transcript scoring at the
shipped alert threshold, risk head + tactic head exactly as `predict._linear_score` computes
them (minus the per-utterance highlight pass, which does not change risk or tags).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import numpy as np  # noqa: E402

from qorgan.config import V2_NEW_TACTICS, get_config  # noqa: E402
from qorgan.eval.intervals import binomial_interval  # noqa: E402
from qorgan.eval.run import load_split  # noqa: E402


def _rate(k: int, n: int) -> dict:
    ci = binomial_interval(k, n)
    return {"k": k, "n": n, "rate": round(k / n, 4) if n else None,
            "ci95": [round(ci.low, 4), round(ci.high, 4)] if ci else None}


def _metrics(truth: np.ndarray, alert: np.ndarray) -> dict:
    neg, pos = truth == 0, truth == 1
    return {"fpr": _rate(int((alert & neg).sum()), int(neg.sum())), "recall": _rate(int((alert & pos).sum()), int(pos.sum()))}


def score(bundle, dialogues) -> tuple[np.ndarray, np.ndarray]:
    """(risk per row, tactic probabilities (n, L))."""
    from qorgan.classifier import features as feat

    texts = [d.transcript() for d in dialogues]
    if bundle.hard_signal_enabled:
        blocks = feat.compute_feature_blocks(texts, model_name=bundle.embed_model_name, lexicon=bundle.lexicon,
                                             reassurance_patterns=bundle.reassurance_patterns)
        risk = bundle.risk_clf.predict_proba(feat.hybrid_matrix(blocks))[:, 1]
        tactic = bundle.tactic_clf.predict_proba(blocks.embedding)
    else:
        from qorgan.classifier.embed import embed_texts

        emb = embed_texts(texts, model_name=bundle.embed_model_name)
        risk, tactic = bundle.risk_clf.predict_proba(emb)[:, 1], bundle.tactic_clf.predict_proba(emb)
    return risk, tactic


def _aucs(truth: np.ndarray, risk: np.ndarray) -> dict:
    from sklearn.metrics import average_precision_score, roc_auc_score

    if truth.min() == truth.max():
        return {}
    return {"roc_auc": round(float(roc_auc_score(truth, risk)), 4), "pr_auc": round(float(average_precision_score(truth, risk)), 4)}


def auc_breakdown(dialogues, generator, truth, risk) -> dict:
    out = {"overall": _aucs(truth, risk)}
    for g in ("claude", "gemini"):
        idx = np.array([i for i, d in enumerate(dialogues) if generator.get(d.id, "gemini") == g])
        out[f"generator={g}"] = _aucs(truth[idx], risk[idx])
    return out


def rule_threshold(truth: np.ndarray, risk: np.ndarray, max_false_positives: int) -> float | None:
    """The LOWEST threshold on a 0.01 grid whose false positives <= `max_false_positives`
    (ADR D63 phase 2 operating-point rule, decided before the held-out run)."""
    for step in range(1, 100):
        t = round(step / 100, 2)
        if int(((risk >= t) & (truth == 0)).sum()) <= max_false_positives:
            return t
    return None


def false_positive_table(bundle, dialogues, generator, risk, tactic, threshold) -> list[dict]:
    from qorgan.classifier.features import match_cues

    rows = []
    for i, d in enumerate(dialogues):
        if d.label.is_scam or risk[i] < threshold:
            continue
        top = sorted(zip(bundle.label_space, tactic[i]), key=lambda kv: -kv[1])[:2]
        rows.append({"id": d.id, "generator": generator.get(d.id, "gemini"), "language": d.language, "risk": round(float(risk[i]), 3),
                     "hard_cue": bool(match_cues(d.transcript(), bundle.lexicon)) if bundle.lexicon else None,
                     "top_heads": [f"{t}:{p:.2f}" for t, p in top]})
    return rows


def _breakdown(dialogues, generator, truth, alert) -> dict:
    out = {"overall": _metrics(truth, alert)}
    groups: dict[str, list[int]] = {}
    for i, d in enumerate(dialogues):
        g = generator.get(d.id, "gemini")
        groups.setdefault(f"generator={g}", []).append(i)
        groups.setdefault(f"generator={g} language={d.language}", []).append(i)
    for key in sorted(groups):
        idx = np.array(groups[key])
        out[key] = _metrics(truth[idx], alert[idx])
    return out


def per_tactic(bundle, dialogues, generator, tactic_probs, labelled=None) -> dict:
    """Per-tactic F1 on labelled rows only. `labelled` is the trainer's `(n, L)` mask (D73: by each
    row's label space); without it, the D63 rule (the v2 five only on Claude rows)."""
    from qorgan.classifier.labels import encode_tactics

    out = {}
    for j, tactic in enumerate(bundle.label_space):
        threshold = bundle.tactic_thresholds.get(tactic, bundle.tactic_threshold)
        if labelled is not None:
            rows = [i for i in range(len(dialogues)) if labelled[i, j]]
        else:
            rows = [i for i, d in enumerate(dialogues) if tactic not in V2_NEW_TACTICS or generator.get(d.id) == "claude"]
        truth = np.array([encode_tactics([t.id for t in dialogues[i].label.tactic_tags], bundle.label_space)[j] for i in rows])
        pred = tactic_probs[rows, j] >= threshold
        tp, fp, fn = int((pred & (truth == 1)).sum()), int((pred & (truth == 0)).sum()), int((~pred & (truth == 1)).sum())
        out[tactic] = {"support": int(truth.sum()), "rows": len(rows), "threshold": threshold,
                       "precision": round(tp / (tp + fp), 3) if tp + fp else None,
                       "recall": round(tp / (tp + fn), 3) if tp + fn else None,
                       "f1": round(2 * tp / (2 * tp + fp + fn), 3) if (2 * tp + fp + fn) else None}
    return out


def main() -> None:
    from qorgan.classifier.linear_train import load_linear
    from qorgan.classifier.web_bundle import load_linear_from_web

    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, default=None, help="shipped weights.json to score on the same val rows")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    threshold = get_config().risk_threshold
    val = list(load_split(args.processed_dir, "val"))
    provenance = json.loads((args.processed_dir / "provenance.json").read_text(encoding="utf-8"))
    generator = provenance["generator"]
    truth = np.array([int(d.label.is_scam) for d in val])
    # D75: a fused browser weights file is scored exactly as the server would serve it.
    bundle = load_linear_from_web(args.model_dir) if args.model_dir.suffix == ".json" else load_linear(args.model_dir)
    risk, tactic = score(bundle, val)
    labelled = None
    if "row_label_space" in provenance:  # D73: mask exactly as the trainer did
        from qorgan.classifier.linear_train import label_space_mask
        from qorgan.config import LABEL_SPACE_NEW_TACTICS, LABEL_SPACE_ORDER

        labelled = label_space_mask(val, bundle.label_space, provenance["row_label_space"],
                                    new_tactics_by_space=LABEL_SPACE_NEW_TACTICS, order=LABEL_SPACE_ORDER)
    report = {"split": "val", "threshold": threshold, "label_space": list(bundle.label_space),
              "v2_dryrun": _breakdown(val, generator, truth, risk >= threshold),
              "per_tactic": per_tactic(bundle, val, generator, tactic, labelled)}
    report["v2_auc"] = auc_breakdown(val, generator, truth, risk)
    report["v2_false_positives"] = false_positive_table(bundle, val, generator, risk, tactic, threshold)
    if args.baseline:
        base = load_linear_from_web(args.baseline)
        base_risk, _ = score(base, val)
        report["shipped_v1_baseline"] = _breakdown(val, generator, truth, base_risk >= threshold)
        report["shipped_v1_auc"] = auc_breakdown(val, generator, truth, base_risk)
        base_fp = int(((base_risk >= threshold) & (truth == 0)).sum())
        rule = rule_threshold(truth, risk, base_fp)
        report["operating_point_rule"] = {"baseline_false_positives": base_fp, "enter": rule,
                                          "exit": None if rule is None else round(rule - 0.10, 2)}
        if rule is not None:
            report["v2_at_rule"] = _breakdown(val, generator, truth, risk >= rule)
    out = args.out or args.model_dir / "dryrun_val_report.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: report.get(k) for k in ("v2_auc", "shipped_v1_auc", "operating_point_rule")}, indent=1))
    for name in ("v2_dryrun", "shipped_v1_baseline", "v2_at_rule"):
        if name in report:
            print(f"== {name} (val, threshold {threshold})")
            for key, m in report[name].items():
                f, r = m["fpr"], m["recall"]
                print(f"  {key:40s} FPR {f['k']}/{f['n']} {f['rate']} {f['ci95']}   recall {r['k']}/{r['n']} {r['rate']} {r['ci95']}")
    print(f"report -> {out}")


if __name__ == "__main__":
    main()
