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


def per_tactic(bundle, dialogues, generator, tactic_probs) -> dict:
    from qorgan.classifier.labels import encode_tactics

    out = {}
    for j, tactic in enumerate(bundle.label_space):
        threshold = bundle.tactic_thresholds.get(tactic, bundle.tactic_threshold)
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
    generator = json.loads((args.processed_dir / "provenance.json").read_text(encoding="utf-8"))["generator"]
    truth = np.array([int(d.label.is_scam) for d in val])
    bundle = load_linear(args.model_dir)
    risk, tactic = score(bundle, val)
    report = {"split": "val", "threshold": threshold, "label_space": list(bundle.label_space),
              "v2_dryrun": _breakdown(val, generator, truth, risk >= threshold),
              "per_tactic": per_tactic(bundle, val, generator, tactic)}
    if args.baseline:
        base = load_linear_from_web(args.baseline)
        base_risk, _ = score(base, val)
        report["shipped_v1_baseline"] = _breakdown(val, generator, truth, base_risk >= threshold)
    out = args.out or args.model_dir / "dryrun_val_report.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    for name in ("v2_dryrun", "shipped_v1_baseline"):
        if name in report:
            print(f"== {name} (val, threshold {threshold})")
            for key, m in report[name].items():
                f, r = m["fpr"], m["recall"]
                print(f"  {key:40s} FPR {f['k']}/{f['n']} {f['rate']} {f['ci95']}   recall {r['k']}/{r['n']} {r['rate']} {r['ci95']}")
    print(f"report -> {out}")


if __name__ == "__main__":
    main()
