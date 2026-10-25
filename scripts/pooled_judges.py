"""The shipped model on every independent local judge at once (ADR D76, track E).

v1 was frozen before any `shift_local*` batch existed and nothing was tuned on them for it,
so for v1 they are one independent sample. Each batch was spent on judging a retrain; that
does not touch v1's numbers. Pooled kk + ru only (the voice languages, D70): `shift_local`'s
English rows are left out.

    QORGAN_CLASSIFIER_BACKEND=linear QORGAN_EMBED_BACKEND=device QORGAN_LINEAR_WEIGHTS=web \\
      python scripts/pooled_judges.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from qorgan.classifier.predict import score  # noqa: E402
from qorgan.config import get_config  # noqa: E402
from qorgan.eval.intervals import binomial_interval  # noqa: E402
from qorgan.eval.run import evaluate_by_language, evaluate_split, format_report, load_split  # noqa: E402

JUDGES = ("shift_local", "shift_local_2", "shift_local_3", "shift_local_4", "shift_local_5", "shift_local_6")
LANGUAGES = ("kk", "ru")


def _counts(name: str, dialogues, risks, threshold: float) -> str:
    neg = [r for d, r in zip(dialogues, risks) if not d.label.is_scam]
    pos = [r for d, r in zip(dialogues, risks) if d.label.is_scam]
    fp, tp = sum(r >= threshold for r in neg), sum(r >= threshold for r in pos)
    fci, rci = binomial_interval(fp, len(neg)), binomial_interval(tp, len(pos))
    return (f"{name:12s} false alarms {fp}/{len(neg)} [{fci.low:.3f}, {fci.high:.3f}]   "
            f"scams caught {tp}/{len(pos)} [{rci.low:.3f}, {rci.high:.3f}]")


def main() -> None:
    cfg = get_config()
    processed = cfg.data_dir / "processed"
    pooled = tuple(d for name in JUDGES for d in load_split(processed, name) if d.language in LANGUAGES)
    cache: dict[str, float] = {}

    def score_fn(transcript: str):
        result = score(transcript)
        cache[transcript] = result.risk
        return result

    threshold = cfg.risk_threshold
    results = {"shift_local_pooled (kk+ru)": evaluate_split(pooled, score_fn=score_fn, alert_threshold=threshold)}
    results.update({f"pooled {lang}": r for lang, r in evaluate_by_language(pooled, score_fn=score_fn, alert_threshold=threshold).items()})
    print(f"backend={cfg.classifier_backend} weights={cfg.linear_weights_source} threshold={threshold} judges={','.join(JUDGES)}")
    print(format_report(results))
    risks = [cache[d.transcript()] for d in pooled]
    print(_counts("kk+ru", pooled, risks, threshold))
    for lang in LANGUAGES:
        rows = [(d, r) for d, r in zip(pooled, risks) if d.language == lang]
        print(_counts(lang, [d for d, _ in rows], [r for _, r in rows], threshold))


if __name__ == "__main__":
    main()
