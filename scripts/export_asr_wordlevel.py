"""Export the word-level ASR models for the browser and their Python↔JS golden fixtures (ADR D64).

    python scripts/export_asr_wordlevel.py

Writes
- `site/core/asr-wordlevel.json`: the per-word language tagger (`word_lid.py`: char-trigram LMs +
  word-frequency lexicons + Viterbi, fitted on the `train` split only) and the calibrated span
  vote's parameters (`spanvote_results.json → js_port`: 13 features, three logistic calibrators in
  raw feature space, switch penalty, missing logit, freeze). The span vote's char-trigram feature
  reads the same LMs, so one file serves both modules.
- `tests_js/fixtures/word_lid.json`: tags for real GigaAM transcripts (code-switch + FLEURS decodes)
  and hand-made edge cases, from the spike's `WordLID.tag` — `site/core/word-lid.js` must match.
- `tests_js/fixtures/span_vote.json`: the live replay of DEV rows (`spanvote_dev_decodes.*`, built
  from `train` text) through the spike's features → raw-space calibrators → `span_vote` → 1.0 s
  freeze loop (`realtime.simulate`), step by step — `site/core/span-vote.js` must match.

Reference = the spike code in `scripts/spikes/asr_wordlevel/`. The only deviation, on purpose:
calibrated probabilities are computed from the exported raw-space coefficients (rounded to 5
decimals, as the browser gets them), not from the scikit-learn objects.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from glob import glob
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SPIKE = REPO / "scripts/spikes/asr_wordlevel"
sys.path[:0] = [str(SPIKE), str(SPIKE / "spanvote")]

import word_lid as wl  # noqa: E402
from common import LANGS, clean_words, hyp_features, hypothesis_at  # noqa: E402
from vote import Params, span_vote  # noqa: E402

FORMAT_VERSION = 1
MODEL_OUT = REPO / "site/core/asr-wordlevel.json"
LID_FIXTURE = REPO / "tests_js/fixtures/word_lid.json"
VOTE_FIXTURE = REPO / "tests_js/fixtures/span_vote.json"
RESULTS = REPO / "data/asr_capture/spanvote_results.json"
CS_DECODES = REPO / "data/asr_capture/cs_decodes.jsonl"
FLEURS_DECODES = REPO / "data/asr_capture/fleurs_decodes_clean.jsonl"
DEV_DECODES = "data/asr_capture/spanvote_dev_decodes.*.jsonl"
LEXICON_VOCAB = 50000  # word_lid.WordLID.emission's smoothing denominator
N_CS, N_FLEURS = 40, 45
N_DEV_EN, N_DEV_OTHER = 20, 20
EDGE_CASES = [
    "", "мен", "мен банк картасы", "код келді ме", "назовите код из смс", "алло hello сәлем",
    "банктің службы безопасности қызметкерімін", "ваша карта заблокирована", "мына кодты айтыңыз",
    "anydesk орнатыңыз", "ok", "123 456", "қ", "е", "the code please",
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def export_lid(model: wl.WordLID) -> dict:
    return {
        "order": wl.ORDER,
        "switch_penalty": model.switch_penalty,
        "lexicon_weight": model.lexicon_weight,
        "lexicon_vocab": LEXICON_VOCAB,
        "hard": wl.HARD,
        "kazakh_only": "".join(sorted(wl.KAZAKH_ONLY)),
        "char_lms": {
            lang: {
                "alphabet_size": lm.alphabet_size,
                "counts": {ctx: dict(sorted(c.items())) for ctx, c in sorted(lm.counts.items())},
            }
            for lang, lm in model.char_lms.items()
        },
        "lexicons": {lang: dict(sorted(lex.items())) for lang, lex in model.lexicons.items()},
    }


def raw_predict(calibrators: dict, names: list, feats: dict) -> dict:
    """p = sigmoid(intercept + Σ coef·x) per word, per language (the browser's arithmetic)."""
    out = {}
    for lang in LANGS:
        cal = calibrators[lang]
        coef = [cal["coef"][n] for n in names]
        rows = feats[lang]
        probs = []
        for row in (rows if len(rows) and len(rows[0]) else []):
            z = cal["intercept"] + sum(c * float(x) for c, x in zip(coef, row))
            probs.append(1.0 / (1.0 + math.exp(-z)))
        out[lang] = probs
    return out


def lid_fixture(model: wl.WordLID) -> list[dict]:
    texts = list(EDGE_CASES)
    rows = [json.loads(line) for line in CS_DECODES.read_text(encoding="utf-8").splitlines() if line.strip()]
    texts += [r["gigaam220"]["text"] for r in rows if r.get("gigaam220")][:N_CS]
    if FLEURS_DECODES.exists():
        per_lang: dict[str, list[str]] = {}
        for line in FLEURS_DECODES.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            lang = r["id"].split("_")[1]
            if r.get("gigaam220") and len(per_lang.setdefault(lang, [])) < N_FLEURS // 3:
                per_lang[lang].append(r["gigaam220"]["text"])
        texts += [t for lang in sorted(per_lang) for t in per_lang[lang]]
    cases = []
    for text in texts:
        words = wl.normalize(text).split()
        cases.append({"text": text, "words": words, "tags": model.tag(words)})
    return cases


def dev_rows() -> list[dict]:
    decodes = {}
    for path in sorted(glob(str(REPO / DEV_DECODES))):
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if line.strip():
                d = json.loads(line)
                decodes[(d["id"], d["condition"], d["channel"])] = d
    manifest = [json.loads(line) for line in (REPO / "data/asr_capture/spanvote_dev_set.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    with_en, without = [], []
    for m in manifest:
        d = decodes.get((m["id"], m["condition"], m["channel"]))
        if d is None:
            continue
        has_en = any(w["lang"] == "en" for w in m["words"])
        bucket = with_en if has_en else without
        if len(bucket) < (N_DEV_EN if has_en else N_DEV_OTHER):
            bucket.append(d)
    return with_en + without


def replay(row: dict, model: wl.WordLID, port: dict) -> dict:
    """realtime.simulate(method='span', freeze=port['freeze_s']) with raw-space calibrators."""
    params = Params(switch=port["switch_penalty"], missing=port["missing_logit"], pause=port["pause_reset_s"],
                    bias=port["bias_logit"])
    times = sorted({e["t"] for l in LANGS for k in ("finals", "partials") for e in row["models"][l][k]}
                   | {round(row["duration"], 3)})
    frozen, frozen_until, steps = [], 0.0, []
    for t in times:
        raw = {l: hypothesis_at(row["models"][l], t) for l in LANGS}
        hyps = {l: clean_words(raw[l]) for l in LANGS}
        feats = hyp_features(hyps, model)
        probs = raw_predict(port["calibrators"], port["features"], {l: feats[l].tolist() for l in LANGS})
        words, labelled = span_vote(hyps, probs, params, list(frozen))
        steps.append({
            "t": t, "hyps": raw, "probs": probs,
            "words": [[w, s, e, l] for w, s, e, l in words],
            "labelled": [[a, b, l] for a, b, l in labelled],
        })
        newest = max((w[2] for h in hyps.values() for w in h), default=0.0)
        until = max(frozen_until, newest - port["freeze_s"])
        for a, b, l in labelled:
            lo, hi = max(a, frozen_until), min(b, until)
            if hi > lo:
                frozen.append((lo, hi, l))
        frozen_until = until
        steps[-1]["frozen_after"] = [[a, b, l] for a, b, l in frozen]
    return {"id": row["id"], "condition": row["condition"], "channel": row["channel"], "steps": steps}


def main() -> None:
    model = wl.fitted()
    port = json.loads(RESULTS.read_text(encoding="utf-8"))["js_port"]
    bundle = {
        "format_version": FORMAT_VERSION,
        "provenance": {
            "word_lid": "scripts/spikes/asr_wordlevel/word_lid.py fitted() on the train split",
            "train_sha256": sha256(REPO / "data/processed/train.jsonl"),
            "span_vote": "data/asr_capture/spanvote_results.json js_port",
            "span_vote_sha256": sha256(RESULTS),
        },
        "lid": export_lid(model),
        "span_vote": {k: port[k] for k in ("features", "calibrators", "switch_penalty", "missing_logit",
                                            "pause_reset_s", "bias_logit", "freeze_s", "conf_clip", "min_duration_s")},
    }
    MODEL_OUT.write_text(json.dumps(bundle, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
    LID_FIXTURE.write_text(json.dumps({"cases": lid_fixture(model)}, ensure_ascii=False, indent=0) + "\n", encoding="utf-8")
    rows = [replay(r, model, port) for r in dev_rows()]
    VOTE_FIXTURE.write_text(json.dumps({"rows": rows}, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
    for path in (MODEL_OUT, LID_FIXTURE, VOTE_FIXTURE):
        print(f"{path.relative_to(REPO)}: {path.stat().st_size / 1e3:.0f} kB")
    print(f"word_lid: switch_penalty={model.switch_penalty} lexicon_weight={model.lexicon_weight}; "
          f"span_vote rows={len(rows)} steps={sum(len(r['steps']) for r in rows)}")


if __name__ == "__main__":
    main()
