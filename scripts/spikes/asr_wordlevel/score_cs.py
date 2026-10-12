"""Score the code-switched decodes (word-level STT track).

Per engine and per (condition, channel):
- WER / CER after one shared normalisation, with a 95 % bootstrap interval over utterances;
- WER attributed to the reference words of each language (kk / ru / en): hits, substitutions and
  deletions are charged to the reference word's language; insertions are reported separately;
- per-word language-tag accuracy on aligned (hit or substituted) words, where a hypothesis word is
  tagged from its letters only: any Latin letter -> en, any Kazakh-only letter -> kk, otherwise ru.
  Plain-Cyrillic Kazakh words are therefore tagged ru -- that floor is the point of the number;
  the same words are also tagged by `word_lid.py` (char-LM + lexicon + Viterbi, train split only);
- hard-signal cue survival: tactics the shipped matcher finds in the reference that it still
  finds in the hypothesis, and tactics it finds only in the hypothesis (false fires);
- speed: decode seconds / audio seconds (real-time factor) for engines that report time.

    python scripts/spikes/asr_wordlevel/score_cs.py [--json out.json]
"""

from __future__ import annotations

import argparse
import sys
import json
import re
import unicodedata
from collections import defaultdict
from pathlib import Path

import jiwer
import numpy as np

from word_lid import fitted

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
from qorgan.classifier.cue_match import find_cue  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
MANIFEST = REPO / "data/asr_capture/cs_set.jsonl"
DECODES = REPO / "data/asr_capture/cs_decodes.jsonl"
ENGINES = ("vosk", "gigaam220", "gigaam600")
KAZAKH_ONLY = set("әғқңөұүһі")
CUES = REPO / "data/lexicon/hard_signal_cues.yaml"
BOOTSTRAP = 1000
SEED = 20261008


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFC", text).lower().replace("ё", "е")
    text = re.sub(r"[^\w\s']", " ", text)
    return re.sub(r"\s+", " ", text.replace("_", " ")).strip()


def tag_word(word: str) -> str:
    if re.search(r"[a-z]", word):
        return "en"
    return "kk" if any(c in KAZAKH_ONLY for c in word) else "ru"


def align(ref_words: list[str], hyp_words: list[str]):
    """(op, ref_index | None, hyp_index | None) triples from jiwer's alignment."""
    if not hyp_words:
        return [("delete", i, None) for i in range(len(ref_words))]
    out = jiwer.process_words(" ".join(ref_words), " ".join(hyp_words))
    pairs = []
    for chunk in out.alignments[0]:
        if chunk.type in ("equal", "substitute"):
            for k in range(chunk.ref_end_idx - chunk.ref_start_idx):
                pairs.append(("hit" if chunk.type == "equal" else "sub", chunk.ref_start_idx + k, chunk.hyp_start_idx + k))
        elif chunk.type == "delete":
            pairs += [("delete", i, None) for i in range(chunk.ref_start_idx, chunk.ref_end_idx)]
        else:
            pairs += [("insert", None, j) for j in range(chunk.hyp_start_idx, chunk.hyp_end_idx)]
    return pairs


def load_cues() -> dict[str, list[str]]:
    import yaml

    return yaml.safe_load(CUES.read_text(encoding="utf-8"))["cues"]


def cue_tactics(text: str, cues: dict[str, list[str]]) -> set[str]:
    """Hard-signal tactics the shipped bounded-edit matcher finds (ADR D39)."""
    return {t for t, cue_list in cues.items() if any(find_cue(text, c) for c in cue_list)}


def bootstrap(errors: np.ndarray, totals: np.ndarray) -> tuple[float, float]:
    rng = np.random.default_rng(SEED)
    picks = rng.integers(0, len(errors), size=(BOOTSTRAP, len(errors)))
    rates = errors[picks].sum(axis=1) / np.maximum(totals[picks].sum(axis=1), 1)
    return float(np.percentile(rates, 2.5)), float(np.percentile(rates, 97.5))


def score_group(rows: list[tuple[dict, dict]], engine: str, lid, cues) -> dict:
    errs, tots, char_err, char_tot = [], [], 0, 0
    by_lang = defaultdict(lambda: {"ref": 0, "err": 0})
    tag_ok = defaultdict(lambda: [0, 0])
    lid_ok = defaultdict(lambda: [0, 0])
    insertions, seconds, audio = 0, 0.0, 0.0
    cue_ref = cue_kept = cue_false = 0
    for ref_row, dec in rows:
        ref_words = [normalize(w["word"]) for w in ref_row["words"]]
        langs = [w["lang"] for w, n in zip(ref_row["words"], ref_words) if n]
        ref_words = [n for n in ref_words if n]
        hyp = normalize(dec[engine]["text"])
        hyp_words = hyp.split()
        pairs = align(ref_words, hyp_words)
        lid_tags = lid.tag(hyp_words)
        ref_cues, hyp_cues = cue_tactics(ref_row["reference"], cues), cue_tactics(dec[engine]["text"], cues)
        cue_ref += len(ref_cues)
        cue_kept += len(ref_cues & hyp_cues)
        cue_false += len(hyp_cues - ref_cues)
        n_err = sum(op != "hit" for op, _, _ in pairs)
        errs.append(n_err)
        tots.append(len(ref_words))
        cer = jiwer.process_characters(" ".join(ref_words), hyp or " ")
        char_err += cer.substitutions + cer.deletions + cer.insertions
        char_tot += cer.substitutions + cer.deletions + cer.hits
        for op, ri, hi in pairs:
            if ri is None:
                insertions += 1
                continue
            lang = langs[ri]
            by_lang[lang]["ref"] += 1
            by_lang[lang]["err"] += op != "hit"
            if hi is not None:
                tag_ok[lang][0] += tag_word(hyp_words[hi]) == lang
                tag_ok[lang][1] += 1
                lid_ok[lang][0] += lid_tags[hi] == lang
                lid_ok[lang][1] += 1
        if "seconds" in dec[engine]:
            seconds += dec[engine]["seconds"]
            audio += dec["duration"]
    errs_a, tots_a = np.array(errs), np.array(tots)
    low, high = bootstrap(errs_a, tots_a)
    return {
        "n": len(rows), "wer": errs_a.sum() / tots_a.sum(), "wer_ci": [low, high],
        "cer": char_err / max(char_tot, 1), "insertions": insertions,
        "wer_by_lang": {l: v["err"] / v["ref"] for l, v in sorted(by_lang.items())},
        "words_by_lang": {l: v["ref"] for l, v in sorted(by_lang.items())},
        "tag_acc_by_lang": {l: ok / n for l, (ok, n) in sorted(tag_ok.items()) if n},
        "tag_acc": sum(v[0] for v in tag_ok.values()) / max(sum(v[1] for v in tag_ok.values()), 1),
        "lid_acc_by_lang": {l: ok / n for l, (ok, n) in sorted(lid_ok.items()) if n},
        "lid_acc": sum(v[0] for v in lid_ok.values()) / max(sum(v[1] for v in lid_ok.values()), 1),
        "cues_in_reference": cue_ref, "cues_kept": cue_kept, "cue_false_fires": cue_false,
        "rtf_1thread": seconds / audio if audio else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()
    manifest = {(r["id"], r["condition"], r["channel"]): r for r in map(json.loads, MANIFEST.read_text(encoding="utf-8").splitlines())}
    decodes = [json.loads(l) for l in DECODES.read_text(encoding="utf-8").splitlines() if l.strip()]
    groups = defaultdict(list)
    for dec in decodes:
        ref = manifest[(dec["id"], dec["condition"], dec["channel"])]
        groups[(dec["condition"], dec["channel"])].append((ref, dec))
    lid = fitted()
    cues = load_cues()
    report = {}
    for key in sorted(groups):
        for engine in ENGINES:
            if engine not in groups[key][0][1]:
                continue
            s = score_group(groups[key], engine, lid, cues)
            report[f"{key[0]}/{key[1]}/{engine}"] = s
            lang = " ".join(f"{l} {v:.3f}" for l, v in s["wer_by_lang"].items())
            tags = " ".join(f"{l} {v:.3f}" for l, v in s["lid_acc_by_lang"].items())
            rtf = f" rtf {s['rtf_1thread']:.3f}" if s["rtf_1thread"] else ""
            print(f"{key[0]:8} {key[1]:6} {engine:10} n={s['n']:3} WER {s['wer']:.3f} [{s['wer_ci'][0]:.3f}, {s['wer_ci'][1]:.3f}] "
                  f"CER {s['cer']:.3f} | WER by lang: {lang} | tags: letters {s['tag_acc']:.3f}, tagger {s['lid_acc']:.3f} ({tags})| cues kept {s['cues_kept']}/{s['cues_in_reference']}, false {s['cue_false_fires']}{rtf}")
    if args.json:
        args.json.write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
