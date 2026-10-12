"""Score the real-speech decodes (FLEURS kk/ru/en + Tim2190 KK-RU code-switch clips).

Per (set, channel, engine), with the code-switch spike's normalisation, bootstrap and cue matcher
(`score_cs.py`) and its word tagger (`word_lid.py`):
- WER / CER (95 % bootstrap CI over utterances); for FLEURS also WER on the digit-free subset;
- `vosk_vote` = the shipped per-utterance vote; `vosk_oracle` = the Vosk model of the TRUE
  language alone (kk for the code-switch clips, whose matrix language is Kazakh);
- vote language choice: which model won vs the true language;
- script on en: share of hypothesis words written in Latin vs Cyrillic letters;
- hard-signal cue survival where the reference contains a cue;
- word tagger on kk / ru hypotheses: share of hypothesis words tagged with the utterance language;
- RTF single-thread native (Vosk vote = the three models summed; oracle = its one model);
  pooled and per-utterance median (the median is immune to a laptop sleeping mid-decode).

    python scripts/spikes/asr_wordlevel/fleurs/score.py   # -> data/asr_capture/fleurs_results.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import jiwer
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from score_cs import bootstrap, cue_tactics, load_cues, normalize  # noqa: E402
from word_lid import fitted  # noqa: E402

REPO = HERE.parents[3]
MANIFEST = REPO / "data/asr_capture/fleurs_set.jsonl"
DECODES = [REPO / f"data/asr_capture/fleurs_decodes_{c}.jsonl" for c in ("clean", "phone")]
OUT = REPO / "data/asr_capture/fleurs_results.json"
ENGINES = ("gigaam220", "gigaam600", "vosk_vote", "vosk_oracle")
EXAMPLES = 5


def hyp_of(dec: dict, engine: str, lang: str) -> tuple[str, float]:
    if engine == "vosk_vote":
        return dec["vosk"]["text"], dec["vosk"]["seconds"]
    if engine == "vosk_oracle":
        model = dec["vosk"]["per_model"]["kk" if lang == "kk+ru" else lang]
        return model["text"], model["seconds"]
    return dec[engine]["text"], dec[engine]["seconds"]


def word_errors(ref: str, hyp: str) -> tuple[int, int]:
    n_ref = len(ref.split())
    if not hyp:
        return n_ref, n_ref
    out = jiwer.process_words(ref, hyp)
    return out.substitutions + out.deletions + out.insertions, n_ref


def char_errors(ref: str, hyp: str) -> tuple[int, int]:
    out = jiwer.process_characters(ref, hyp or " ")
    return out.substitutions + out.deletions + out.insertions, out.substitutions + out.deletions + out.hits


def rate(errors: list[int], totals: list[int]) -> dict:
    if not totals:
        return {"n": 0}
    e, t = np.array(errors), np.array(totals)
    low, high = bootstrap(e, t)
    return {"n": len(t), "value": round(float(e.sum() / t.sum()), 4), "ci": [round(low, 4), round(high, 4)]}


def script_share(texts: list[str]) -> dict:
    words = [w for t in texts for w in normalize(t).split() if re.search(r"[^\W\d_]", w)]
    latin = sum(bool(re.search(r"[a-z]", w)) for w in words)
    cyr = sum(bool(re.search(r"[а-яәғқңөұүһі]", w)) for w in words)
    return {"words": len(words), "latin": round(latin / max(len(words), 1), 4), "cyrillic": round(cyr / max(len(words), 1), 4)}


def score_group(pairs: list[tuple[dict, dict]], engine: str, lid, cues) -> dict:
    lang = pairs[0][0]["lang"]
    w_err, w_tot, c_err, c_tot, nd_err, nd_tot = [], [], [], [], [], []
    wr_err, wr_tot = [], []
    seconds = audio = 0.0
    cue_ref = cue_kept = cue_false = 0
    tag_hits = tag_total = 0
    hyps, rtfs = [], []
    for ref_row, dec in pairs:
        text, secs = hyp_of(dec, engine, lang)
        ref, hyp = normalize(ref_row["reference"]), normalize(text)
        hyps.append(text)
        e, t = word_errors(ref, hyp)
        w_err.append(e), w_tot.append(t)
        ce, ct = char_errors(ref, hyp)
        c_err.append(ce), c_tot.append(ct)
        if not ref_row["has_digits"]:
            nd_err.append(e), nd_tot.append(t)
        if "reference_written" in ref_row:
            we, wt = word_errors(normalize(ref_row["reference_written"]), hyp)
            wr_err.append(we), wr_tot.append(wt)
        seconds, audio = seconds + secs, audio + dec["duration"]
        rtfs.append(secs / dec["duration"])
        ref_cues, hyp_cues = cue_tactics(ref_row["reference"], cues), cue_tactics(text, cues)
        cue_ref, cue_kept, cue_false = cue_ref + len(ref_cues), cue_kept + len(ref_cues & hyp_cues), cue_false + len(hyp_cues - ref_cues)
        if lang in ("kk", "ru") and hyp:
            tags = lid.tag(hyp.split())
            tag_hits, tag_total = tag_hits + sum(t == lang for t in tags), tag_total + len(tags)
    out = {"wer": rate(w_err, w_tot), "cer": rate(c_err, c_tot), "rtf_1thread": round(seconds / audio, 4), "rtf_median": round(float(np.median(rtfs)), 4),
           "audio_seconds": round(audio, 1),
           "cues": {"in_reference": cue_ref, "kept": cue_kept, "false_fires": cue_false}}
    if lang in ("kk", "ru", "en"):
        out["wer_digit_free"] = rate(nd_err, nd_tot)
    if wr_tot:
        out["wer_vs_written"] = rate(wr_err, wr_tot)
    if tag_total:
        out["tagger_share_true_lang"] = {"words": tag_total, "share": round(tag_hits / tag_total, 4)}
    if lang == "en":
        out["script"] = script_share(hyps)
    if engine == "vosk_vote":
        winners = Counter(dec["vosk"]["winner"] for _, dec in pairs)
        out["vote_winner"] = dict(winners)
        if lang in ("kk", "ru", "en"):
            out["vote_language_accuracy"] = round(winners[lang] / len(pairs), 4)
    return out


def examples(pairs: list[tuple[dict, dict]]) -> list[dict]:
    """The first EXAMPLES clips in id order (not picked): reference and every engine's text."""
    chosen = sorted(pairs, key=lambda p: int(p[0]["id"].rsplit("_", 1)[1]))[:EXAMPLES]
    return [{"id": r["id"], "reference": r["reference"], **{e: hyp_of(d, e, r["lang"])[0] for e in ENGINES}} for r, d in chosen]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path, default=OUT)
    args = parser.parse_args()
    manifest = {(r["id"], r["channel"]): r for r in map(json.loads, MANIFEST.read_text(encoding="utf-8").splitlines())}
    groups = defaultdict(list)
    for path in DECODES:
        for dec in map(json.loads, path.read_text(encoding="utf-8").splitlines()):
            ref = manifest[(dec["id"], dec["channel"])]
            name = "tim_cs" if ref["source"] == "tim" else f"fleurs_{ref['lang']}"
            groups[(name, dec["channel"])].append((ref, dec))
    lid, cues = fitted(), load_cues()
    report = {"groups": {}, "digits": {}, "examples": {}}
    for name, channel in sorted(groups):
        pairs = groups[(name, channel)]
        if channel == "clean":
            report["digits"][name] = {"rows": len(pairs), "with_digits": sum(r["has_digits"] for r, _ in pairs)}
        for engine in ENGINES:
            s = score_group(pairs, engine, lid, cues)
            report["groups"][f"{name}/{channel}/{engine}"] = s
            extra = ""
            if "wer_digit_free" in s:
                extra += f" | no-digit WER {s['wer_digit_free']['value']:.3f} (n={s['wer_digit_free']['n']})"
            if "wer_vs_written" in s:
                extra += f" | WER vs written {s['wer_vs_written']['value']:.3f} {s['wer_vs_written']['ci']}"
            if "vote_language_accuracy" in s:
                extra += f" | vote lang acc {s['vote_language_accuracy']:.3f} {s['vote_winner']}"
            if "script" in s:
                extra += f" | latin {s['script']['latin']:.3f} cyr {s['script']['cyrillic']:.3f}"
            if "tagger_share_true_lang" in s:
                extra += f" | tagger {s['tagger_share_true_lang']['share']:.3f}"
            c = s["cues"]
            print(f"{name:10} {channel:5} {engine:11} n={s['wer']['n']:3} WER {s['wer']['value']:.3f} {s['wer']['ci']} "
                  f"CER {s['cer']['value']:.3f} {s['cer']['ci']} RTF {s['rtf_1thread']:.3f} (median {s['rtf_median']:.3f}) "
                  f"cues {c['kept']}/{c['in_reference']} false {c['false_fires']}{extra}")
        if name == "tim_cs":
            report["examples"][channel] = examples(pairs)
    print("digits:", report["digits"])
    args.json.write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
