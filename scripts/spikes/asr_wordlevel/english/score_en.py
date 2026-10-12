"""Score the English recogniser candidates (english/decode_en.py output).

Per engine:
- WER / CER on `authored` (clean, phone, both), `cs_span`, `fleurs_en`, one shared English
  normalisation (lowercase, digits -> words, hyphens split, punctuation dropped, a few GB->US
  spellings), corpus-level, 95 % bootstrap interval over utterances;
- confidence the engine exposes (Vosk mean word `conf`; sherpa mean token probability), its
  distribution on English vs on the kk/ru probe, and the ROC AUC of "this is English" from it;
- for Vosk-family English models: the shipped per-utterance vote (decode_cs.vote_utterance, 0.15
  handicap) with the kk/ru Vosk models -> share of probe utterances English wins (false win) and
  share of English utterances it wins (true win);
- probe garbage: share of kk/ru utterances with any English output, words per second emitted;
- whether per-word confidence and timestamps exist; native single-thread RTF; size on disk.

    .venv/bin/python scripts/spikes/asr_wordlevel/english/score_en.py
"""

from __future__ import annotations

import json
import re
import sys
import unicodedata
from pathlib import Path

import jiwer
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from decode_cs import vote_utterance  # noqa: E402

REPO = HERE.parents[3]
CAPTURE = REPO / "data/asr_capture"
MANIFESTS = (CAPTURE / "english_set.jsonl", CAPTURE / "english_fleurs.jsonl")
OUT = CAPTURE / "english_results.json"
ENGINES = ("vosk_small", "vosk_lgraph", "fc_rnnt480", "fc_rnnt480_fp32", "fc_ctc480", "moon_tiny", "moon_base", "parakeet110", "gigaam220", "gigaam600")
VOSK_EN = ("vosk_small", "vosk_lgraph")
EN_SETS = ("authored", "cs_span", "fleurs_en")
PROBE_SETS = ("probe_cs", "probe_fleurs_kk", "probe_fleurs_ru")
HANDICAPS = (0.0, 0.05, 0.10, 0.15, 0.25)
BOOTSTRAP = 1000
MOONSHINE_MAX_S = 9.0
OPERATING_TPR = 0.90
WASM_N = 60
SEED = 20261008
MODEL_DIRS = {
    "vosk_small": Path.home() / ".cache/vosk/vosk-model-small-en-us-0.15",
    "vosk_lgraph": REPO / "models/asr/vosk-model-en-us-0.22-lgraph",
    "fc_rnnt480": REPO / "models/asr/sherpa-onnx-nemo-streaming-fast-conformer-transducer-en-480ms-int8",
    "fc_rnnt480_fp32": REPO / "models/asr/sherpa-onnx-nemo-streaming-fast-conformer-transducer-en-480ms",
    "fc_ctc480": REPO / "models/asr/sherpa-onnx-nemo-streaming-fast-conformer-ctc-en-480ms-int8",
    "moon_tiny": REPO / "models/asr/sherpa-onnx-moonshine-tiny-en-quantized-2026-02-27",
    "moon_base": REPO / "models/asr/sherpa-onnx-moonshine-base-en-quantized-2026-02-27",
    "gigaam220": REPO / "models/asr/gigaam-ml/model.int8.onnx",
    "gigaam600": REPO / "models/asr/gigaam-ml/large/model.int8.onnx",
    "parakeet110": REPO / "models/asr/sherpa-onnx-nemo-parakeet_tdt_ctc_110m-en-36000-int8",
}
GB_TO_US = {"neighbour": "neighbor", "neighbours": "neighbors", "centre": "center", "colour": "color",
            "favourite": "favorite", "organisation": "organization", "behaviour": "behavior"}
ONES = "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen".split()
TENS = "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()


def number_words(n: int) -> str:
    if n < 20:
        return ONES[n]
    if n < 100:
        return TENS[n // 10] + ("" if n % 10 == 0 else " " + ONES[n % 10])
    if n < 1000:
        return ONES[n // 100] + " hundred" + ("" if n % 100 == 0 else " " + number_words(n % 100))
    if n < 1_000_000:
        return number_words(n // 1000) + " thousand" + ("" if n % 1000 == 0 else " " + number_words(n % 1000))
    return " ".join(ONES[int(d)] for d in str(n))


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower()
    text = text.replace("%", " percent ").replace("&", " and ").replace("-", " ").replace("’", "'")
    text = re.sub(r"(?<=\d),(?=\d{3})", "", text)
    text = re.sub(r"\d+", lambda m: f" {number_words(int(m.group()))} ", text)
    text = re.sub(r"[^\w\s']", " ", text)
    words = [w.strip("'") for w in text.replace("_", " ").split()]
    return " ".join(GB_TO_US.get(w, w) for w in words if w)


def join_split_words(ref: str, hyp: str) -> str:
    """Merge adjacent hypothesis words whose concatenation is a reference word ("any one" ->
    "anyone", "to morrow" -> "tomorrow"): the NeMo models' text normalisation splits such words.
    Applied to every engine alike, for the secondary `wer_join` figure only."""
    vocab, words, out, i = set(ref.split()), hyp.split(), [], 0
    while i < len(words):
        if i + 1 < len(words) and words[i] + words[i + 1] in vocab and words[i] not in vocab:
            out.append(words[i] + words[i + 1])
            i += 2
        else:
            out.append(words[i])
            i += 1
    return " ".join(out)


def join_errors(ref: str, hyp: str) -> int:
    r, h = normalize(ref), join_split_words(normalize(ref), normalize(hyp))
    if not r:
        return 0
    if not h:
        return len(r.split())
    w = jiwer.process_words(r, h)
    return w.substitutions + w.deletions + w.insertions


def counts(ref: str, hyp: str) -> tuple[int, int, int, int]:
    """(word errors, ref words, char errors, ref chars)."""
    r, h = normalize(ref), normalize(hyp)
    if not r:
        return 0, 0, 0, 0
    if not h:
        return len(r.split()), len(r.split()), len(r), len(r)
    w = jiwer.process_words(r, h)
    c = jiwer.process_characters(r, h)
    return (w.substitutions + w.deletions + w.insertions, len(r.split()),
            c.substitutions + c.deletions + c.insertions, len(r))


def rate_ci(errors: np.ndarray, totals: np.ndarray, rng: np.random.Generator) -> dict:
    n = len(errors)
    point = errors.sum() / totals.sum()
    idx = rng.integers(0, n, size=(BOOTSTRAP, n))
    boots = errors[idx].sum(axis=1) / totals[idx].sum(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"rate": round(float(point), 4), "ci95": [round(float(lo), 4), round(float(hi), 4)], "n": n}


def auc(pos: list[float], neg: list[float]) -> float | None:
    if not pos or not neg:
        return None
    p, q = np.asarray(pos), np.asarray(neg)
    greater = (p[:, None] > q[None, :]).sum() + 0.5 * (p[:, None] == q[None, :]).sum()
    return round(float(greater / (len(p) * len(q))), 4)


def WASM_ROWS(rows: dict) -> list[str]:  # noqa: N802 -- the rows bench_wasm.cjs times
    return [i for i, r in rows.items() if r["set"] == "authored" and r["channel"] == "phone"][:WASM_N]


def rtf(decodes: list[dict]) -> float | None:
    return round(sum(d["seconds"] for d in decodes) / sum(d["duration"] for d in decodes), 4) if decodes else None


def decode_files(engine: str) -> list[Path]:
    """The engine's decode file plus its `--shard` files (not other engines sharing a prefix)."""
    base = REPO / "data/asr_capture"
    return sorted([base / f"english_decodes_{engine}.jsonl", *base.glob(f"english_decodes_{engine}__s*.jsonl")])


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()] if path.exists() else []


def disk_mb(path: Path) -> float:
    if path.is_file():
        return round(path.stat().st_size / 1e6, 1)
    files = [f for f in path.rglob("*") if f.is_file() and "test_wavs" not in f.parts and f.name not in ("LICENSE", "README")]
    return round(sum(f.stat().st_size for f in files) / 1e6, 1)


def latin_share(decodes: list[dict]) -> float | None:
    """Share of hypothesis words written with Latin letters (GigaAM may transliterate English)."""
    words = [w for d in decodes for w in d["text"].split() if re.search(r"[^\W\d_]", w)]
    return round(sum(bool(re.search(r"[a-zA-Z]", w)) for w in words) / len(words), 4) if words else None


def accuracy_block(rows: dict, decodes: dict, rng) -> dict:
    out = {}
    groups = {s: [r for r in rows.values() if r["set"] == s] for s in EN_SETS}
    groups["authored_clean"] = [r for r in groups["authored"] if r["channel"] == "clean"]
    groups["authored_phone"] = [r for r in groups["authored"] if r["channel"] == "phone"]
    groups["cs_span_phone"] = [r for r in groups["cs_span"] if r["channel"] == "phone"]
    # Moonshine (sherpa export) returns nothing for inputs longer than ~9 s, so also score the
    # FLEURS rows every engine can take whole.
    groups["fleurs_en_le9s"] = [r for r in groups["fleurs_en"] if r["duration"] <= MOONSHINE_MAX_S]
    for name, members in groups.items():
        members = [r for r in members if r["id"] in decodes]
        if not members:
            continue
        c = np.array([counts(r["reference"], decodes[r["id"]]["text"]) for r in members], dtype=float)
        j = np.array([join_errors(r["reference"], decodes[r["id"]]["text"]) for r in members], dtype=float)
        out[name] = {"wer": rate_ci(c[:, 0], c[:, 1], rng), "cer": rate_ci(c[:, 2], c[:, 3], rng),
                     "wer_join": rate_ci(j, c[:, 1], rng)}
    return out


def confidence_block(rows: dict, decodes: dict) -> dict:
    def confs(sets):
        return [d["confidence"] for i, d in decodes.items() if rows[i]["set"] in sets and d.get("confidence") is not None and d["text"]]

    en, probe = confs(EN_SETS), confs(PROBE_SETS)
    probe_rows = [d for i, d in decodes.items() if rows[i]["set"] in PROBE_SETS]
    block = {"exposed": bool(en)}
    if en:
        block |= {"en_mean": round(float(np.mean(en)), 4), "en_p10_p50_p90": [round(float(x), 4) for x in np.percentile(en, [10, 50, 90])]}
    if probe:
        block |= {"probe_mean": round(float(np.mean(probe)), 4), "probe_p10_p50_p90": [round(float(x), 4) for x in np.percentile(probe, [10, 50, 90])]}
    block["auc_en_vs_probe"] = auc(en, probe)
    if en and probe:
        threshold = float(np.percentile(en, 100 * (1 - OPERATING_TPR)))
        block["threshold_at_en_tpr90"] = round(threshold, 4)
        block["probe_pass_at_en_tpr90"] = round(float(np.mean(np.asarray(probe) >= threshold)), 4)
    if probe_rows:
        block["probe_nonempty_share"] = round(sum(bool(d["text"]) for d in probe_rows) / len(probe_rows), 4)
        block["probe_words_per_sec"] = round(sum(len(d["text"].split()) for d in probe_rows) / sum(d["duration"] for d in probe_rows), 3)
    return block


def vote_block(rows: dict, en: dict, kk: dict, ru: dict) -> dict:
    import decode_cs

    out = {}
    for handicap in HANDICAPS:
        decode_cs.EN_HANDICAP = handicap
        per_set = {}
        for s in EN_SETS + PROBE_SETS:
            ids = [i for i, r in rows.items() if r["set"] == s and i in en and i in kk and i in ru]
            if not ids:
                continue
            wins = sum(vote_utterance({"kk": kk[i], "ru": ru[i], "en": en[i]}) == "en" for i in ids)
            per_set[s] = {"en_wins": round(wins / len(ids), 4), "n": len(ids)}
        out[f"handicap_{handicap}"] = per_set
    decode_cs.EN_HANDICAP = 0.15
    return out


def main() -> None:
    rng = np.random.default_rng(SEED)
    rows = {r["id"]: r for m in MANIFESTS for r in load_jsonl(m)}
    kk = {d["id"]: d for d in load_jsonl(CAPTURE / "english_decodes_vosk_kk.jsonl")}
    ru = {d["id"]: d for d in load_jsonl(CAPTURE / "english_decodes_vosk_ru.jsonl")}
    results = {"sets": {s: sum(r["set"] == s for r in rows.values()) for s in EN_SETS + PROBE_SETS}, "engines": {}}
    for engine in ENGINES:
        decodes = {d["id"]: d for path in decode_files(engine)
                   for d in load_jsonl(path) if d["id"] in rows}
        if not decodes:
            continue
        sample = next(iter(decodes.values()))
        has_words = "words" in sample
        block = {
            "decoded": len(decodes),
            "accuracy": accuracy_block(rows, decodes, rng),
            "confidence": confidence_block(rows, decodes),
            "per_word_conf": has_words or any(d.get("logprobs") for d in decodes.values()),
            "timestamps": has_words or any(d.get("timestamps") for d in decodes.values()),
            "rtf_native_1t": round(sum(d["seconds"] for d in decodes.values()) / sum(d["duration"] for d in decodes.values()), 4),
            "rtf_native_1t_fleurs": rtf([d for d in decodes.values() if d["set"] == "fleurs_en"]),
            "rtf_native_1t_wasm_rows": rtf([decodes[i] for i in WASM_ROWS(rows) if i in decodes]),
            "size_mb": disk_mb(MODEL_DIRS[engine]),
            "latin_word_share": {g: latin_share([d for d in decodes.values() if d["set"] in sets])
                                 for g, sets in (("authored", ("authored",)), ("cs_span", ("cs_span",)),
                                                 ("fleurs_en", ("fleurs_en",)), ("probe", PROBE_SETS))},
        }
        if engine in VOSK_EN and kk and ru:
            block["vote"] = vote_block(rows, decodes, kk, ru)
        results["engines"][engine] = block
    OUT.write_text(json.dumps(results, indent=1, ensure_ascii=False), encoding="utf-8")
    print_table(results)


def print_table(results: dict) -> None:
    print("sets:", results["sets"])
    keys = ("authored_clean", "authored_phone", "cs_span", "fleurs_en", "fleurs_en_le9s")
    print(f"{'engine':12} " + " ".join(f"{k:>24}" for k in keys) + "   RTF    MB  AUC   probeConf enConf")
    for engine, b in results["engines"].items():
        cells = []
        for k in keys:
            a = b["accuracy"].get(k)
            cells.append(f"{a['wer']['rate']:.3f} [{a['wer']['ci95'][0]:.3f},{a['wer']['ci95'][1]:.3f}]" if a else "-")
        c = b["confidence"]
        print(f"{engine:12} " + " ".join(f"{x:>24}" for x in cells) +
              f" {b['rtf_native_1t']:.3f} {b['size_mb']:5.0f} {c.get('auc_en_vs_probe')} {c.get('probe_mean')} {c.get('en_mean')}")
        print(f"    wer_join: " + ", ".join(f"{k}={b['accuracy'][k]['wer_join']['rate']:.3f}" for k in keys if k in b["accuracy"]))
        print(f"    cer: " + ", ".join(f"{k}={b['accuracy'][k]['cer']['rate']:.3f}" for k in keys if k in b["accuracy"]) +
              f" | probe nonempty={c.get('probe_nonempty_share')} words/s={c.get('probe_words_per_sec')}"
              f" | probe pass @EN TPR90={c.get('probe_pass_at_en_tpr90')} | rtf fleurs={b['rtf_native_1t_fleurs']}"
              f" wasm-rows={b['rtf_native_1t_wasm_rows']} | word conf={b['per_word_conf']} ts={b['timestamps']}"
              f" | latin share={b['latin_word_share']}")
        if "vote" in b:
            for h, v in b["vote"].items():
                print(f"    vote {h}: " + ", ".join(f"{s}={x['en_wins']:.3f}/{x['n']}" for s, x in v.items()))


if __name__ == "__main__":
    main()
