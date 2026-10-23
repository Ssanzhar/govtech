"""GigaAM's words inherit the live span vote's language tags (ADR D70, the endpoint pass).

GigaAM's text replaces the Vosk words at the endpoint and used to be tagged from scratch by the word
LID, so words the person had just seen tagged correctly could change language. Here:
1. GigaAM's words are aligned to the live (span-vote) words by a word-level edit distance whose
   substitution cost is 1 − character similarity (normalised Levenshtein); a pair counts only when
   the similarity reaches `min_sim`;
2. the word LID's Viterbi runs over GigaAM's words with a hint: for a paired word, every language
   other than the live tag pays `weight` nats. Letters still decide first (a Kazakh-only letter is
   kk whatever the hint), and unpaired words follow the LID and their neighbours.

`min_sim` and `weight` are chosen on DEV (GigaAM decodes of DEV: `decode_dev_gigaam.py`), then TEST
is scored once:

    python scripts/spikes/asr_wordlevel/spanvote/inherit.py   # -> data/asr_capture/inherit_results.json

`site/core/word-align.js` + `word-lid.js tagWithHints` are the 1:1 port (fixture: export_asr_wordlevel.py).
"""

from __future__ import annotations

import itertools
import json
import sys
import time

from common import REPO, load, normalize
from metrics import row_counts, summarize

sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "scripts/spikes/asr_wordlevel"))

OUT = REPO / "data/asr_capture/inherit_results.json"
DEV_GIGAAM = REPO / "data/asr_capture/spanvote_dev_gigaam.jsonl"
TEST_GIGAAM = REPO / "data/asr_capture/cs_decodes.jsonl"
GRID = {"min_sim": (0.34, 0.5, 0.67), "weight": (1.0, 2.0, 4.0, 8.0, 16.0, 32.0)}


def levenshtein(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def similarity(a: str, b: str) -> float:
    """1 − Levenshtein / longer length, in [0, 1]; two empty strings are identical."""
    longest = max(len(a), len(b))
    return 1.0 if longest == 0 else 1.0 - levenshtein(a, b) / longest


def align_words(new: list[str], old: list[str]) -> list[tuple[int, int, float]]:
    """Minimal-cost word alignment -> pairs (new index, old index, similarity). Substitution costs
    1 − similarity, insertion/deletion 1; ties prefer a substitution, then skipping a new word."""
    n, m = len(new), len(old)
    sims = [[similarity(a, b) for b in old] for a in new]
    cost = [[0.0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        cost[i][0] = float(i)
    for j in range(1, m + 1):
        cost[0][j] = float(j)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost[i][j] = min(cost[i - 1][j - 1] + 1.0 - sims[i - 1][j - 1], cost[i - 1][j] + 1.0, cost[i][j - 1] + 1.0)
    pairs, i, j = [], n, m
    while i > 0 and j > 0:
        if cost[i][j] == cost[i - 1][j - 1] + 1.0 - sims[i - 1][j - 1]:
            pairs.append((i - 1, j - 1, sims[i - 1][j - 1]))
            i, j = i - 1, j - 1
        elif cost[i][j] == cost[i - 1][j] + 1.0:
            i -= 1
        else:
            j -= 1
    return pairs[::-1]


def hints_for(new: list[str], live: list[tuple[str, str]], min_sim: float) -> list[str | None]:
    """The live tag each new word inherits (None: no confident partner)."""
    hints: list[str | None] = [None] * len(new)
    for i, j, sim in align_words(new, [w for w, _ in live]):
        if sim >= min_sim:
            hints[i] = live[j][1]
    return hints


def tag_with_hints(lid, words: list[str], hints: list[str | None], weight: float) -> list[str]:
    """WordLID.tag with a per-word penalty `weight` on every language but the hint."""
    from word_lid import LANGS

    def em(word, hint, lang):
        return lid.emission(word, lang) - (weight if hint is not None and lang != hint else 0.0)

    if not words:
        return []
    scores = {l: em(words[0], hints[0], l) for l in LANGS}
    back: list[dict] = []
    for word, hint in zip(words[1:], hints[1:]):
        step, ptr = {}, {}
        for l in LANGS:
            prev = max(LANGS, key=lambda p: scores[p] - (0 if p == l else lid.switch_penalty))
            step[l] = scores[prev] - (0 if prev == l else lid.switch_penalty) + em(word, hint, l)
            ptr[l] = prev
        scores, back = step, back + [ptr]
    best = max(LANGS, key=lambda l: scores[l])
    path = [best]
    for ptr in reversed(back):
        path.append(ptr[path[-1]])
    return path[::-1]


def live_final_words(row: dict, model, port: dict) -> list[tuple[str, str]]:
    """What the page commits at the endpoint: the span vote's last replay step (ADR D70)."""
    from export_asr_wordlevel import replay

    steps = replay(row, model, port)["steps"]
    return [(w[0], w[3]) for w in steps[-1]["words"]] if steps else []


def gigaam_texts(path) -> dict:
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            if r.get("gigaam220"):
                out[(r["id"], r["condition"], r["channel"])] = r["gigaam220"]["text"]
    return out


def prepare(split: str, model, port: dict) -> list[dict]:
    texts = gigaam_texts(DEV_GIGAAM if split == "dev" else TEST_GIGAAM)
    rows = []
    for row in load(split):
        text = texts.get((row["id"], row["condition"], row["channel"]))
        if text is None:
            continue
        rows.append({**row, "giga": normalize(text).split(), "live": live_final_words(row, model, port)})
    return rows


def outputs(rows, model, method, min_sim=None, weight=None):
    out = []
    for r in rows:
        if method == "lid_only":
            tags = model.tag(r["giga"])
        else:
            tags = tag_with_hints(model, r["giga"], hints_for(r["giga"], r["live"], min_sim), weight)
        out.append([(w, 0.0, 0.0, t) for w, t in zip(r["giga"], tags)])
    return out


def flip_rate(rows, outs) -> float:
    """Share of GigaAM words paired (similarity ≥ 0.5) with a live word whose final tag differs."""
    paired = flipped = 0
    for r, o in zip(rows, outs):
        for i, j, sim in align_words(r["giga"], [w for w, _ in r["live"]]):
            if sim >= 0.5:
                paired += 1
                flipped += o[i][3] != r["live"][j][1]
    return round(flipped / paired, 4) if paired else float("nan")


def score(rows, outs, ci=True) -> dict:
    groups = {"all": list(range(len(rows))), "no_en": [i for i, r in enumerate(rows) if "en" not in r["ref_langs"]]}
    res = {g: summarize([row_counts(rows[i], outs[i]) for i in idx], ci=ci) for g, idx in groups.items()}
    res["endpoint_flip_rate"] = flip_rate(rows, outs)
    return res


def main() -> None:
    import word_lid as wl

    started = time.time()
    model = wl.fitted()
    port = json.loads((REPO / "data/asr_capture/spanvote_results_kkru.json").read_text(encoding="utf-8"))["js_port"]
    dev = [r for r in prepare("dev", model, port) if "en" not in r["ref_langs"]]
    print(f"DEV rows with GigaAM decodes (kk/ru speech): {len(dev)}", flush=True)
    dev_lid = score(dev, outputs(dev, model, "lid_only"), ci=False)
    print("DEV lid_only", dev_lid["all"]["word_lang_acc"], "flips", dev_lid["endpoint_flip_rate"], flush=True)
    grid = {}
    for min_sim, weight in itertools.product(*GRID.values()):
        s = score(dev, outputs(dev, model, "inherit", min_sim, weight), ci=False)
        grid[f"{min_sim}/{weight}"] = {"word_lang_acc": s["all"]["word_lang_acc"], "flips": s["endpoint_flip_rate"]}
        print(f"DEV inherit min_sim={min_sim} weight={weight}: {grid[f'{min_sim}/{weight}']}", flush=True)
    best = max(grid, key=lambda k: (grid[k]["word_lang_acc"], -grid[k]["flips"]))
    min_sim, weight = map(float, best.split("/"))
    print(f"chosen on DEV: min_sim={min_sim} weight={weight}", flush=True)

    test = prepare("test", model, port)
    print(f"TEST rows with GigaAM decodes: {len(test)}", flush=True)
    result = {
        "dev": {"rows": len(dev), "lid_only": dev_lid["all"], "grid": grid, "chosen": {"min_sim": min_sim, "weight": weight}},
        "test": {
            "rows": len(test),
            "gigaam_lid_only": score(test, outputs(test, model, "lid_only")),
            "gigaam_inherit": score(test, outputs(test, model, "inherit", min_sim, weight)),
        },
        "js_port": {"min_sim": min_sim, "weight": weight},
        "seconds": round(time.time() - started, 1),
    }
    for name in ("gigaam_lid_only", "gigaam_inherit"):
        r = result["test"][name]
        print(f"TEST {name}: acc {r['all']['word_lang_acc']} {r['all'].get('word_lang_acc_ci')} kk {r['all'].get('word_lang_acc_kk')} "
              f"ru {r['all'].get('word_lang_acc_ru')} | no-EN {r['no_en']['word_lang_acc']} {r['no_en'].get('word_lang_acc_ci')} "
              f"WER {r['no_en']['wer']} | endpoint flips {r['endpoint_flip_rate']}", flush=True)
    OUT.write_text(json.dumps(result, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
