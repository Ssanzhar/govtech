"""Per-word language tags for a transcript from a joint kk+ru+en recogniser (word-level STT track).

The letters decide when they can: a Latin letter -> en, a Kazakh-only letter (әғқңөұүһі) -> kk.
Plain-Cyrillic words (мен, бе, келген, код ...) are scored by two character-trigram language
models (kk, ru) blended with a word-frequency lexicon, and the utterance is decoded with Viterbi
under a language-switch penalty, so a lone ambiguous word follows its neighbours.

Trained on the `train` split only (never on a held-out split or on the code-switched test set).
The switch penalty and the lexicon weight are tuned on a synthetic dev set built from `train`
text alone: fragments of kk and ru train utterances spliced into one word sequence, each word
labelled with its source language.

    python scripts/spikes/asr_wordlevel/word_lid.py --tune       # fit + tune, prints dev accuracy
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
TRAIN = REPO / "data/processed/train.jsonl"
LANGS = ("kk", "ru", "en")
CYRILLIC_LANGS = ("kk", "ru")
KAZAKH_ONLY = set("әғқңөұүһі")
ORDER = 3
SEED = 20261008
HARD = 1e9  # a letter-decided word cannot take another language


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFC", text).lower().replace("ё", "е")
    text = re.sub(r"[^\w\s']", " ", text)
    return re.sub(r"\s+", " ", text.replace("_", " ")).strip()


def letter_language(word: str) -> str | None:
    if re.search(r"[a-z]", word):
        return "en"
    if any(c in KAZAKH_ONLY for c in word):
        return "kk"
    return None


@dataclass(frozen=True)
class CharLM:
    counts: dict  # context -> Counter(next char)
    alphabet_size: int

    @classmethod
    def fit(cls, words: list[str]) -> "CharLM":
        counts: dict[str, Counter] = {}
        alphabet = set()
        for w in words:
            padded = "^" * (ORDER - 1) + w + "$"
            alphabet.update(padded)
            for i in range(ORDER - 1, len(padded)):
                counts.setdefault(padded[i - ORDER + 1:i], Counter())[padded[i]] += 1
        return cls(counts=counts, alphabet_size=len(alphabet) + 1)

    def logprob(self, word: str) -> float:
        padded = "^" * (ORDER - 1) + word + "$"
        total = 0.0
        for i in range(ORDER - 1, len(padded)):
            ctx = self.counts.get(padded[i - ORDER + 1:i], Counter())
            total += math.log((ctx[padded[i]] + 0.1) / (sum(ctx.values()) + 0.1 * self.alphabet_size))
        return total


@dataclass(frozen=True)
class WordLID:
    char_lms: dict
    lexicons: dict  # lang -> Counter
    switch_penalty: float = 4.0
    lexicon_weight: float = 1.0

    @classmethod
    def fit(cls, texts_by_lang: dict[str, list[str]], **kw) -> "WordLID":
        words = {l: [w for t in texts for w in normalize(t).split() if letter_language(w) != "en"] for l, texts in texts_by_lang.items()}
        return cls(char_lms={l: CharLM.fit(words[l]) for l in CYRILLIC_LANGS},
                   lexicons={l: Counter(words[l]) for l in CYRILLIC_LANGS}, **kw)

    def emission(self, word: str, lang: str) -> float:
        decided = letter_language(word)
        if decided:
            return 0.0 if decided == lang else -HARD
        if lang == "en":
            return -HARD
        lex = self.lexicons[lang]
        lex_lp = math.log((lex[word] + 0.5) / (sum(lex.values()) + 0.5 * 50000))
        return self.char_lms[lang].logprob(word) + self.lexicon_weight * lex_lp

    def tag(self, words: list[str]) -> list[str]:
        if not words:
            return []
        scores = {l: self.emission(words[0], l) for l in LANGS}
        back: list[dict] = []
        for word in words[1:]:
            step, ptr = {}, {}
            for l in LANGS:
                prev = max(LANGS, key=lambda p: scores[p] - (0 if p == l else self.switch_penalty))
                step[l] = scores[prev] - (0 if prev == l else self.switch_penalty) + self.emission(word, l)
                ptr[l] = prev
            scores, back = step, back + [ptr]
        best = max(LANGS, key=lambda l: scores[l])
        path = [best]
        for ptr in reversed(back):
            path.append(ptr[path[-1]])
        return path[::-1]

    def with_params(self, switch_penalty: float, lexicon_weight: float) -> "WordLID":
        return WordLID(self.char_lms, self.lexicons, switch_penalty, lexicon_weight)


def train_texts(holdout_every: int = 5) -> tuple[dict, dict]:
    """(fit texts, dev texts) per language from the `train` split; every 5th dialogue is dev."""
    fit, dev = {l: [] for l in CYRILLIC_LANGS}, {l: [] for l in CYRILLIC_LANGS}
    for i, line in enumerate(TRAIN.read_text(encoding="utf-8").splitlines()):
        row = json.loads(line)
        if row["language"] not in CYRILLIC_LANGS:
            continue
        target = dev if i % holdout_every == 0 else fit
        target[row["language"]] += [u["text"] for u in row["utterances"]]
    return fit, dev


def synthetic_dev(dev: dict, n: int = 600) -> list[tuple[list[str], list[str]]]:
    rng = random.Random(SEED)
    rows = []
    for _ in range(n):
        matrix, other = rng.sample(CYRILLIC_LANGS, 2)
        a, b, c = (normalize(rng.choice(dev[l])).split() for l in (matrix, other, matrix))
        parts = [(a[: rng.randint(1, 5)], matrix), (b[: rng.randint(1, 3)], other), (c[: rng.randint(0, 4)], matrix)]
        words = [w for p, _ in parts for w in p]
        langs = [l for p, l in parts for _ in p]
        rows.append((words, langs))
    return rows


def accuracy(model: WordLID, rows) -> float:
    ok = total = 0
    for words, langs in rows:
        ok += sum(p == g for p, g in zip(model.tag(words), langs))
        total += len(words)
    return ok / total


def fitted(tune: bool = True) -> WordLID:
    fit, dev = train_texts()
    model = WordLID.fit(fit)
    if not tune:
        return model
    rows = synthetic_dev(dev)
    grid = [(s, w) for s in (0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0) for w in (0.0, 0.5, 1.0, 2.0)]
    best = max(grid, key=lambda p: accuracy(model.with_params(*p), rows))
    return model.with_params(*best)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tune", action="store_true")
    parser.parse_args()
    fit, dev = train_texts()
    base = WordLID.fit(fit)
    rows = synthetic_dev(dev)
    letters_only = sum(sum((letter_language(w) or "ru") == g for w, g in zip(ws, ls)) for ws, ls in rows) / sum(len(ws) for ws, _ in rows)
    print(f"synthetic dev: {len(rows)} spliced kk/ru sequences from train text")
    print(f"letters only (plain Cyrillic -> ru): {letters_only:.3f}")
    model = fitted()
    print(f"char-LM + lexicon + Viterbi: {accuracy(model, rows):.3f} "
          f"(switch_penalty={model.switch_penalty}, lexicon_weight={model.lexicon_weight})")
    print(f"no context (switch_penalty=0): {accuracy(model.with_params(0.0, model.lexicon_weight), rows):.3f}")


if __name__ == "__main__":
    main()
