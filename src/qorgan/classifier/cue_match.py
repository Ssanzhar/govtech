"""Cue matching that survives the speech recogniser (Tier A, 2026-09-23).

The hard-signal lexicon is matched against a transcript that, in microphone mode, is
recogniser output. Exact substring matching breaks on errors that leave the phrase perfectly
recognisable to a reader — above all **moved word boundaries**, which are the recogniser's
most common mistake:

    reference   «назовите три цифры на обороте»
    hypothesis  «назовите три цифры наоборот де»

So a cue is searched twice: verbatim first, and only if that fails, as a bounded-edit
substring of the **de-spaced** normalised text, which makes a boundary shift cost nothing.

Two rules keep this safe:

* **An exact hit always wins.** Clean text — every training row, every evaluation split —
  therefore matches exactly as it did before, so the trained cue features do not move and a
  retrain is not forced by the algorithm alone. Only text where exact matching *failed* can
  gain a match.
* **Short cues get no fuzz at all.** `AnyDesk` → «не доски» is a real recogniser error, but
  no edit budget can rescue it without also matching unrelated words; a lexicon variant is
  the honest fix for those.
* **The utterance boundary is a wall.** De-spacing would otherwise let the tail of one turn
  and the head of the next concatenate into a cue neither contains — «…назовите три цифры
  на» + «обороте карты…» — so the fuzzy pass runs per utterance, never across `UTTERANCE_
  BOUNDARY`. (Found in review; measured cost on the corpus: zero dialogues, scam or negative,
  relied on bridging.) The verbatim pass still scans the whole text, which cannot bridge
  because no cue contains a newline.

Offsets are each runtime's own string indexing — code points in Python, UTF-16 code units in
JavaScript. For every character this system handles (Cyrillic, Kazakh, Latin, digits) those
coincide, and the golden fixture pins it; they would differ only for supplementary-plane
characters, which `normalize` drops in both languages anyway.

The matcher is versioned because the cue features are computed with it: a bundle trained
under one version must not be scored under another (`MATCHER_VERSION`, carried in the
bundle metadata). `site/core/lexicon.js` is a 1:1 port — change both, regenerate the golden
parity fixtures.
"""

from __future__ import annotations

import re
import unicodedata

# 1 = exact substring only (shipped until 2026-09-23); 2 = exact, then bounded-edit fallback.
MATCHER_VERSION = 2

# Utterances are joined with this before scoring (`config.window.join`); the fuzzy search
# treats it as a hard boundary so a cue can never be assembled out of two different turns.
UTTERANCE_BOUNDARY = "\n"

# Everything that is not a letter or digit is dropped, spaces included: the de-spaced form is
# what makes a moved word boundary free. Kazakh-specific letters are listed explicitly so a
# narrower `\w` locale can never silently drop them.
_KEEP = re.compile(r"[0-9a-zа-яёәғқңөұүһі]")

# Roughly one edit per ten characters, with nothing at all for short cues (see module docstring).
_BUDGET_STEPS = ((12, 0), (21, 1), (31, 2))
_MAX_BUDGET = 3


def budget_for(length: int) -> int:
    """Edit budget allowed for a cue of `length` normalised characters (monotonic)."""
    for limit, budget in _BUDGET_STEPS:
        if length < limit:
            return budget
    return _MAX_BUDGET


def normalize(text: str) -> tuple[str, list[int]]:
    """`(de-spaced normalised text, index map)`.

    Normalisation is NFKC + casefold + `ё`→`е`, keeping only letters and digits. `index_map[i]`
    is the offset in the ORIGINAL string of normalised character `i`, so a match can always be
    reported as a verbatim span of the text the user actually sees.
    """
    normalized: list[str] = []
    index_map: list[int] = []
    for index, character in enumerate(text):
        folded = unicodedata.normalize("NFKC", character).lower().replace("ё", "е")
        for piece in folded:
            if _KEEP.match(piece):
                normalized.append(piece)
                index_map.append(index)
    return "".join(normalized), index_map


def find_cue(text: str, cue: str, *, _prefilter: bool = True) -> tuple[int, int] | None:
    """Offsets `(start, end)` of `cue` in `text`, or `None`.

    Verbatim first (case-insensitive, so clean text is unchanged); then, for a cue long
    enough to earn a budget, the best bounded-edit occurrence in the de-spaced form.
    `_prefilter` is for tests only — turning it off must never change the answer.
    """
    if not text or not cue:
        return None

    lowered_text, lowered_cue = text.lower(), cue.lower()
    exact = lowered_text.find(lowered_cue)
    if exact != -1:
        return exact, exact + len(cue)

    normalized_cue, _ = normalize(cue)
    budget = budget_for(len(normalized_cue))
    if not normalized_cue or budget == 0:
        return None

    offset = 0
    for segment in text.split(UTTERANCE_BOUNDARY):
        found = _find_in_segment(segment, normalized_cue, budget, _prefilter)
        if found is not None:
            start, end = found
            return offset + start, offset + end
        offset += len(segment) + len(UTTERANCE_BOUNDARY)
    return None


def _find_in_segment(segment: str, normalized_cue: str, budget: int, prefilter: bool) -> tuple[int, int] | None:
    normalized_text, index_map = normalize(segment)
    if not normalized_text:
        return None
    if prefilter and not _may_match(normalized_cue, normalized_text, budget):
        return None
    window = _best_window(normalized_cue, normalized_text, budget)
    if window is None:
        return None
    start, end = window
    return index_map[start], index_map[end - 1] + 1


def _may_match(cue: str, text: str, budget: int) -> bool:
    """Pigeonhole prefilter: an alignment costing at most `budget` edits can damage at most
    `budget` of the cue's `budget + 1` disjoint blocks, so at least one block must appear in
    the text verbatim. Exact — it can only rule matches out — and it skips the quadratic
    search for almost every (cue, text) pair, which is what keeps the browser fast.
    """
    blocks = budget + 1
    size, remainder = divmod(len(cue), blocks)
    if size == 0:
        return True  # a cue shorter than its block count: no block is informative
    start = 0
    for index in range(blocks):
        end = start + size + (1 if index < remainder else 0)
        if cue[start:end] in text:
            return True
        start = end
    return False


def _best_window(cue: str, text: str, budget: int) -> tuple[int, int] | None:
    """Start/end of the lowest-cost occurrence of `cue` in `text` within `budget` edits.

    Levenshtein with a free start (row 0 is all zeros), carrying each cell's start offset so
    the winning alignment can be reported. Ties resolve substitution → deletion → insertion
    and then the earliest end, so Python and the JS port agree character for character.
    """
    cue_length, text_length = len(cue), len(text)
    costs = [0] * (text_length + 1)  # row 0: a match may start anywhere, at no cost
    starts = list(range(text_length + 1))

    for i in range(1, cue_length + 1):
        previous_costs, previous_starts = costs, starts
        costs = [i] + [0] * text_length
        starts = [0] * (text_length + 1)
        for j in range(1, text_length + 1):
            substitute = previous_costs[j - 1] + (0 if cue[i - 1] == text[j - 1] else 1)
            delete = previous_costs[j] + 1       # a cue character the recogniser dropped
            insert = costs[j - 1] + 1            # a character the recogniser added
            best = min(substitute, delete, insert)
            costs[j] = best
            if best == substitute:
                starts[j] = previous_starts[j - 1]
            elif best == delete:
                starts[j] = previous_starts[j]
            else:
                starts[j] = starts[j - 1]

    best_cost, best_end = budget + 1, -1
    for j in range(1, text_length + 1):
        if costs[j] < best_cost:
            best_cost, best_end = costs[j], j
    if best_end < 0 or best_cost > budget:
        return None
    start = starts[best_end]
    return (start, best_end) if best_end > start else None
