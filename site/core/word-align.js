/* Which live word each of GigaAM's words replaces (ADR D70), so the endpoint pass can keep the
   language tags the person already saw instead of re-tagging the line from scratch.

   A 1:1 port of `scripts/spikes/asr_wordlevel/spanvote/inherit.py` (align_words, hints_for),
   pinned by tests_js/fixtures/word_align.json (`scripts/export_asr_wordlevel.py`):
   a word-level edit distance whose substitution costs 1 − character similarity (normalised
   Levenshtein) and whose insertion/deletion costs 1; ties prefer a substitution, then skipping a
   new word. A pair passes its live tag on only when the similarity reaches `minSim`. Pure. */

import { normalizeWords } from "./word-lid.js";

/** Levenshtein distance between two strings (by code point). */
export function levenshtein(a, b) {
  const x = [...a];
  const y = [...b];
  let prev = Array.from({ length: y.length + 1 }, (_, j) => j);
  for (let i = 1; i <= x.length; i++) {
    const cur = [i];
    for (let j = 1; j <= y.length; j++) cur.push(Math.min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x[i - 1] !== y[j - 1] ? 1 : 0)));
    prev = cur;
  }
  return prev[y.length];
}

/** 1 − Levenshtein / longer length, in [0, 1]; two empty strings are identical. */
export function similarity(a, b) {
  const longest = Math.max([...a].length, [...b].length);
  return longest === 0 ? 1 : 1 - levenshtein(a, b) / longest;
}

/** Minimal-cost alignment of `next` words to `prev` words -> `[[nextIndex, prevIndex, sim]]`. */
export function alignWords(next, prev) {
  const n = next.length;
  const m = prev.length;
  const sims = next.map((a) => prev.map((b) => similarity(a, b)));
  const cost = Array.from({ length: n + 1 }, (_, i) => Array.from({ length: m + 1 }, (_, j) => (i === 0 ? j : j === 0 ? i : 0)));
  for (let i = 1; i <= n; i++) {
    for (let j = 1; j <= m; j++) {
      cost[i][j] = Math.min(cost[i - 1][j - 1] + 1 - sims[i - 1][j - 1], cost[i - 1][j] + 1, cost[i][j - 1] + 1);
    }
  }
  const pairs = [];
  let i = n;
  let j = m;
  while (i > 0 && j > 0) {
    if (cost[i][j] === cost[i - 1][j - 1] + 1 - sims[i - 1][j - 1]) {
      pairs.push([i - 1, j - 1, sims[i - 1][j - 1]]);
      i -= 1;
      j -= 1;
    } else if (cost[i][j] === cost[i - 1][j] + 1) {
      i -= 1;
    } else {
      j -= 1;
    }
  }
  return pairs.reverse();
}

/** The live tag each new word inherits (`null`: no partner at `minSim` or above).
    `live` = `[{word, lang}]`, `next` = normalised words. */
export function hintsFor(next, live, minSim) {
  const hints = next.map(() => null);
  for (const [i, j, sim] of alignWords(next, live.map((w) => w.word))) if (sim >= minSim) hints[i] = live[j].lang;
  return hints;
}

/** A `tagText(text, live?)` for the endpoint pass: with the utterance's live span-vote words
    (`[{word, lang}]`) and `inherit` (`asr-wordlevel.json → inherit`: `{min_sim, weight}`), the
    new words inherit the live tags through the word LID's hinted Viterbi; otherwise plain
    `lid.tagText`. */
export function createInheritingTagger(lid, inherit) {
  return (text, live = null) => {
    if (!inherit || !live?.length) return lid.tagText(text);
    const words = normalizeWords(text);
    const tags = lid.tagWithHints(words, hintsFor(words, live, inherit.min_sim), inherit.weight);
    return words.map((word, i) => ({ word, lang: tags[i] }));
  };
}
