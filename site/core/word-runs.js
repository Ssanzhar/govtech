/* Display helpers for per-word language tags (ADR D64). Pure; the page renders the result.

   `alignTokens(text, words)` pairs each whitespace token of the transcript with its tagged
   words (`{word, lang}` from word-lid.js; a hyphenated token takes several, a punctuation-only
   token none), keeping the token's character offsets so evidence highlighting still works on the
   verbatim text; null when the two do not line up (then the page shows the line without word
   tags rather than guessing).
   `runsOf(items)` groups consecutive items of one language: a language chip is drawn per run,
   not per word. */

import { normalizeWords } from "./word-lid.js";

export function alignTokens(text, words) {
  const tokens = [];
  const re = /\S+/g;
  let m;
  while ((m = re.exec(text)) !== null) tokens.push({ text: m[0], start: m.index, end: m.index + m[0].length });
  let next = 0;
  const out = tokens.map((token) => {
    const pieces = normalizeWords(token.text);
    const taken = words.slice(next, next + pieces.length);
    if (taken.length !== pieces.length || pieces.some((p, i) => p !== taken[i].word)) return null;
    next += pieces.length;
    return { ...token, lang: pieceLanguage(taken) };
  });
  if (!words.length || next !== words.length || out.some((t) => t === null)) return null;
  // a punctuation-only token ("—") has no word: it takes the previous token's language, or the next one's
  const neighbour = (i) => out.slice(0, i).reverse().find((t) => t.lang)?.lang ?? out.slice(i + 1).find((t) => t.lang).lang;
  return out.map((t, i) => (t.lang ? t : { ...t, lang: neighbour(i) }));
}

/** The language of a token made of several words ("из-за", "Kaspi-ге"): the majority, ties to the
    last piece (a Kazakh case ending decides a loanword); null for a token with no word. */
function pieceLanguage(pieces) {
  if (!pieces.length) return null;
  const counts = {};
  for (const p of pieces) counts[p.lang] = (counts[p.lang] ?? 0) + 1;
  const best = Math.max(...Object.values(counts));
  for (let i = pieces.length - 1; i >= 0; i--) if (counts[pieces[i].lang] === best) return pieces[i].lang;
  return null;
}

export function runsOf(items) {
  const runs = [];
  for (const item of items) {
    const last = runs[runs.length - 1];
    if (last && last.lang === item.lang) last.items.push(item);
    else runs.push({ lang: item.lang, items: [item] });
  }
  return runs;
}

/** Character ranges `[start, end)` of the first occurrence of each evidence phrase in `text`. */
export function evidenceRanges(text, evidence) {
  const ranges = [];
  for (const phrase of evidence || []) {
    if (!phrase) continue;
    const at = text.indexOf(phrase);
    if (at >= 0) ranges.push([at, at + phrase.length]);
  }
  return ranges;
}

export const overlaps = (token, ranges) => ranges.some(([a, b]) => token.start < b && a < token.end);
