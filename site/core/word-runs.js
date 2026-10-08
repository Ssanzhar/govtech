/* Display helpers for per-word language tags (ADR D63). Pure; the page renders the result.

   `alignTokens(text, words)` pairs each whitespace token of the transcript with its tagged
   word (`{word, lang}` from word-lid.js), keeping the token's character offsets so evidence
   highlighting still works on the verbatim text; null when the two do not line up (then the
   page shows the line without word tags rather than guessing).
   `runsOf(items)` groups consecutive items of one language: a language chip is drawn per run,
   not per word. */

import { normalizeWords } from "./word-lid.js";

export function alignTokens(text, words) {
  const tokens = [];
  const re = /\S+/g;
  let m;
  while ((m = re.exec(text)) !== null) tokens.push({ text: m[0], start: m.index, end: m.index + m[0].length });
  if (tokens.length !== words.length) return null;
  for (let i = 0; i < tokens.length; i++) {
    const pieces = normalizeWords(tokens[i].text);
    if (pieces.length !== 1 || pieces[0] !== words[i].word) return null;
    tokens[i].lang = words[i].lang;
  }
  return tokens;
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
