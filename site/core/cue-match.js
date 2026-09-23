/* Cue matching that survives the speech recogniser -- a 1:1 port of
   `qorgan/classifier/cue_match.py` (ADR D39). Change both files together and regenerate the
   golden parity fixtures; the Python module's docstring carries the reasoning.

   Verbatim first, then a bounded-edit search over the DE-SPACED normalised text, so a phrase
   the recogniser broke across a word boundary («на обороте» -> «наоборот де») is still found.
   An exact hit always wins, short cues get no edit budget at all, and the utterance boundary
   is a wall: de-spacing would otherwise let the tail of one turn and the head of the next
   concatenate into a cue neither contains.

   Offsets are this runtime's own (UTF-16 code units); Python's are code points. They coincide
   for every character this system handles -- `normalize` drops supplementary-plane ones. */

export const MATCHER_VERSION = 2;

// `config.window.join`: the fuzzy search never crosses it.
export const UTTERANCE_BOUNDARY = "\n";

// Letters and digits only -- spaces included in what is dropped, which is what makes a moved
// word boundary free. Kazakh letters are listed explicitly so no locale can drop them.
const KEEP = /[0-9a-zа-яёәғқңөұүһі]/;

const BUDGET_STEPS = [[12, 0], [21, 1], [31, 2]];
const MAX_BUDGET = 3;

/** Edit budget allowed for a cue of `length` normalised characters (monotonic). */
export function budgetFor(length) {
  for (const [limit, budget] of BUDGET_STEPS) if (length < limit) return budget;
  return MAX_BUDGET;
}

/** `[normalised de-spaced text, indexMap]`; `indexMap[i]` is the offset in the ORIGINAL
    string of normalised character `i`, so a match is always a verbatim span of what is shown. */
export function normalize(text) {
  const normalized = [];
  const indexMap = [];
  for (let index = 0; index < text.length; index += 1) {
    const folded = text[index].normalize("NFKC").toLowerCase().replace(/ё/g, "е");
    for (const piece of folded) {
      if (KEEP.test(piece)) {
        normalized.push(piece);
        indexMap.push(index);
      }
    }
  }
  return [normalized.join(""), indexMap];
}

/** `[start, end]` offsets of `cue` in `text`, or `null`. */
export function findCue(text, cue, { prefilter = true } = {}) {
  if (!text || !cue) return null;

  const loweredText = text.toLowerCase();
  const exact = loweredText.indexOf(cue.toLowerCase());
  if (exact !== -1) return [exact, exact + cue.length];

  const [normalizedCue] = normalize(cue);
  const budget = budgetFor(normalizedCue.length);
  if (!normalizedCue || budget === 0) return null;

  let offset = 0;
  for (const segment of text.split(UTTERANCE_BOUNDARY)) {
    const found = findInSegment(segment, normalizedCue, budget, prefilter);
    if (found) return [offset + found[0], offset + found[1]];
    offset += segment.length + UTTERANCE_BOUNDARY.length;
  }
  return null;
}

function findInSegment(segment, normalizedCue, budget, prefilter) {
  const [normalizedText, indexMap] = normalize(segment);
  if (!normalizedText) return null;
  if (prefilter && !mayMatch(normalizedCue, normalizedText, budget)) return null;
  const window = bestWindow(normalizedCue, normalizedText, budget);
  if (!window) return null;
  const [start, end] = window;
  return [indexMap[start], indexMap[end - 1] + 1];
}

/** Pigeonhole prefilter: at most `budget` of the cue's `budget + 1` disjoint blocks can be
    damaged, so at least one must survive verbatim. Exact -- it only rules matches out. */
function mayMatch(cue, text, budget) {
  const blocks = budget + 1;
  const size = Math.floor(cue.length / blocks);
  const remainder = cue.length % blocks;
  if (size === 0) return true;
  let start = 0;
  for (let index = 0; index < blocks; index += 1) {
    const end = start + size + (index < remainder ? 1 : 0);
    if (text.includes(cue.slice(start, end))) return true;
    start = end;
  }
  return false;
}

/** Levenshtein with a free start, carrying each cell's start offset. Ties resolve
    substitution -> deletion -> insertion, then earliest end, exactly as in Python. */
function bestWindow(cue, text, budget) {
  const cueLength = cue.length;
  const textLength = text.length;
  let costs = new Array(textLength + 1).fill(0);
  let starts = Array.from({ length: textLength + 1 }, (_, j) => j);

  for (let i = 1; i <= cueLength; i += 1) {
    const previousCosts = costs;
    const previousStarts = starts;
    costs = new Array(textLength + 1).fill(0);
    starts = new Array(textLength + 1).fill(0);
    costs[0] = i;
    for (let j = 1; j <= textLength; j += 1) {
      const substitute = previousCosts[j - 1] + (cue[i - 1] === text[j - 1] ? 0 : 1);
      const del = previousCosts[j] + 1;
      const ins = costs[j - 1] + 1;
      const best = Math.min(substitute, del, ins);
      costs[j] = best;
      if (best === substitute) starts[j] = previousStarts[j - 1];
      else if (best === del) starts[j] = previousStarts[j];
      else starts[j] = starts[j - 1];
    }
  }

  let bestCost = budget + 1;
  let bestEnd = -1;
  for (let j = 1; j <= textLength; j += 1) {
    if (costs[j] < bestCost) {
      bestCost = costs[j];
      bestEnd = j;
    }
  }
  if (bestEnd < 0 || bestCost > budget) return null;
  const start = starts[bestEnd];
  return bestEnd > start ? [start, bestEnd] : null;
}
