/* Per-word language tags (kk / ru / en) for a transcript from the joint recogniser (ADR D63).

   A 1:1 port of `scripts/spikes/asr_wordlevel/word_lid.py`, pinned by
   tests_js/fixtures/word_lid.json (`scripts/export_asr_wordlevel.py`):
   - the letters decide when they can: a Latin letter -> en, a Kazakh-only letter -> kk;
   - plain-Cyrillic words are scored by a char-trigram LM per language (kk, ru) plus a
     word-frequency lexicon, and the utterance is decoded with Viterbi under a language-switch
     penalty, so a lone ambiguous word ("мен", "код") follows its neighbours.
   The model (`asr-wordlevel.json → lid`) was fitted on the `train` split only. Pure, no I/O. */

export const LANGS = Object.freeze(["kk", "ru", "en"]);
const PAD_START = "^";
const PAD_END = "$";
const ADD_K = 0.1; // CharLM add-k smoothing
const LEX_ADD = 0.5; // lexicon add-k smoothing

/** NFC, lower case, ё→е, punctuation to spaces, `_` to space, collapsed whitespace
    (Python `normalize`: `[^\w\s']` with Unicode `\w`). */
export function normalizeText(text) {
  return String(text ?? "")
    .normalize("NFC")
    .toLowerCase()
    .replace(/ё/g, "е")
    .replace(/[^\p{L}\p{N}_\s']/gu, " ")
    .replace(/_/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

/** Words of a transcript after normalisation. */
export const normalizeWords = (text) => {
  const norm = normalizeText(text);
  return norm ? norm.split(" ") : [];
};

const LATIN = /[a-z]/;

/** "en" for a word with a Latin letter, "kk" for one with a Kazakh-only letter, else null. */
export function letterLanguage(word, kazakhOnly = "әғқңөұүһі") {
  if (LATIN.test(word)) return "en";
  for (const ch of word) if (kazakhOnly.includes(ch)) return "kk";
  return null;
}

/** Compile `asr-wordlevel.json → lid` into lookup tables (context totals precomputed). */
export function createWordLid(lid) {
  const lms = {};
  for (const [lang, lm] of Object.entries(lid.char_lms)) {
    const totals = {};
    for (const [ctx, counts] of Object.entries(lm.counts)) totals[ctx] = Object.values(counts).reduce((a, b) => a + b, 0);
    lms[lang] = { counts: lm.counts, totals, alphabetSize: lm.alphabet_size };
  }
  const lexicons = {};
  for (const [lang, lex] of Object.entries(lid.lexicons)) {
    lexicons[lang] = { counts: lex, total: Object.values(lex).reduce((a, b) => a + b, 0) };
  }
  const model = {
    order: lid.order,
    switchPenalty: lid.switch_penalty,
    lexiconWeight: lid.lexicon_weight,
    lexiconVocab: lid.lexicon_vocab,
    hard: lid.hard,
    kazakhOnly: lid.kazakh_only,
    lms,
    lexicons,
  };
  return {
    charLogprob: (word, lang) => charLogprob(model, word, lang),
    emission: (word, lang) => emission(model, word, lang),
    tag: (words) => tagWords(model, words),
    tagText: (text) => {
      const words = normalizeWords(text);
      const tags = tagWords(model, words);
      return words.map((word, i) => ({ word, lang: tags[i] }));
    },
  };
}

function charLogprob(model, word, lang) {
  const lm = model.lms[lang];
  const padded = [...PAD_START.repeat(model.order - 1), ...word, PAD_END];
  let total = 0;
  for (let i = model.order - 1; i < padded.length; i++) {
    const ctx = padded.slice(i - model.order + 1, i).join("");
    const counts = lm.counts[ctx];
    const n = counts?.[padded[i]] ?? 0;
    const sum = lm.totals[ctx] ?? 0;
    total += Math.log((n + ADD_K) / (sum + ADD_K * lm.alphabetSize));
  }
  return total;
}

function emission(model, word, lang) {
  const decided = letterLanguage(word, model.kazakhOnly);
  if (decided) return decided === lang ? 0 : -model.hard;
  if (lang === "en") return -model.hard;
  const lex = model.lexicons[lang];
  const lexLp = Math.log(((lex.counts[word] ?? 0) + LEX_ADD) / (lex.total + LEX_ADD * model.lexiconVocab));
  return charLogprob(model, word, lang) + model.lexiconWeight * lexLp;
}

/** Viterbi over LANGS; ties go to the earlier language in LANGS (Python `max` semantics). */
function tagWords(model, words) {
  if (!words.length) return [];
  let scores = Object.fromEntries(LANGS.map((l) => [l, emission(model, words[0], l)]));
  const back = [];
  for (const word of words.slice(1)) {
    const step = {};
    const ptr = {};
    for (const l of LANGS) {
      let prev = null;
      let best = -Infinity;
      for (const p of LANGS) {
        const s = scores[p] - (p === l ? 0 : model.switchPenalty);
        if (s > best) { best = s; prev = p; }
      }
      step[l] = best + emission(model, word, l);
      ptr[l] = prev;
    }
    scores = step;
    back.push(ptr);
  }
  let last = LANGS[0];
  for (const l of LANGS) if (scores[l] > scores[last]) last = l;
  const path = [last];
  for (let i = back.length - 1; i >= 0; i--) path.push(back[i][path[path.length - 1]]);
  return path.reverse();
}
