/* Live per-word kk/ru/en tags from the three Vosk recognisers: the calibrated span vote (ADR D63).

   A 1:1 port of `scripts/spikes/asr_wordlevel/spanvote/{common,vote,realtime}.py`, pinned by
   tests_js/fixtures/span_vote.json (`scripts/export_asr_wordlevel.py`):
   1. per recogniser, every word gets 13 cheap features (its confidence, duration, letters, the
      rivals' confidence over the same audio, the kk−ru char-trigram score...) and a logistic
      calibrator turns them into P(word is right AND in the right language);
   2. the timeline is cut at every word boundary; in each elementary span a language scores
      duration × logit(p) (a fixed `missing_logit` where it has no word); a 3-state Viterbi with a
      switch penalty picks one language per span; each language's words whose midpoint lies in its
      spans are emitted once, in time order;
   3. a span more than `freeze_s` behind the newest decoded word is frozen (its language locked
      for every later pass) so the display stops flickering.
   Rerun on every partial/result event. Measured (spike REPORT): word-language accuracy 0.873
   on the code-switch TEST set, but live English recall only 0.29 -- English words stay provisional
   until the endpoint pass replaces the utterance. Pure, no I/O. */

import { LANGS, normalizeWords } from "./word-lid.js";

const EPS = 1e-6;
const NEG = -1e9;
const KAZAKH_ONLY = "әғқңөұүһі";
const LATIN = /[a-z]/;

const mean = (xs) => (xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : 0);

/** logit with the confidence clipped to [clip, 1 - clip]. */
export function logit(p, clip = 1e-3) {
  const q = Math.min(1 - clip, Math.max(clip, p));
  return Math.log(q / (1 - q));
}

/** Vosk words (`{word, start, end, conf}` or `[word, start, end, conf]`) -> `[norm, start, end,
    conf]`; words that normalise to nothing are dropped, a multi-piece word keeps its first piece. */
export function cleanWords(words) {
  const out = [];
  for (const w of words || []) {
    const [word, start, end, conf] = Array.isArray(w) ? w : [w.word, w.start, w.end, typeof w.conf === "number" ? w.conf : 1];
    const pieces = normalizeWords(word);
    if (pieces.length) out.push([pieces[0], start, end, conf]);
  }
  return out;
}

function overlapStats(word, others, minDur) {
  const [, s, e] = word;
  const dur = Math.max(e - s, minDur);
  let best = 0;
  let count = 0;
  for (const hyp of others) {
    let num = 0;
    let den = 0;
    for (const w of hyp) {
      const ov = Math.min(e, w[2]) - Math.max(s, w[1]);
      if (ov > 0) { num += ov * w[3]; den += ov; count += 1; }
    }
    if (den > 0) best = Math.max(best, num / den);
  }
  return [best, count / dur];
}

function wordFeatures(word, hyp, othersMean, others, lid, clip, minDur) {
  const [w, s, e, c] = word;
  const dur = Math.max(e - s, minDur);
  const n = Math.max([...w].length, 1);
  const span = hyp.length ? Math.max(hyp[hyp.length - 1][2] - hyp[0][1], minDur) : minDur;
  const latin = LATIN.test(w);
  const feats = [
    logit(c, clip), c >= 0.999 ? 1 : 0, dur, Math.log(n / dur), Math.log(n),
    [...w].some((ch) => KAZAKH_ONLY.includes(ch)) ? 1 : 0, latin ? 1 : 0,
    hyp.length ? mean(hyp.map((x) => x[3])) : 0, othersMean, hyp.length / span,
    ...overlapStats(word, others, minDur),
  ];
  if (lid) feats.push(latin ? 0 : (lid.charLogprob(w, "kk") - lid.charLogprob(w, "ru")) / n);
  return feats;
}

/** Feature rows per language for cleaned hypotheses `{kk, ru, en}` (missing = []). */
export function hypFeatures(hyps, lid, { clip = 1e-3, minDur = 0.03 } = {}) {
  const means = Object.fromEntries(LANGS.map((l) => [l, hyps[l]?.length ? mean(hyps[l].map((x) => x[3])) : 0]));
  const out = {};
  for (const l of LANGS) {
    const h = hyps[l] || [];
    const rivals = LANGS.filter((o) => o !== l);
    const othersMean = Math.max(...rivals.map((o) => means[o]));
    out[l] = h.map((w) => wordFeatures(w, h, othersMean, rivals.map((o) => hyps[o] || []), lid, clip, minDur));
  }
  return out;
}

/** P(right word, right language) per word from the raw-space logistic calibrators. */
export function calibrate(features, calibrators, names) {
  const out = {};
  for (const l of LANGS) {
    const cal = calibrators[l];
    const coef = names.map((n) => cal.coef[n]);
    out[l] = (features[l] || []).map((row) => {
      let z = cal.intercept;
      for (let i = 0; i < coef.length; i++) z += coef[i] * row[i];
      return 1 / (1 + Math.exp(-z));
    });
  }
  return out;
}

function wordAt(words, x) {
  for (let i = 0; i < words.length; i++) if (words[i][1] - EPS <= x && x < words[i][2] + EPS) return i;
  return null;
}

/** The language of the labelled span containing `x`, or null. */
export function stateAt(labelled, x) {
  for (const [a, b, l] of labelled) if (a - EPS <= x && x < b + EPS) return l;
  return null;
}

function elementarySpans(hyps, cuts) {
  const points = new Set(cuts);
  for (const l of LANGS) for (const w of hyps[l] || []) { points.add(w[1]); points.add(w[2]); }
  const sorted = [...points].sort((a, b) => a - b);
  const spans = [];
  for (let i = 0; i + 1 < sorted.length; i++) if (sorted[i + 1] - sorted[i] > EPS) spans.push([sorted[i], sorted[i + 1]]);
  return spans;
}

function viterbi(em, gaps, params) {
  const k = LANGS.length;
  let score = em[0].slice();
  const back = [new Array(k).fill(0)];
  for (let t = 1; t < em.length; t++) {
    const free = params.pause !== null && params.pause !== undefined && gaps[t] >= params.pause;
    const pen = free ? 0 : params.switch;
    const next = new Array(k);
    const ptr = new Array(k);
    for (let j = 0; j < k; j++) {
      let best = -Infinity;
      let arg = 0;
      for (let i = 0; i < k; i++) {
        const v = score[i] - (i === j ? 0 : pen);
        if (v > best) { best = v; arg = i; }
      }
      ptr[j] = arg;
      next[j] = best + em[t][j];
    }
    back.push(ptr);
    score = next;
  }
  let last = 0;
  for (let i = 1; i < k; i++) if (score[i] > score[last]) last = i;
  const path = [last];
  for (let t = em.length - 1; t > 0; t--) path.push(back[t][path[path.length - 1]]);
  return path.reverse();
}

/** `{words: [[word, start, end, lang]], labelled: [[a, b, lang]]}` for cleaned `hyps` and their
    calibrated `probs`; `frozen` spans `[[a, b, lang]]` keep their language. */
export function spanVote(hyps, probs, params, frozen = []) {
  const bias = params.bias || {};
  const spans = elementarySpans(hyps, frozen.flatMap(([a, b]) => [a, b]));
  const rows = [];
  const gaps = [];
  const keep = [];
  let gap = Infinity;
  for (const [a, b] of spans) {
    const x = (a + b) / 2;
    const idx = LANGS.map((l) => wordAt(hyps[l] || [], x));
    if (idx.every((i) => i === null)) {
      gap = (gap !== Infinity ? gap : 0) + (b - a);
      continue;
    }
    const dur = b - a;
    let em = LANGS.map((l, li) => {
      const score = idx[li] !== null ? logit(probs[l][idx[li]]) : params.missing;
      return dur * (score + (bias[l] ?? 0));
    });
    const lock = frozen.find(([fa, fb]) => fa - EPS <= x && x < fb + EPS);
    if (lock) em = em.map((e, li) => (LANGS[li] === lock[2] ? e : NEG));
    rows.push(em);
    gaps.push(gap);
    keep.push([a, b]);
    gap = 0;
  }
  if (!rows.length) return { words: [], labelled: [] };
  const states = viterbi(rows, gaps, params);
  const labelled = keep.map(([a, b], i) => [a, b, LANGS[states[i]]]);
  const words = [];
  for (const l of LANGS) {
    for (const w of hyps[l] || []) if (stateAt(labelled, (w[1] + w[2]) / 2) === l) words.push([w[0], w[1], w[2], l]);
  }
  words.sort((p, q) => p[1] - q[1] || p[2] - q[2]);
  return { words, labelled };
}

/** Freeze state after a vote: spans more than `freezeS` behind the newest word end are locked. */
export function advanceFreeze(freeze, labelled, hyps, freezeS) {
  let newest = 0;
  for (const l of LANGS) for (const w of hyps[l] || []) newest = Math.max(newest, w[2]);
  const until = Math.max(freeze.until, newest - freezeS);
  const added = [];
  for (const [a, b, l] of labelled) {
    const lo = Math.max(a, freeze.until);
    const hi = Math.min(b, until);
    if (hi > lo) added.push([lo, hi, l]);
  }
  return { spans: [...freeze.spans, ...added], until };
}

export const initialFreeze = () => ({ spans: [], until: 0 });

/** Compile `asr-wordlevel.json → span_vote` against a word-LID instance into one `label` call:
    raw per-language Vosk words -> `{words, labelled, freeze}` (inputs untouched). */
export function createSpanVoter(config, lid) {
  const params = { switch: config.switch_penalty, missing: config.missing_logit, pause: config.pause_reset_s, bias: config.bias_logit };
  const opts = { clip: config.conf_clip, minDur: config.min_duration_s };
  return {
    freezeS: config.freeze_s,
    label(rawHyps, freeze = initialFreeze()) {
      const hyps = Object.fromEntries(LANGS.map((l) => [l, cleanWords(rawHyps[l] || [])]));
      const probs = calibrate(hypFeatures(hyps, lid, opts), config.calibrators, config.features);
      const { words, labelled } = spanVote(hyps, probs, params, freeze.spans);
      return { words, labelled, probs, freeze: advanceFreeze(freeze, labelled, hyps, config.freeze_s) };
    },
  };
}
