/* Hard-signal cue lexicon + reassurance matcher -- a 1:1 port of
   `qorgan/classifier/features.py` (match_cues, hard_signal_features) and
   `qorgan/classifier/reassurance.py`. Pure functions over the lexicon block that ships
   inside weights.json, so the client can never drift from the weights it was trained with. */

import { findCue } from "./cue-match.js";

/** Cue presence per hard-signal tactic, in the fixed column order. Matching is `findCue`
    (verbatim, then bounded-edit over the de-spaced text), so the feature and the highlight
    can never disagree. */
export function hardSignalFeatures(text, cues, hardSignalIds) {
  return hardSignalIds.map((tacticId) =>
    (cues[tacticId] || []).some((cue) => findCue(text, cue) !== null) ? 1 : 0,
  );
}

/** One grounded match per tactic that fired: `{tacticId, span:{text,start,end}}`. The span is
    sliced from the ORIGINAL text, so a fuzzy hit highlights what was actually said. */
export function matchCues(text, cues, hardSignalIds) {
  const matches = [];
  for (const tacticId of hardSignalIds) {
    const span = firstGroundedSpan(text, cues[tacticId] || []);
    if (span) matches.push({ tacticId, span });
  }
  return matches;
}

function firstGroundedSpan(text, cueList) {
  for (const cue of cueList) {
    const found = findCue(text, cue);
    if (!found) continue;
    const [start, end] = found;
    const candidate = text.slice(start, end);
    // An exact hit must slice back to the cue; a fuzzy hit is the recogniser's own wording.
    if (candidate && (candidate.length !== cue.length || candidate.toLowerCase() === cue.toLowerCase())) {
      return { text: candidate, start, end };
    }
  }
  return null;
}

const SENTENCE_STOP = "[^.!?\\n]";

function escapeRegex(term) {
  return term.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

/** A term as a regex: tokens joined by optional whitespace, so ASR output that glues a
    negation to the next word ("ненужно") still matches -- 1:1 with `reassurance._term`. */
function termPattern(term) {
  return term.split(/\s+/).filter(Boolean).map(escapeRegex).join("\\s*");
}

/** Compile the reassurance matcher: a sensitive term within `window_chars` of a
    reassurance term, either order, never crossing a sentence stop. */
export function compileReassurance(patterns) {
  const sensitive = "(?:" + patterns.sensitive_terms.map(termPattern).join("|") + ")";
  const reassure = "(?:" + patterns.reassurance_terms.map(termPattern).join("|") + ")";
  const gap = `${SENTENCE_STOP}{0,${patterns.window_chars}}?`;
  return new RegExp(`${sensitive}${gap}${reassure}|${reassure}${gap}${sensitive}`, "iu");
}

/** 1 if `text` contains a reassurance, else 0. `matcher` from `compileReassurance`. */
export function reassuranceFeature(text, matcher) {
  return matcher.test(text) ? 1 : 0;
}
