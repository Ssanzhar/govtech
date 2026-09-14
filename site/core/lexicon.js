/* Hard-signal cue lexicon + reassurance matcher -- a 1:1 port of
   `qorgan/classifier/features.py` (match_cues, hard_signal_features) and
   `qorgan/classifier/reassurance.py`. Pure functions over the lexicon block that ships
   inside weights.json, so the client can never drift from the weights it was trained with. */

/** Case-insensitive substring presence per hard-signal tactic, in the fixed column order. */
export function hardSignalFeatures(text, cues, hardSignalIds) {
  const lowered = text.toLowerCase();
  return hardSignalIds.map((tacticId) =>
    (cues[tacticId] || []).some((cue) => lowered.includes(cue.toLowerCase())) ? 1 : 0,
  );
}

/** One grounded (verbatim) match per tactic that fired: `{tacticId, span:{text,start,end}}`. */
export function matchCues(text, cues, hardSignalIds) {
  const lowered = text.toLowerCase();
  const matches = [];
  for (const tacticId of hardSignalIds) {
    const span = firstGroundedSpan(text, lowered, cues[tacticId] || []);
    if (span) matches.push({ tacticId, span });
  }
  return matches;
}

function firstGroundedSpan(text, lowered, cueList) {
  for (const cue of cueList) {
    const index = lowered.indexOf(cue.toLowerCase());
    if (index === -1) continue;
    const candidate = text.slice(index, index + cue.length);
    if (candidate.toLowerCase() === cue.toLowerCase()) {
      return { text: candidate, start: index, end: index + candidate.length };
    }
  }
  return null;
}

const SENTENCE_STOP = "[^.!?\\n]";

function escapeRegex(term) {
  return term.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

/** Compile the reassurance matcher: a sensitive term within `window_chars` of a
    reassurance term, either order, never crossing a sentence stop. */
export function compileReassurance(patterns) {
  const sensitive = "(?:" + patterns.sensitive_terms.map(escapeRegex).join("|") + ")";
  const reassure = "(?:" + patterns.reassurance_terms.map(escapeRegex).join("|") + ")";
  const gap = `${SENTENCE_STOP}{0,${patterns.window_chars}}?`;
  return new RegExp(`${sensitive}${gap}${reassure}|${reassure}${gap}${sensitive}`, "iu");
}

/** 1 if `text` contains a reassurance, else 0. `matcher` from `compileReassurance`. */
export function reassuranceFeature(text, matcher) {
  return matcher.test(text) ? 1 : 0;
}
