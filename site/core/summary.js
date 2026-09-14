/* Post-call summary + report draft -- port of `qorgan/live/summary.py` (summarize,
   build_report). The draft is what the citizen reviews before an explicit submit. */

import { displayName } from "./explain.js";
import { band } from "./meter.js";
import { recommend } from "./recommend.js";

const MAX_RECOMMENDED_ACTIONS = 3;

export function summarize(state, config) {
  const names = state.tags.map((t) => displayName(t.id, state.locale, config)).filter((n) => n);
  const recommendation = recommend(state.tags, state.locale, null, config);
  return {
    final_score: state.meter.score,
    band: band(state.meter.score, config.meter),
    tactic_names: names,
    recommended_actions: recommendation.advices.slice(0, MAX_RECOMMENDED_ACTIONS),
    human_note: config.templates[state.locale].human_note,
  };
}

export function buildReport(state, config, { phoneNumber = null, timestamp = new Date() } = {}) {
  return {
    phone_number: phoneNumber,
    transcript: state.utterances.join(config.window.join),
    flagged_phrases: [...state.seen_evidence],
    tactic_ids: state.tags.map((t) => t.id),
    timestamp: timestamp.toISOString(),
    risk_score: state.meter.score,
  };
}
