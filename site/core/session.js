/* Per-utterance live pipeline -- port of `qorgan/live/session.py`: utterance -> rolling
   window (head + tail) -> score -> meter -> evidence -> advice. Pure; `advance` returns
   `{state, update}`. The scorer is injected (see score.js). */

import { band, initialMeter, updateMeter } from "./meter.js";
import { emptyRecommendation, recommend } from "./recommend.js";

export function initialSession(locale, config) {
  if (!config.locales.includes(locale)) throw new RangeError(`Unsupported locale ${locale}`);
  return { locale, utterances: [], meter: initialMeter(), tags: [], seen_evidence: [] };
}

export function transcriptOf(state, config) {
  return state.utterances.join(config.window.join);
}

export function rollingWindow(utterances, w) {
  const full = utterances.join(w.join);
  if (full.length <= w.max_chars) return full;
  const head = utterances.slice(0, w.head_utterances);
  let budget = w.max_chars - head.join(w.join).length - w.join.length;
  const tail = [];
  const rest = utterances.slice(w.head_utterances);
  for (let i = rest.length - 1; i >= 0; i -= 1) {
    const cost = rest[i].length + w.join.length;
    if (budget - cost < 0 && tail.length) break;
    tail.unshift(rest[i]);
    budget -= cost;
  }
  return [...head, ...tail].join(w.join);
}

function accumulateTags(existing, fresh) {
  const weights = new Map(existing.map((t) => [t.id, t.weight]));
  for (const t of fresh) weights.set(t.id, Math.max(weights.get(t.id) ?? 0, t.weight));
  return [...weights].map(([id, weight]) => ({ id, weight }));
}

export async function advance(state, utterance, { score, config }) {
  const utterances = [...state.utterances, utterance.text];
  const windowText = rollingWindow(utterances, config.window);
  const result = await score(windowText);
  const hard = new Set(config.taxonomy.hard_signal_ids);
  const hardSignals = Object.fromEntries(result.tags.filter((t) => hard.has(t.id)).map((t) => [t.id, t.weight]));
  const meter = updateMeter(state.meter, { risk: result.risk, asrConfidence: utterance.confidence ?? 1, hardSignals }, config.meter);
  const currentBand = band(meter.score, config.meter);
  const tags = accumulateTags(state.tags, result.tags);
  const seen = new Set(state.seen_evidence);
  const newEvidence = result.attributions.filter((s) => !seen.has(s.text));
  const recommendation = currentBand !== "low" ? recommend(tags, state.locale, result.raw_confidence, config) : emptyRecommendation();
  return {
    state: { locale: state.locale, utterances, meter, tags, seen_evidence: [...state.seen_evidence, ...newEvidence.map((s) => s.text)] },
    update: { meter, band: currentBand, window_text: windowText, result, new_evidence: newEvidence, recommendation },
  };
}
