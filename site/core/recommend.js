/* Tactic -> advice -- port of `qorgan/explain/recommend.py`. Hard signals first, then by
   weight; below the confidence floor the engine softens to verification questions. */

export class AdviceError extends Error {}

export const emptyRecommendation = () => ({ advices: [], verification_questions: [], softened: false, note: null });

export function recommend(tags, locale, confidence, config) {
  if (!config.locales.includes(locale)) throw new AdviceError(`Unsupported locale ${locale}`);
  if (!tags.length) return emptyRecommendation();
  const advice = config.advice[locale];
  const known = new Set(config.taxonomy.tactics.map((t) => t.id));
  const hard = new Set(config.taxonomy.hard_signal_ids);
  const ranked = tags
    .filter((tag) => known.has(tag.id))
    .map((tag, index) => ({ tag, index }))
    .sort((a, b) => (Number(!hard.has(a.tag.id)) - Number(!hard.has(b.tag.id))) || (b.tag.weight - a.tag.weight) || (a.index - b.index))
    .map(({ tag }) => tag);
  const advices = [];
  for (const tag of ranked) {
    const text = advice.tactic_advice[tag.id];
    if (!advices.includes(text)) advices.push(text);
  }
  if (confidence !== null && confidence !== undefined && confidence < config.scoring.low_confidence_floor) {
    return { advices: [], verification_questions: [...advice.verification_questions], softened: true, note: advice.low_confidence_note };
  }
  return { advices, verification_questions: [...advice.verification_questions], softened: false, note: null };
}
