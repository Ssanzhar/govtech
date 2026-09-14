/* Grounded top-utterance attribution -- port of
   `qorgan/classifier/attribution.py::select_top_utterance_spans`. Offsets are computed
   against `join.join(utterances)`, so duplicate utterances resolve to distinct positions. */

export function selectTopUtteranceSpans(utterances, scores, { topK, minScore = 0, join = "\n" }) {
  if (utterances.length !== scores.length) {
    throw new RangeError(`utterances (${utterances.length}) and scores (${scores.length}) differ in length`);
  }
  if (topK <= 0) throw new RangeError(`topK must be > 0, got ${topK}`);
  const candidates = [];
  let cursor = 0;
  utterances.forEach((utterance, index) => {
    const start = cursor;
    const end = cursor + utterance.length;
    cursor = end + join.length;
    if (scores[index] >= minScore && utterance.trim()) candidates.push({ score: scores[index], index, start, end });
  });
  const top = candidates.sort((a, b) => (b.score - a.score) || (a.index - b.index)).slice(0, topK);
  return top
    .map(({ index, start, end }) => ({ text: utterances[index], start, end }))
    .sort((a, b) => a.start - b.start);
}
