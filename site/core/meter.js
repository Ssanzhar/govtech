/* The 0-100 suspicion meter -- port of `qorgan/live/meter.py`. Pure: `update` returns a
   new state. Arithmetic order matches Python so trajectories are bit-identical. */

export function initialMeter() {
  return { score: 0, latched: false, turn_index: 0, hard_signal_ids: [], ledger: [] };
}

export function band(score, m) {
  if (!(score >= 0 && score <= m.bands.score_max)) throw new RangeError(`score out of range: ${score}`);
  if (score <= m.bands.low_max) return "low";
  if (score <= m.bands.medium_max) return "medium";
  if (score <= m.bands.high_max) return "high";
  return "critical";
}

export function updateMeter(state, { risk, asrConfidence = 1, hardSignals = {} }, m) {
  if (!(risk >= 0 && risk <= 1)) throw new RangeError(`risk out of range: ${risk}`);
  if (!(asrConfidence >= 0 && asrConfidence <= 1)) throw new RangeError(`asrConfidence out of range: ${asrConfidence}`);
  const target = m.bands.score_max * risk;
  const alpha = target > state.score ? m.alpha_up : m.alpha_down;
  let score = state.score + alpha * asrConfidence * (target - state.score);

  const confident = Object.entries(hardSignals)
    .filter(([, c]) => c >= m.hard_signal_confidence_floor)
    .map(([id]) => id);
  const accumulated = [...new Set([...state.hard_signal_ids, ...confident])].sort();
  if (accumulated.length >= 2) score = Math.max(score, m.double_hard_signal_floor);
  else if (accumulated.length === 1) score = Math.max(score, m.single_hard_signal_floor);
  score = Math.min(Math.max(score, 0), m.bands.score_max);

  const enter = m.enter * m.bands.score_max;
  const exit = m.exit * m.bands.score_max;
  const latched = state.latched ? score > exit : score >= enter;
  const before = new Set(state.hard_signal_ids);
  const event = {
    turn_index: state.turn_index + 1,
    risk,
    asr_confidence: asrConfidence,
    score_before: state.score,
    score_after: score,
    new_hard_signal_ids: confident.filter((id) => !before.has(id)).sort(),
  };
  return { score, latched, turn_index: state.turn_index + 1, hard_signal_ids: accumulated, ledger: [...state.ledger, event] };
}
