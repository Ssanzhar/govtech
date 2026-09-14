/* The exported linear heads (`qorgan/classifier/web_bundle.py` layout), evaluated in
   double precision exactly like sklearn does on float32 inputs. */

const sigmoid = (x) => 1 / (1 + Math.exp(-x));

function dot(coef, row) {
  if (coef.length !== row.length) {
    throw new RangeError(`feature width ${row.length} != coefficient width ${coef.length}`);
  }
  let acc = 0;
  for (let i = 0; i < coef.length; i += 1) acc += coef[i] * row[i];
  return acc;
}

/** Calibrated scam probability: mean over CV members of sigmoid(-(a·d + b)), d = coef·x + b0. */
export function riskProba(row, riskHead) {
  if (riskHead.type !== "calibrated_lr_sigmoid_mean") {
    throw new Error(`unsupported risk head type: ${riskHead.type}`);
  }
  let total = 0;
  for (const m of riskHead.members) {
    const decision = dot(m.coef, row) + m.intercept;
    total += sigmoid(-(m.calib_a * decision + m.calib_b));
  }
  return total / riskHead.members.length;
}

/** Per-tactic probabilities in label-space order; tactics without a model score 0. */
export function tacticProba(embedding, tacticHead, labelSpace) {
  return labelSpace.map((tacticId) => {
    const model = tacticHead.models[tacticId];
    return model ? sigmoid(dot(model.coef, embedding) + model.intercept) : 0;
  });
}

/** `[embedding | cue block | reassurance]` as one plain array (the hybrid layout). */
export function hybridRow(embedding, cueBlock, reassurance) {
  return [...embedding, ...cueBlock, reassurance];
}

/** `(id, prob)` pairs at/above threshold, sorted by prob desc then id asc (labels.decode_tactics). */
export function decodeTactics(probs, labelSpace, threshold) {
  const selected = [];
  labelSpace.forEach((id, i) => {
    if (probs[i] >= threshold) selected.push([id, probs[i]]);
  });
  return selected.sort((a, b) => (b[1] - a[1]) || (a[0] < b[0] ? -1 : a[0] > b[0] ? 1 : 0));
}
