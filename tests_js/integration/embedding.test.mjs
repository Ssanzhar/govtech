/* B3 gate (decision-level). The device and the server run the SAME int8 graph, but a
   dynamically-quantised graph is not bit-stable across ONNX Runtime versions (Python ORT
   1.27 vs onnxruntime-node 1.21/1.24 vs onnxruntime-web in the browser): measured cosine
   0.985-0.994 mean / 0.97-0.98 min between runtimes, tokenisation identical. With heads
   trained on int8 embeddings (A4) the residual is DECISION-safe (0 flips / 28 across every
   runtime tried) but not tag-stable (|Δrisk| up to 0.11, tag-set Jaccard 0.94-0.98). The
   gate below is what is guaranteed today; bit-level parity needs static (calibrated)
   quantisation -- PLAN B8, promoted to the next item. Skipped when the self-hosted model files
   are absent (run scripts/deploy_bootstrap.py first). */
import test from "node:test";
import assert from "node:assert/strict";
import { existsSync } from "node:fs";
import { join } from "node:path";
import { createScorer } from "../../site/core/score.js";
import { REPO, config, fixtureEmbed, parity, weights } from "../helpers.mjs";

const MODEL_DIR = join(REPO, "site", "models");
const MODEL_ID = "Xenova/multilingual-e5-base";
const available = existsSync(join(MODEL_DIR, MODEL_ID, "onnx", "model_quantized.onnx"));

const cosine = (a, b) => a.reduce((s, x, i) => s + x * b[i], 0);

test("int8 ONNX in Node agrees with Python-ONNX at the decision level (same graph, different ORT)", { skip: !available && "self-hosted model not downloaded" }, async () => {
  const { env, pipeline } = await import("@huggingface/transformers");
  env.allowRemoteModels = false;
  env.localModelPath = MODEL_DIR;
  const extractor = await pipeline("feature-extraction", MODEL_ID, { dtype: "q8" });
  // One text per run -- dynamic int8 quantisation is batch-sensitive (see embed-worker.js).
  const onnxEmbed = async (texts) => {
    const rows = [];
    for (const t of texts) rows.push(Array.from((await extractor(["query: " + t], { pooling: "mean", normalize: true })).data));
    return rows;
  };

  const texts = parity.cases.map((c) => c.transcript);
  const pyVectors = await fixtureEmbed(texts);
  const t0 = performance.now();
  const jsVectors = await onnxEmbed(texts);
  const ms = (performance.now() - t0) / texts.length;
  const cosines = jsVectors.map((v, i) => cosine(v, pyVectors[i]));
  const mean = cosines.reduce((a, b) => a + b, 0) / cosines.length;
  const min = Math.min(...cosines);
  console.log(`e5-base int8 (Node/onnxruntime-node) vs Python (onnxruntime): cosine mean=${mean.toFixed(4)} min=${min.toFixed(4)}; ${ms.toFixed(0)} ms/transcript`);
  assert.ok(mean >= 0.98, `mean cosine ${mean}`);
  assert.ok(min >= 0.96, `min cosine ${min}`);

  const score = createScorer({ weights, config, embed: onnxEmbed });
  let maxDelta = 0;
  let flips = 0;
  let jaccard = 0;
  for (const c of parity.cases) {
    const r = await score(c.transcript);
    maxDelta = Math.max(maxDelta, Math.abs(r.risk - c.score.risk));
    if ((r.risk >= config.scoring.risk_threshold) !== (c.score.risk >= config.scoring.risk_threshold)) flips += 1;
    const a = new Set(r.tags.map((t) => t.id));
    const b = new Set(c.score.tags.map((t) => t.id));
    const union = new Set([...a, ...b]).size;
    jaccard += union ? [...a].filter((x) => b.has(x)).length / union : 1;
  }
  jaccard /= parity.cases.length;
  console.log(`risk |delta| max=${maxDelta.toFixed(4)}, decision flips=${flips}/${parity.cases.length}, tag-set Jaccard=${jaccard.toFixed(3)}`);
  assert.ok(maxDelta < 0.15, `max risk delta ${maxDelta}`);
  assert.equal(flips, 0);
  assert.ok(jaccard >= 0.9, `tag-set Jaccard ${jaccard}`);
});
