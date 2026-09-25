/* Node reproduces the server (ADR D32). The server pins `onnxruntime` to the version that
   transformers.js bundles as `onnxruntime-node` (1.21 today; tests/test_runtime_pin.py keeps
   them together), and with the same version the int8 graph is bit-identical on both sides:
   cosine 1.00000, |Δembedding| 0 on the 200-case gate (2026-09-20). Before the pin (Python
   1.27 vs Node 1.21) it was cosine 0.980 / min 0.958 and 1-1.5 % decision flips -- which D28
   had taken for an inherent runtime residual. This test asserts the reproduction on the 28
   golden transcripts; the DEVICE gate (the browser's WASM runtime, which is a third build and
   does drift) is tests_js/integration/browser_gate.test.mjs. Skipped when the self-hosted
   model files are absent (run scripts/deploy_bootstrap.py first). */
import test from "node:test";
import assert from "node:assert/strict";
import { existsSync } from "node:fs";
import { join } from "node:path";
import { createScorer } from "../../site/core/score.js";
import { REPO, config, fixtureEmbed, parity, weights } from "../helpers.mjs";

const MODEL_DIR = join(REPO, "site", "models");
const MODEL_ID = config.embedder.model_id;
const available = existsSync(join(MODEL_DIR, MODEL_ID, "onnx", "model_quantized.onnx"));
// Fixtures exported with QORGAN_EMBED_BACKEND=onnx come from the server's native ORT: Node
// must reproduce them bit-for-bit. Fixtures exported with the `device` backend (ADR D33) come
// from the browser's WASM build, of which native ORT is only a cosine-0.98 proxy: then this
// test documents the proxy level instead (the device gate is browser_gate.test.mjs).
const EXPECT = parity.embed_backend === "device"
  ? { minCosine: 0.94, maxRiskDelta: 0.35, label: "proxy of the device (browser WASM) embeddings" }
  : { minCosine: 0.9999, maxRiskDelta: 1e-3, label: "bit-for-bit (same graph, same ORT version)" };

const cosine = (a, b) => a.reduce((s, x, i) => s + x * b[i], 0);

test(`int8 ONNX in Node vs the Python fixtures: ${EXPECT.label}`, { skip: !available && "self-hosted model not downloaded" }, async () => {
  const { env, pipeline } = await import("@huggingface/transformers");
  env.allowRemoteModels = false;
  env.localModelPath = MODEL_DIR;
  const extractor = await pipeline("feature-extraction", MODEL_ID, { dtype: config.embedder.dtype });
  // One text per run -- dynamic int8 quantisation is batch-sensitive (see embed-worker.js).
  const onnxEmbed = async (texts) => {
    const rows = [];
    for (const t of texts) rows.push(Array.from((await extractor([config.embedder.prefix + t], { pooling: "mean", normalize: true })).data));
    return rows;
  };

  const texts = parity.cases.map((c) => c.transcript);
  const pyVectors = await fixtureEmbed(texts);
  const t0 = performance.now();
  const jsVectors = await onnxEmbed(texts);
  const ms = (performance.now() - t0) / texts.length;
  const cosines = jsVectors.map((v, i) => cosine(v, pyVectors[i]));
  const min = Math.min(...cosines);
  console.log(`${MODEL_ID} (Node/onnxruntime-node) vs Python fixtures (${parity.embed_backend}): cosine min=${min.toFixed(6)}; ${ms.toFixed(0)} ms/transcript`);
  assert.ok(min >= EXPECT.minCosine, `min cosine ${min}: ${parity.embed_backend === "device" ? "the native proxy drifted further from the device than measured" : "Node and Python are not running the same ONNX Runtime version (see tests/test_runtime_pin.py)"}`);

  const score = createScorer({ weights, config, embed: onnxEmbed });
  let maxDelta = 0;
  for (const c of parity.cases) {
    const r = await score(c.transcript);
    maxDelta = Math.max(maxDelta, Math.abs(r.risk - c.score.risk));
  }
  console.log(`risk |delta| max=${maxDelta.toExponential(2)} over ${parity.cases.length} golden cases`);
  assert.ok(maxDelta < EXPECT.maxRiskDelta, `max risk delta ${maxDelta}`);
});
