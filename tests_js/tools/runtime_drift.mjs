/* Cross-runtime drift probe (PLAN B8): embed the same texts with transformers.js in Node
   (onnxruntime-node) and compare with the Python-ORT vectors dumped by
   scripts/embed_probe.py. Usage: node tests_js/tools/runtime_drift.mjs <probe.json> <model_id> */
import { readFileSync } from "node:fs";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";

const [probePath, modelId] = process.argv.slice(2);
if (!probePath || !modelId) { console.error("usage: runtime_drift.mjs <probe.json> <model_id>"); process.exit(2); }
const REPO = join(dirname(fileURLToPath(import.meta.url)), "..", "..");
const probe = JSON.parse(readFileSync(probePath, "utf8"));
const decode = (b64) => { const b = Buffer.from(b64, "base64"); return Array.from(new Float32Array(b.buffer, b.byteOffset, b.byteLength / 4)); };

const { env, pipeline } = await import("@huggingface/transformers");
env.allowRemoteModels = false; env.localModelPath = join(REPO, "site", "models");
const extractor = await pipeline("feature-extraction", modelId, { dtype: "q8" });
const cos = [];
let ms = 0;
for (let i = 0; i < probe.texts.length; i += 1) {
  const t0 = performance.now();
  const out = await extractor(["query: " + probe.texts[i]], { pooling: "mean", normalize: true });
  ms += performance.now() - t0;
  const a = Array.from(out.data), b = decode(probe.embeddings[i]);
  cos.push(a.reduce((s, x, k) => s + x * b[k], 0));
}
const mean = cos.reduce((a, b) => a + b, 0) / cos.length;
console.log(`${modelId}: Node-ORT vs Python-ORT cosine mean=${mean.toFixed(5)} min=${Math.min(...cos).toFixed(5)} (${(ms / cos.length).toFixed(0)} ms/text)`);
