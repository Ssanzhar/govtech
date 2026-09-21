/* Cross-runtime decision probe for ANY heads + gate set (PLAN B3 / D17 residual):
   node tests_js/tools/runtime_gate_probe.mjs <weights.json> <runtime_gate.json>
   Embeds every transcript with onnxruntime-node and reports flips, |Δrisk| and tag Jaccard
   against the Python verdicts recorded in the gate file. */
import { readFileSync } from "node:fs";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import { createScorer } from "../../site/core/score.js";

const [weightsPath, gatePath] = process.argv.slice(2);
if (!weightsPath || !gatePath) { console.error("usage: runtime_gate_probe.mjs <weights.json> <runtime_gate.json>"); process.exit(2); }
const REPO = join(dirname(fileURLToPath(import.meta.url)), "..", "..");
const config = JSON.parse(readFileSync(join(REPO, "site", "core", "qorgan-config.json"), "utf8"));
const weights = JSON.parse(readFileSync(weightsPath, "utf8"));
const gate = JSON.parse(readFileSync(gatePath, "utf8"));
const { env, pipeline } = await import("@huggingface/transformers");
env.allowRemoteModels = false; env.localModelPath = join(REPO, "site", "models");
const extractor = await pipeline("feature-extraction", config.embedder.model_id, { dtype: config.embedder.dtype });
const embed = async (texts) => { const rows = []; for (const t of texts) rows.push(Array.from((await extractor([config.embedder.prefix + t], { pooling: "mean", normalize: true })).data)); return rows; };
const score = createScorer({ weights, config, embed });
const thr = gate.threshold ?? config.scoring.risk_threshold;
let maxDelta = 0, jaccard = 0; const flips = [], deltas = [];
for (const c of gate.cases) {
  const r = await score(c.transcript);
  const d = Math.abs(r.risk - c.risk); deltas.push(d); maxDelta = Math.max(maxDelta, d);
  if ((r.risk >= thr) !== (c.risk >= thr)) flips.push(`${c.id} py=${c.risk.toFixed(3)} node=${r.risk.toFixed(3)}`);
  const a = new Set(r.tags.map((t) => t.id)), b = new Set(c.tags.map((t) => t.id)); const u = new Set([...a, ...b]).size;
  jaccard += u ? [...a].filter((x) => b.has(x)).length / u : 1;
}
deltas.sort((x, y) => x - y);
console.log(`${gate.cases.length} cases: flips=${flips.length} (${flips.join("; ") || "none"}); |Δrisk| max=${maxDelta.toFixed(3)} p95=${deltas[Math.floor(0.95 * deltas.length)].toFixed(3)} median=${deltas[Math.floor(deltas.length / 2)].toFixed(3)}; tag Jaccard=${(jaccard / gate.cases.length).toFixed(3)}`);
