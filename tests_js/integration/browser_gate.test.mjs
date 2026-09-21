/* The runtime gate against the DEVICE runtime (ADR D32): Python's verdicts on the 200 gate
   transcripts vs the JS scorer fed the browser's own embeddings (headless Chromium, WASM,
   captured by tests_js/tools/browser_gate_embed.mjs). Measured 2026-09-20: the int8 graph's
   browser embeddings sit at cosine 0.982 mean / 0.949 min to the server's, and on the shipped
   and the retrained heads alike the decisions differ on 5-6 / 200 with p95 |Δrisk| ~0.10-0.12
   and a max up to 0.5 on one code-switched legit call. The Node-based gate (embedding.test.mjs)
   had read 1-1.5 % because onnxruntime-node is native, like the server; the browser is not.
   Gate: same decision on >= 97 %, p95 |Δrisk| < 0.15, tag-set Jaccard >= 0.9; the max is
   reported, not gated (one transcript dominates it). Runs in seconds -- no model load. */
import test from "node:test";
import assert from "node:assert/strict";
import { existsSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { createScorer } from "../../site/core/score.js";
import { transcriptHash } from "../tools/browser_gate_embed.mjs";
import { REPO, config, weights } from "../helpers.mjs";

const GATE_PATH = join(REPO, "tests_js", "fixtures", "runtime_gate.json");
const EMB_PATH = join(REPO, "tests_js", "fixtures", "runtime_gate_browser.f32");
const META_PATH = join(REPO, "tests_js", "fixtures", "runtime_gate_browser.json");
const MAX_FLIP_RATE = 0.03;
const MAX_RISK_DELTA_P95 = 0.15;
const MIN_TAG_JACCARD = 0.9;
const MAX_RISK_DELTA_DEVICE = 1e-3;
const available = existsSync(GATE_PATH) && existsSync(EMB_PATH) && existsSync(META_PATH);

test("browser (WASM) embeddings reproduce Python's decisions on the 200-case gate", { skip: !available && "run `npm run gate:browser` to capture the browser embeddings" }, async () => {
  const gate = JSON.parse(readFileSync(GATE_PATH, "utf8"));
  const meta = JSON.parse(readFileSync(META_PATH, "utf8"));
  assert.equal(meta.transcript_hash, transcriptHash(gate.cases), "gate transcripts changed since the browser embeddings were captured: run `npm run gate:browser`");
  assert.equal(meta.model.model_id, config.embedder.model_id, "embedder changed: run `npm run gate:browser`");
  const buf = readFileSync(EMB_PATH);
  const flat = new Float32Array(buf.buffer, buf.byteOffset, buf.byteLength / 4);
  assert.equal(flat.length, meta.cases * meta.dim);
  const byText = new Map(gate.cases.map((c, i) => [c.transcript, Array.from(flat.subarray(i * meta.dim, (i + 1) * meta.dim))]));
  // The fixture holds transcript embeddings only. `score()` also embeds each utterance, but
  // those rows feed the highlight spans alone -- risk and tags come from the transcript row --
  // so utterances get the transcript's vector as a stand-in; spans are not gated here.
  const embed = async (texts) => {
    const transcript = byText.get(texts[0]);
    if (!transcript) throw new Error(`not a gate transcript: ${texts[0].slice(0, 40)}`);
    return texts.map(() => transcript);
  };
  const score = createScorer({ weights, config, embed });
  const thr = config.scoring.risk_threshold;
  const deltas = []; const flipped = []; let jaccard = 0;
  for (const c of gate.cases) {
    const r = await score(c.transcript);
    deltas.push(Math.abs(r.risk - c.risk));
    if ((r.risk >= thr) !== (c.risk >= thr)) flipped.push(`${c.id} py=${c.risk.toFixed(3)} browser=${r.risk.toFixed(3)}`);
    const a = new Set(r.tags.map((t) => t.id)), b = new Set(c.tags.map((t) => t.id));
    const union = new Set([...a, ...b]).size;
    jaccard += union ? [...a].filter((x) => b.has(x)).length / union : 1;
  }
  jaccard /= gate.cases.length;
  deltas.sort((x, y) => x - y);
  const p95 = deltas[Math.floor(0.95 * (deltas.length - 1))];
  const max = deltas[deltas.length - 1];
  console.log(`browser gate (transformers.js ${meta.transformers_js} / onnxruntime-web ${meta.onnxruntime_web}, ${meta.device}): flips=${flipped.length}/${gate.cases.length} (${flipped.join("; ") || "none"}), |Δrisk| p95=${p95.toFixed(3)} max=${max.toFixed(3)}, tag Jaccard=${jaccard.toFixed(3)}`);
  assert.ok(flipped.length <= Math.floor(MAX_FLIP_RATE * gate.cases.length), `decision flips ${flipped.length}/${gate.cases.length} exceed ${MAX_FLIP_RATE * 100} %: ${flipped.join("; ")}`);
  assert.ok(p95 < MAX_RISK_DELTA_P95, `p95 |Δrisk| ${p95}`);
  assert.ok(jaccard >= MIN_TAG_JACCARD, `tag-set Jaccard ${jaccard}`);
  if (gate.embed_backend === "device") {
    // Python's verdicts were computed on these very vectors (ADR D33): the JS scorer must
    // reproduce them to float precision -- this is the port's parity on real device input.
    assert.equal(flipped.length, 0, `device-trained heads must not flip on device embeddings: ${flipped.join("; ")}`);
    assert.ok(max < MAX_RISK_DELTA_DEVICE, `max |Δrisk| ${max} on device embeddings`);
  }
});
