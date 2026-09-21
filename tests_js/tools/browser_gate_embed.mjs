/* Regenerate the DEVICE-side embeddings of the runtime gate (ADR D32):
     npm run gate:browser            (needs `npx playwright install chromium` once)
   Embeds every gate transcript through the site's real worker in headless Chromium
   (tests_js/tools/device_runtime.mjs) and writes tests_js/fixtures/runtime_gate_browser.f32
   (Float32 rows in gate order) + .json sidecar (transcript hash, runtime versions). The gate
   test then runs in seconds, off the fixture, against whatever heads are exported. */
import { createHash } from "node:crypto";
import { readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { REPO, startDeviceRuntime } from "./device_runtime.mjs";

const GATE_PATH = join(REPO, "tests_js", "fixtures", "runtime_gate.json");
const OUT_F32 = join(REPO, "tests_js", "fixtures", "runtime_gate_browser.f32");
const OUT_META = join(REPO, "tests_js", "fixtures", "runtime_gate_browser.json");
const CHUNK = 25;
const RECORD_SEP = "\n--record--\n";
const FIELD_SEP = "\n--field--\n";

/** Fingerprint of the gate transcripts; the gate test refuses stale browser embeddings. */
export function transcriptHash(cases) {
  return createHash("sha256").update(cases.map((c) => `${c.id}${FIELD_SEP}${c.transcript}`).join(RECORD_SEP)).digest("hex");
}

async function main() {
  const gate = JSON.parse(readFileSync(GATE_PATH, "utf8"));
  const runtime = await startDeviceRuntime();
  try {
    const t0 = performance.now();
    const rows = [];
    for (let i = 0; i < gate.cases.length; i += CHUNK) {
      rows.push(...(await runtime.embed(gate.cases.slice(i, i + CHUNK).map((c) => c.transcript))));
    }
    const msPerTranscript = Math.round((performance.now() - t0) / gate.cases.length);
    const dim = rows[0].length;
    const flat = new Float32Array(rows.length * dim);
    rows.forEach((row, i) => flat.set(row, i * dim));
    writeFileSync(OUT_F32, Buffer.from(flat.buffer));
    const info = runtime.info();
    writeFileSync(OUT_META, JSON.stringify({
      cases: rows.length, dim, device: info.device, transcript_hash: transcriptHash(gate.cases),
      transformers_js: info.transformers_js, onnxruntime_web: info.onnxruntime_web,
      model: info.embedder, user_agent: info.user_agent, ms_per_transcript: msPerTranscript, generated_at: new Date().toISOString(),
    }, null, 2) + "\n");
    console.log(`wrote ${rows.length}x${dim} browser embeddings -> ${OUT_F32} (${msPerTranscript} ms/transcript)`);
  } finally {
    await runtime.close();
  }
}

if (process.argv[1] === fileURLToPath(import.meta.url)) await main();
