/* The device embedding bridge (PLAN B10, ADR D33): headless Chromium + the site's real
   embedding worker behind a tiny localhost HTTP API, so the Python pipeline can train and
   evaluate on the embeddings the citizen's browser computes:
     npm run device:serve [-- --port 8765 --pages 4]
     QORGAN_EMBED_BACKEND=device python -m qorgan.classifier.linear_train
   API:  GET  /info               -> { embedder, transformers_js, onnxruntime_web, device, user_agent }
         POST /embed {"texts":[…]} -> { "rows": [[…768 floats], …] }
   Texts arrive ALREADY prefixed (Python's `embed_texts` adds `query: `), so the worker runs
   with an empty prefix; the strings embedded are byte-identical to the site's. Loopback only,
   one request at a time (the worker is single-threaded anyway); no auth because it never
   leaves 127.0.0.1 and carries no data of its own. */
import { createServer } from "node:http";
import { fileURLToPath } from "node:url";
import { startDeviceRuntime } from "./device_runtime.mjs";

const DEFAULT_PORT = 8765;
const DEFAULT_PAGES = 4;  // tabs = parallel WASM workers; each text is still one graph run
const MAX_BODY_BYTES = 8 * 1024 * 1024;
const MAX_TEXTS = 256;

function readJson(req) {
  return new Promise((resolve, reject) => {
    const chunks = []; let size = 0;
    req.on("data", (c) => { size += c.length; if (size > MAX_BODY_BYTES) { reject(new Error("body too large")); req.destroy(); } chunks.push(c); });
    req.on("end", () => { try { resolve(JSON.parse(Buffer.concat(chunks).toString("utf8"))); } catch (e) { reject(e); } });
    req.on("error", reject);
  });
}

function send(res, status, payload) {
  res.writeHead(status, { "content-type": "application/json" });
  res.end(JSON.stringify(payload));
}

export async function startBridge({ port = DEFAULT_PORT, pages = DEFAULT_PAGES } = {}) {
  const runtime = await startDeviceRuntime({ prefix: "", pages });
  let queue = Promise.resolve();  // serialise embed calls: one page, one worker
  const server = createServer(async (req, res) => {
    try {
      if (req.method === "GET" && req.url === "/info") return send(res, 200, runtime.info());
      if (req.method === "POST" && req.url === "/embed") {
        const body = await readJson(req);
        const texts = body?.texts;
        if (!Array.isArray(texts) || texts.length === 0 || texts.length > MAX_TEXTS || !texts.every((t) => typeof t === "string" && t.length > 0)) {
          return send(res, 422, { error: `texts must be 1..${MAX_TEXTS} non-empty strings` });
        }
        const rows = await (queue = queue.then(() => runtime.embed(texts), () => runtime.embed(texts)));
        return send(res, 200, { rows });
      }
      send(res, 404, { error: "not found" });
    } catch (err) {
      send(res, 500, { error: err?.message || String(err) });
    }
  });
  await new Promise((resolve) => server.listen(port, "127.0.0.1", resolve));
  return { port: server.address().port, info: runtime.info(), close: async () => { server.close(); await runtime.close(); } };
}

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  const argValue = (flag, fallback) => { const i = process.argv.indexOf(flag); return i > -1 ? Number(process.argv[i + 1]) : fallback; };
  const bridge = await startBridge({ port: argValue("--port", DEFAULT_PORT), pages: argValue("--pages", DEFAULT_PAGES) });
  console.log(`device embedding bridge on http://127.0.0.1:${bridge.port} (${bridge.info.transformers_js} / onnxruntime-web ${bridge.info.onnxruntime_web}, ${bridge.info.device}, ${bridge.info.pages} tabs)`);
  const stop = async () => { await bridge.close(); process.exit(0); };
  process.on("SIGINT", stop); process.on("SIGTERM", stop);
}
