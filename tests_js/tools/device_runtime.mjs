/* The device runtime, driven from Node (ADR D32 / PLAN B10): a static server for site/, a
   headless Chromium page, and the site's REAL embedding worker (site/core/embed-worker.js,
   WASM) behind an `embed(texts)` function. This is the only way to obtain the embeddings the
   citizen's browser computes -- transformers.js in Node is native-only and drifts (cosine
   0.98) from onnxruntime-web. Used by browser_gate_embed.mjs (gate fixture) and
   device_embed_server.mjs (the bridge the Python `device` embed backend talks to). */
import { createReadStream, existsSync, readFileSync, statSync } from "node:fs";
import { createServer } from "node:http";
import { extname, join, normalize } from "node:path";
import { fileURLToPath } from "node:url";

export const REPO = join(fileURLToPath(new URL(".", import.meta.url)), "..", "..");
const SITE = join(REPO, "site");
const MIME = { ".html": "text/html", ".js": "text/javascript", ".mjs": "text/javascript", ".json": "application/json", ".onnx": "application/octet-stream", ".wasm": "application/wasm", ".css": "text/css" };
const DEVICE = "wasm";

/** Serve `root` (plus in-memory `extraFiles`, path -> JSON string) on an ephemeral port. */
export function serveSite(extraFiles = {}, root = SITE) {
  return createServer((req, res) => {
    const path = normalize(decodeURIComponent(new URL(req.url, "http://x").pathname));
    if (extraFiles[path] !== undefined) { res.writeHead(200, { "content-type": "application/json" }); res.end(extraFiles[path]); return; }
    const file = join(root, path === "/" ? "index.html" : path);
    if (!file.startsWith(root) || !existsSync(file) || statSync(file).isDirectory()) { res.writeHead(404); res.end(); return; }
    res.writeHead(200, { "content-type": MIME[extname(file)] || "application/octet-stream", "content-length": statSync(file).size });
    createReadStream(file).pipe(res);
  });
}

/** Versions of what the page will run, for fixtures and bundle metadata. */
export function runtimeVersions() {
  const transformers = JSON.parse(readFileSync(join(REPO, "node_modules", "@huggingface", "transformers", "package.json"), "utf8"));
  return { transformers_js: transformers.version, onnxruntime_web: transformers.dependencies["onnxruntime-web"], device: DEVICE };
}

/**
 * Start the device runtime. `prefix` overrides the worker's task prefix: the site adds
 * `query: ` itself, so a caller that already prefixes its texts (the Python `embed_texts`
 * path) passes "" to avoid doubling it. `pages` opens that many tabs, each with its own
 * worker: the WASM worker is single-threaded, so a batch is split across tabs and embedded
 * in parallel (each text is still ONE graph run -- batch size 1 is what makes the vector a
 * pure function of its text). Returns { embed(texts), info(), close() }.
 */
export async function startDeviceRuntime({ prefix, pages = 1, extraFiles = {}, onConsoleError } = {}) {
  const { chromium } = await import("playwright");
  const server = serveSite(extraFiles);
  await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
  const port = server.address().port;
  const browser = await chromium.launch();
  const tabs = [];
  let info;
  for (let i = 0; i < Math.max(1, pages); i += 1) {
    const page = await browser.newPage();
    page.on("console", (m) => { if (m.type() === "error" && !/onnxruntime/.test(m.text())) (onConsoleError || console.error)(`[browser] ${m.text()}`); });
    await page.goto(`http://127.0.0.1:${port}/index.html`);
    info = await page.evaluate(async (prefixOverride) => {
      const { createWorkerEmbedder } = await import("/core/embedder.js");
      const config = await (await fetch("/core/qorgan-config.json")).json();
      const embedder = prefixOverride === undefined ? config.embedder : { ...config.embedder, prefix: prefixOverride };
      window.__device = createWorkerEmbedder({ embedder });
      await window.__device.warmup();
      return { embedder, user_agent: navigator.userAgent };
    }, prefix);
    tabs.push(page);
  }
  const embedOn = (page, texts) => page.evaluate((t) => window.__device.embed(t).then((rows) => rows.map((r) => Array.from(r))), texts);
  return {
    info: () => ({ ...info, ...runtimeVersions(), port, pages: tabs.length }),
    async embed(texts) {
      if (tabs.length === 1) return embedOn(tabs[0], texts);
      const per = Math.ceil(texts.length / tabs.length);
      const slices = tabs.map((page, i) => ({ page, texts: texts.slice(i * per, (i + 1) * per) })).filter((s) => s.texts.length);
      const results = await Promise.all(slices.map((s) => embedOn(s.page, s.texts)));
      return results.flat();
    },
    close: async () => { await browser.close(); server.close(); },
  };
}
