/* A static server for site/ with cross-origin isolation ON or OFF (ADR D67 evidence runs).

     node tests_js/tools/static_site.mjs --port 8410 --isolation off   # what an Android WebView gets
     node tests_js/tools/static_site.mjs --port 8411 --isolation on    # the production headers

   `on` sends COOP/COEP exactly where the real server does (src/qorgan/api_limits.py:
   /live.html, /core/, /vendor/), so the page is crossOriginIsolated; `off` sends neither, which
   is what a page inside Android WebView experiences whatever the server sends (WebView never
   grants cross-origin isolation). Loopback only, GET/HEAD only, no API routes: the citizen page
   needs none to listen, transcribe and score. */
import { createReadStream, statSync } from "node:fs";
import { createServer } from "node:http";
import { extname, join, normalize, resolve, sep } from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = resolve(fileURLToPath(new URL("../../site/", import.meta.url)));
const ISOLATED_PATHS = new Set(["/live.html"]);
const ISOLATED_PREFIXES = ["/core/", "/vendor/"];
const TYPES = {
  ".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8", ".mjs": "text/javascript; charset=utf-8",
  ".css": "text/css; charset=utf-8", ".json": "application/json", ".webmanifest": "application/manifest+json",
  ".wasm": "application/wasm", ".onnx": "application/octet-stream", ".gz": "application/gzip", ".txt": "text/plain; charset=utf-8",
  ".png": "image/png", ".svg": "image/svg+xml", ".ico": "image/x-icon",
};

export function startStaticSite({ port = 8410, isolation = "off", host = "127.0.0.1" } = {}) {
  const server = createServer((req, res) => {
    if (req.method !== "GET" && req.method !== "HEAD") { res.writeHead(405).end(); return; }
    const path = decodeURIComponent(new URL(req.url, "http://x").pathname);
    const rel = path === "/" ? "/index.html" : path;
    const file = normalize(join(ROOT, rel));
    if (!file.startsWith(ROOT + sep)) { res.writeHead(403).end(); return; }
    let stat;
    try { stat = statSync(file); } catch { res.writeHead(404).end(); return; }
    if (!stat.isFile()) { res.writeHead(404).end(); return; }
    const headers = { "content-type": TYPES[extname(file)] || "application/octet-stream", "content-length": stat.size, "cache-control": "no-cache" };
    if (isolation === "on" && (ISOLATED_PATHS.has(rel) || ISOLATED_PREFIXES.some((p) => rel.startsWith(p)))) {
      headers["cross-origin-opener-policy"] = "same-origin";
      headers["cross-origin-embedder-policy"] = "require-corp";
    }
    res.writeHead(200, headers);
    if (req.method === "HEAD") { res.end(); return; }
    createReadStream(file).pipe(res);
  });
  return new Promise((ok) => server.listen(port, host, () => ok(server)));
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const args = process.argv.slice(2);
  const opt = (name, fallback) => { const i = args.indexOf(name); return i > -1 ? args[i + 1] : fallback; };
  const isolation = opt("--isolation", "off");
  if (!["on", "off"].includes(isolation)) throw new Error("--isolation on|off");
  const port = Number(opt("--port", isolation === "on" ? "8411" : "8410"));
  await startStaticSite({ port, isolation });
  console.log(`site/ on http://127.0.0.1:${port} (cross-origin isolation ${isolation})`);
}
