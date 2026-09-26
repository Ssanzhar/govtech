/* Service worker: offline after first load (PLAN_2026-09 B4).
   - app shell + core modules: cache-first, refreshed in the background;
   - model files under /models/ (278 MB ONNX + weights.json): cache-first, immutable;
     except /models/vosk/ (speech models, ~106 MB) which Vosklet caches itself;
   - /api/ and anything cross-origin: never cached, never intercepted -- the only network
     traffic with call content is the explicit report submit, and it must stay live. */

const SHELL_CACHE = "qorgan-shell-v4"; // v4: live page in kk/ru/en (i18n.js); v3: report review (D44); v2: COOP/COEP (B9)
const MODEL_CACHE = "qorgan-models-v1";
const SHELL = [
  "/", "/index.html", "/live.html", "/styles.css", "/live.css", "/main.js", "/live.js", "/try.js",
  "/i18n.js", "/manifest.webmanifest",
  "/core/index.js", "/core/score.js", "/core/head.js", "/core/lexicon.js", "/core/attribution.js",
  "/core/explain.js", "/core/recommend.js", "/core/meter.js", "/core/session.js",
  "/core/embedder.js", "/core/embed-worker.js", "/core/qorgan-config.json", "/core/device.js",
  "/core/summary.js", "/core/asr.js", "/core/scenarios.json", "/core/cue-match.js", "/core/report.js",
];

// The on-device speech runtime (pinned, self-hosted by deploy_bootstrap); optional, so a
// deployment without microphone mode still installs the shell.
const OPTIONAL_SHELL = ["/vendor/vosklet/Vosklet.js", "/vendor/vosklet/Vosklet.wasm"];

self.addEventListener("install", (event) => {
  event.waitUntil(
    caches.open(SHELL_CACHE)
      .then((cache) => cache.addAll(SHELL).then(() => Promise.allSettled(OPTIONAL_SHELL.map((url) => cache.add(url)))))
      .then(() => self.skipWaiting())
  );
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    caches.keys().then((keys) => Promise.all(keys.filter((k) => ![SHELL_CACHE, MODEL_CACHE].includes(k)).map((k) => caches.delete(k))))
      .then(() => self.clients.claim()),
  );
});

self.addEventListener("fetch", (event) => {
  const url = new URL(event.request.url);
  if (url.origin !== self.location.origin || event.request.method !== "GET") return;
  if (url.pathname.startsWith("/api/")) return; // live, never cached
  if (url.pathname.startsWith("/models/vosk/")) return; // the speech recogniser keeps its own model cache
  if (url.pathname.startsWith("/models/")) {
    event.respondWith(cacheFirst(MODEL_CACHE, event.request));
    return;
  }
  event.respondWith(staleWhileRevalidate(SHELL_CACHE, event.request));
});

async function cacheFirst(cacheName, request) {
  const cache = await caches.open(cacheName);
  const hit = await cache.match(request);
  if (hit) return hit;
  const response = await fetch(request);
  if (response.ok) cache.put(request, response.clone());
  return response;
}

async function staleWhileRevalidate(cacheName, request) {
  const cache = await caches.open(cacheName);
  const hit = await cache.match(request);
  const refresh = fetch(request).then((response) => {
    if (response.ok) cache.put(request, response.clone());
    return response;
  }).catch(() => hit);
  return hit || refresh;
}
