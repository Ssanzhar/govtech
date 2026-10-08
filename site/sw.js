/* Service worker: offline after first load (PLAN_2026-09 B4).
   - app shell + core modules: network-first, the cache only when offline. (Stale-while-
     revalidate served a cached i18n.js/styles.css next to a fresh index.html after a deploy,
     so the landing showed raw keys like "landing.title_html" and unstyled radios);
   - model files under /models/ (278 MB ONNX + weights.json): cache-first, immutable;
     except /models/vosk/ (speech models, ~106 MB) which Vosklet caches itself;
   - /api/ and anything cross-origin: never cached, never intercepted -- the only network
     traffic with call content is the explicit report submit, and it must stay live. */

const SHELL_CACHE = "qorgan-shell-v9"; // v9: word-level voice mode + GigaAM endpoint pass (D63); v8: large binaries bypass the worker (D61); v7: reload pages a cache-first worker rendered; v6: network-first shell (no mixed versions after a deploy); v5: landing in kk/ru/en (landing.js, i18n-dom.js); v4: live page in kk/ru/en; v3: report review (D44); v2: COOP/COEP (B9)
// v2: the ONNX model is no longer kept here (transformers.js caches it itself); bumping the name
// deletes the v1 copy (~278 MB) on activation.
const MODEL_CACHE = "qorgan-models-v2";
const SHELL = [
  "/", "/index.html", "/live.html", "/styles.css", "/live.css", "/main.js", "/live.js", "/try.js",
  "/landing.js", "/i18n.js", "/i18n-dom.js", "/manifest.webmanifest",
  "/core/index.js", "/core/score.js", "/core/head.js", "/core/lexicon.js", "/core/attribution.js",
  "/core/explain.js", "/core/recommend.js", "/core/meter.js", "/core/session.js",
  "/core/embedder.js", "/core/embed-worker.js", "/core/qorgan-config.json", "/core/device.js",
  "/core/summary.js", "/core/asr.js", "/core/scenarios.json", "/core/cue-match.js", "/core/report.js",
  "/core/word-lid.js", "/core/span-vote.js", "/core/asr-wordlevel.json", "/core/endpoint.js",
  "/core/endpoint-runtime.js", "/core/audio-ring.js", "/core/gigaam.js", "/core/gigaam-worker.js", "/core/word-runs.js",
];

// The on-device speech runtimes (pinned, self-hosted by deploy_bootstrap); optional, so a
// deployment without microphone mode still installs the shell.
const OPTIONAL_SHELL = [
  "/vendor/vosklet/Vosklet.js", "/vendor/vosklet/Vosklet.wasm",
  "/vendor/sherpa-onnx/1.13.8/sherpa-onnx-wasm-web.js", "/vendor/sherpa-onnx/1.13.8/sherpa-onnx-asr.js",
];
// The GigaAM worker keeps its 225 MB model in its own Cache API store (ADR D63): never deleted here.
const OWN_MODEL_CACHE = /^qorgan-gigaam-/;

self.addEventListener("install", (event) => {
  event.waitUntil(
    caches.open(SHELL_CACHE)
      .then((cache) => cache.addAll(SHELL).then(() => Promise.allSettled(OPTIONAL_SHELL.map((url) => cache.add(url)))))
      .then(() => self.skipWaiting())
  );
});

const LARGE_BINARY = /\.(onnx|wasm)$/;

// Shells v1-v5 were served cache-first (stale-while-revalidate): a page open while this worker
// activates was rendered from that cache -- after a deploy, the analyst console's pre-sign-in
// admin.js ("Dashboard offline -- API returned 401"). Replacing such a worker reloads the open
// pages once, now network-first; replacing a network-first shell (v6+) reloads nothing.
const CACHE_FIRST_SHELL = /^qorgan-shell-v[1-5]$/;

self.addEventListener("activate", (event) => {
  const replaced = activate();
  event.waitUntil(replaced);
  // Outside waitUntil: a navigation is handled only once this worker has activated, so waiting
  // for it here would never finish (the page would hang on "Loading").
  replaced.then((stale) => { if (stale.some((k) => CACHE_FIRST_SHELL.test(k))) reloadPages(); });
});

async function activate() {
  const stale = (await caches.keys()).filter((k) => ![SHELL_CACHE, MODEL_CACHE].includes(k) && !OWN_MODEL_CACHE.test(k));
  await Promise.all(stale.map((k) => caches.delete(k)));
  await self.clients.claim();
  return stale;
}

async function reloadPages() {
  const pages = await self.clients.matchAll({ type: "window" });
  await Promise.allSettled(pages.map((page) => page.navigate(page.url)));
}

self.addEventListener("fetch", (event) => {
  const url = new URL(event.request.url);
  if (url.origin !== self.location.origin || event.request.method !== "GET") return;
  if (url.pathname.startsWith("/api/")) return; // live, never cached
  if (url.pathname.startsWith("/models/vosk/")) return; // the speech recogniser keeps its own model cache
  if (url.pathname.startsWith("/models/gigaam/")) return; // so does the GigaAM worker (ADR D63)
  // Large binaries go straight to the network (ADR D61). Firefox stops a service worker ~30 s into
  // an event and cuts off whatever it is still streaming: the 278 MB model failed with "Error in
  // input stream", and the 21 MB WASM runtimes can on a slow connection. transformers.js keeps
  // the model in its own Cache API store; the browser's HTTP cache holds the rest.
  if (LARGE_BINARY.test(url.pathname)) return;
  if (url.pathname.startsWith("/models/")) {
    event.respondWith(cacheFirst(MODEL_CACHE, event.request));
    return;
  }
  event.respondWith(networkFirst(SHELL_CACHE, event.request));
});

async function cacheFirst(cacheName, request) {
  const cache = await caches.open(cacheName);
  const hit = await cache.match(request);
  if (hit) return hit;
  const response = await fetch(request);
  if (response.ok) cache.put(request, response.clone());
  return response;
}

async function networkFirst(cacheName, request) {
  const cache = await caches.open(cacheName);
  try {
    // Sub-resources revalidate past the HTTP cache too (a navigation request rejects an init).
    const response = await (request.mode === "navigate" ? fetch(request) : fetch(request, { cache: "no-cache" }));
    if (response.ok) cache.put(request, response.clone());
    return response;
  } catch (err) {
    const hit = await cache.match(request);
    if (hit) return hit;
    throw err;
  }
}
