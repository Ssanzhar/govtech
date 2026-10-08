import test from "node:test";
import assert from "node:assert/strict";
import { readdirSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { REPO } from "./helpers.mjs";

// The PWA promise is "offline after first load" (PLAN B4): every on-device core module the
// pages can import must be in the service worker's precached shell.
test("every site/core module is precached by the service worker", () => {
  const sw = readFileSync(join(REPO, "site", "sw.js"), "utf8");
  const shell = new Set([...sw.matchAll(/"(\/[^"]+)"/g)].map((m) => m[1]));
  const core = readdirSync(join(REPO, "site", "core"))
    .filter((name) => /\.(js|json)$/.test(name))
    .map((name) => `/core/${name}`);
  const missing = core.filter((path) => !shell.has(path));
  assert.deepEqual(missing, [], `add to SHELL in site/sw.js: ${missing.join(", ")}`);
});

// A module a precached page script imports statically must be precached too, or the page
// breaks offline (live.js imports i18n.js and core modules at load time).
test("every module a precached page script imports is precached", () => {
  const sw = readFileSync(join(REPO, "site", "sw.js"), "utf8");
  const shell = new Set([...sw.matchAll(/"(\/[^"]+)"/g)].map((m) => m[1]));
  const pages = [...shell].filter((path) => /^\/[\w-]+\.js$/.test(path) && path !== "/sw.js");
  assert.ok(pages.includes("/live.js"));
  const missing = [];
  for (const page of pages) {
    const source = readFileSync(join(REPO, "site", page.slice(1)), "utf8");
    for (const m of source.matchAll(/(?:^|\n)\s*import\s[^;]*?from\s+"\.\/([^"]+)"/g)) {
      if (!shell.has(`/${m[1]}`)) missing.push(`${page} -> /${m[1]}`);
    }
  }
  assert.deepEqual(missing, [], `add to SHELL in site/sw.js: ${missing.join(", ")}`);
});

// Run sw.js's activate handler in a minimal fake ServiceWorkerGlobalScope.
async function activateWith(cacheNames) {
  const handlers = {};
  const deleted = [];
  const navigated = [];
  // As in a browser: a navigation's fetch is handled only once activation has finished, so
  // navigate() settles only after it. A worker that awaits navigate() inside waitUntil never
  // activates (the deadlock that left the page "Loading").
  let activated;
  const activation = new Promise((resolve) => { activated = resolve; });
  const navigate = (url) => activation.then(() => { navigated.push(url); });
  const windows = [{ url: "https://host/admin.html", navigate }, { url: "https://host/live.html", navigate }];
  const self = {
    location: { origin: "https://host" },
    addEventListener: (type, fn) => { handlers[type] = fn; },
    skipWaiting: async () => {},
    clients: { claim: async () => {}, matchAll: async () => windows },
  };
  const caches = {
    keys: async () => cacheNames,
    delete: async (name) => { deleted.push(name); return true; },
  };
  const source = readFileSync(join(REPO, "site", "sw.js"), "utf8");
  new Function("self", "caches", "fetch", "URL", source)(self, caches, async () => {}, URL);
  let done;
  handlers.activate({ waitUntil: (promise) => { done = promise; } });
  const timeout = new Promise((_, reject) => setTimeout(() => reject(new Error("activation never finished (deadlock)")), 500));
  await Promise.race([done, timeout]);
  activated();
  await new Promise((resolve) => setTimeout(resolve, 0)); // navigations scheduled after activation
  return { deleted, navigated };
}

// A v1-v5 worker served the shell cache-first: the page open while the new worker activates
// was rendered from that cache (a pre-sign-in admin.js answered "Dashboard offline -- API
// returned 401"). Replacing such a worker reloads the open pages once, from the network.
test("replacing a cache-first (v1-v5) shell reloads the open pages", async () => {
  const { deleted, navigated } = await activateWith(["qorgan-shell-v4", "qorgan-models-v1"]);
  assert.deepEqual(deleted, ["qorgan-shell-v4", "qorgan-models-v1"]); // models-v1 held the ONNX copy (D61)
  assert.deepEqual(navigated, ["https://host/admin.html", "https://host/live.html"]);
});

// ADR D63: the GigaAM worker keeps the 225 MB endpoint model in its own Cache API store; a
// service-worker update must not delete it (the citizen would download it again).
test("activation keeps the GigaAM model cache", async () => {
  const { deleted } = await activateWith(["qorgan-shell-v8", "qorgan-gigaam-gigaam-multilingual-220m-ctc-int8-v1"]);
  assert.deepEqual(deleted, ["qorgan-shell-v8"]);
});

test("the GigaAM model and runtime binaries bypass the service worker", () => {
  const handlers = {};
  const self = { location: { origin: "https://host" }, addEventListener: (t, fn) => { handlers[t] = fn; }, skipWaiting() {}, clients: {} };
  const source = readFileSync(join(REPO, "site", "sw.js"), "utf8");
  const cache = { match: async () => undefined, put: async () => {} };
  new Function("self", "caches", "fetch", "URL", source)(self, { open: async () => cache }, async () => ({ ok: false }), URL);
  const intercepted = (path) => {
    let responded = false;
    handlers.fetch({ request: { url: `https://host${path}`, method: "GET", mode: "cors" }, respondWith: (p) => { responded = true; Promise.resolve(p).catch(() => {}); } });
    return responded;
  };
  assert.equal(intercepted("/models/gigaam/gigaam-multilingual-220m-ctc-int8/model.int8.onnx"), false);
  assert.equal(intercepted("/models/gigaam/gigaam-multilingual-220m-ctc-int8/tokens.txt"), false);
  assert.equal(intercepted("/vendor/sherpa-onnx/1.13.8/sherpa-onnx-wasm-web.wasm"), false);
});

test("a first install or a network-first predecessor reloads nothing", async () => {
  assert.deepEqual((await activateWith([])).navigated, []);
  assert.deepEqual((await activateWith(["qorgan-shell-v6", "qorgan-models-v2"])).navigated, []);
});

// Firefox stops a service worker ~30 s into an event, cutting off any response it is still
// streaming. The 278 MB model routed through the worker failed on prod with "Error in input
// stream" after ~32 s (it loads with service workers blocked). Large binaries bypass the
// worker; transformers.js keeps the model in its own Cache API store (ADR D61).
test("large binaries are never streamed through the service worker", () => {
  const handlers = {};
  const self = { location: { origin: "https://host" }, addEventListener: (t, fn) => { handlers[t] = fn; }, skipWaiting() {}, clients: {} };
  const source = readFileSync(join(REPO, "site", "sw.js"), "utf8");
  const cache = { match: async () => undefined, put: async () => {} };
  const caches = { open: async () => cache };
  new Function("self", "caches", "fetch", "URL", source)(self, caches, async () => ({ ok: false }), URL);
  const intercepted = (path) => {
    let responded = false;
    const respondWith = (promise) => { responded = true; Promise.resolve(promise).catch(() => {}); };
    handlers.fetch({ request: { url: `https://host${path}`, method: "GET", mode: "cors" }, respondWith });
    return responded;
  };
  assert.equal(intercepted("/models/Xenova/multilingual-e5-base/onnx/model_quantized.onnx"), false);
  assert.equal(intercepted("/vendor/transformers/3.8.1/ort-wasm-simd-threaded.jsep.wasm"), false);
  assert.equal(intercepted("/vendor/vosklet/Vosklet.wasm"), false);
  assert.equal(intercepted("/live.js"), true, "the shell stays network-first through the worker");
  assert.equal(intercepted("/models/weights.json"), true);
});
