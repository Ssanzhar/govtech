/* The endpoint pass's decoder (ADR D63): GigaAM-Multilingual 220M CTC int8 in sherpa-onnx WASM.

   A dedicated, classic worker (the Emscripten glue is a classic script). Single-threaded build
   (scripts/sherpa_wasm/), so it needs neither SharedArrayBuffer nor cross-origin isolation.
   Everything it loads is SAME-ORIGIN and self-hosted: the runtime under /vendor/sherpa-onnx/
   (hash-pinned by deploy_bootstrap, asr/gigaam_web.py) and the model under /models/gigaam/.
   The only network primitive is `fetchSameOrigin`, which refuses any other origin; nothing is
   ever sent -- audio arrives by postMessage and only text goes back (tests/test_architecture.py).

   The 225 MB model is fetched once and kept in the Cache API (`cacheName`), then mounted from a
   Blob through WORKERFS rather than copied into the in-heap MEMFS.

   Protocol. In: {type: "init", runtimeBase, modelUrl, tokensUrl, cacheName}
                 {type: "decode", id, samples: Float32Array, sampleRate}
   Out: {type: "progress", loaded, total} · {type: "ready", loadMs, fromCache, heapMb}
        {type: "result", id, text, decodeMs, audioS} · {type: "error", id?, message} */

/* global importScripts, SherpaOnnx, OfflineRecognizer */
"use strict";

const MOUNT = "/gigaam";
const FEATURE_DIM = 64; // GigaAM's 64-bin log-mel front end
const MODEL_SAMPLE_RATE = 16000; // sherpa resamples other rates itself

let recognizer = null;
let wasm = null;

function sameOrigin(url) {
  const resolved = new URL(url, self.location.href);
  if (resolved.origin !== self.location.origin) throw new Error(`refusing a cross-origin load: ${resolved.origin}`);
  return resolved.href;
}

/** The worker's only network primitive: GET of a self-hosted file. */
function fetchSameOrigin(url) {
  return fetch(sameOrigin(url), { credentials: "same-origin" });
}

async function readWithProgress(response) {
  const total = Number(response.headers.get("Content-Length") || 0);
  const reader = response.body.getReader();
  const chunks = [];
  let loaded = 0;
  let lastPost = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    loaded += value.length;
    if (loaded - lastPost > 4 * 1024 * 1024) {
      lastPost = loaded;
      self.postMessage({ type: "progress", loaded, total });
    }
  }
  return new Blob(chunks);
}

/** The model as a Blob: from the Cache API when stored, else downloaded once and stored. */
async function cachedBlob(url, cacheName) {
  const href = sameOrigin(url);
  let cache = null;
  try { cache = await caches.open(cacheName); } catch { /* storage blocked: download every time */ }
  const hit = cache ? await cache.match(href) : null;
  if (hit) return { blob: await hit.blob(), fromCache: true };
  const response = await fetchSameOrigin(href);
  if (!response.ok) throw new Error(`HTTP ${response.status} for ${new URL(href).pathname}`);
  const blob = await readWithProgress(response);
  if (cache) {
    try {
      await cache.put(href, new Response(blob, { headers: { "Content-Type": "application/octet-stream" } }));
      // An older model version's store is dead weight (the service worker never deletes these).
      for (const name of await caches.keys()) if (name.startsWith("qorgan-gigaam-") && name !== cacheName) await caches.delete(name);
      const stored = await cache.match(href); // a disk-backed Blob instead of the in-memory chunks
      if (stored) return { blob: await stored.blob(), fromCache: false };
    } catch { /* quota: keep the in-memory copy for this session */ }
  }
  return { blob, fromCache: false };
}

async function init({ runtimeBase, modelUrl, tokensUrl, cacheName }) {
  const started = performance.now();
  const base = sameOrigin(runtimeBase);
  importScripts(sameOrigin(`${base}sherpa-onnx-asr.js`), sameOrigin(`${base}sherpa-onnx-wasm-web.js`));
  wasm = await SherpaOnnx({ locateFile: (path) => sameOrigin(`${base}${path}`) });
  const [{ blob: model, fromCache }, tokensResponse] = await Promise.all([cachedBlob(modelUrl, cacheName), fetchSameOrigin(tokensUrl)]);
  if (!tokensResponse.ok) throw new Error(`HTTP ${tokensResponse.status} for the GigaAM tokens`);
  const tokens = await tokensResponse.blob();
  wasm.FS.mkdir(MOUNT);
  wasm.FS.mount(wasm.FS.filesystems.WORKERFS, { blobs: [{ name: "model.int8.onnx", data: model }, { name: "tokens.txt", data: tokens }] }, MOUNT);
  recognizer = new OfflineRecognizer({
    featConfig: { sampleRate: MODEL_SAMPLE_RATE, featureDim: FEATURE_DIM },
    modelConfig: { nemoCtc: { model: `${MOUNT}/model.int8.onnx` }, tokens: `${MOUNT}/tokens.txt`, numThreads: 1, provider: "cpu", debug: 0 },
    decodingMethod: "greedy_search",
  }, wasm);
  if (!recognizer.handle) throw new Error("sherpa-onnx could not create the GigaAM recogniser");
  self.postMessage({ type: "ready", loadMs: performance.now() - started, fromCache, heapMb: wasm.HEAPU8.length / 1048576 });
}

function decode({ id, samples, sampleRate }) {
  const started = performance.now();
  const stream = recognizer.createStream();
  try {
    stream.acceptWaveform(sampleRate, samples);
    recognizer.decode(stream);
    const { text } = recognizer.getResult(stream);
    self.postMessage({ type: "result", id, text: String(text || "").trim(), decodeMs: performance.now() - started, audioS: samples.length / sampleRate });
  } finally {
    stream.free();
  }
}

self.onmessage = async (event) => {
  const msg = event.data || {};
  try {
    if (msg.type === "init") await init(msg);
    else if (msg.type === "decode") {
      if (!recognizer) throw new Error("the GigaAM recogniser is not loaded");
      decode(msg);
    } else throw new Error(`unknown message: ${msg.type}`);
  } catch (err) {
    self.postMessage({ type: "error", id: msg.id ?? null, stage: msg.type, message: String(err?.message || err) });
  }
};
