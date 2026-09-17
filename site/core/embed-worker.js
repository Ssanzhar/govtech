/* Embedding worker: loads `multilingual-e5-base` (int8 ONNX) with transformers.js and
   answers `embed` requests with mean-pooled, L2-normalised vectors -- the exact recipe of
   `qorgan/classifier/embed.py` (`query: ` prefix, mean pooling, normalize). Runs off the
   main thread; nothing here ever touches the network except fetching model files from
   this site's own /models/ path (self-hosted, cached by the browser). */

// Pinned to 3.8.1: the 4.2.0 browser bundle throws "this.tokenizer is not a function" for
// feature-extraction (verified in Chromium). `transformers.min.js` is the self-contained bundle;
// `transformers.web.js`
// is the bundler entry with bare `onnxruntime-web` imports and cannot be imported directly.
import { env, pipeline } from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1/dist/transformers.min.js";

// Model id / dtype / prefix arrive with the first request (from qorgan-config.json, which is
// generated from the server's own embedder settings -- ADR D17: one graph on both sides).
let settings = { model_id: "Xenova/multilingual-e5-base", dtype: "q8", prefix: "query: " };

env.allowRemoteModels = false;   // never fall back to huggingface.co from the client
env.allowLocalModels = true;
env.localModelPath = new URL("../models/", import.meta.url).href;

let extractorPromise = null;

async function loadExtractor(preferredDevice) {
  const device = preferredDevice || (typeof navigator !== "undefined" && "gpu" in navigator ? "webgpu" : "wasm");
  const progress_callback = (p) => postMessage({ type: "progress", ...p });
  const { model_id, dtype } = settings;
  try {
    return await pipeline("feature-extraction", model_id, { dtype, device, progress_callback });
  } catch (err) {
    if (device === "webgpu") {
      postMessage({ type: "info", message: `webgpu unavailable (${err?.message || err}); falling back to wasm` });
      return pipeline("feature-extraction", model_id, { dtype, device: "wasm", progress_callback });
    }
    throw err;
  }
}

// ONE text per graph run, deliberately: the int8 graph is dynamically quantised and the
// activation scales are derived over the whole batched tensor (padding included), so a
// batched embedding would depend on its neighbours. The Python server does the same
// (`classifier/embed.py::ONNX_BATCH_SIZE = 1`); that is what makes server == device.
async function embed(texts) {
  if (!extractorPromise) extractorPromise = loadExtractor();
  const extractor = await extractorPromise;
  const rows = [];
  for (const text of texts) {
    const output = await extractor([settings.prefix + text], { pooling: "mean", normalize: true });
    rows.push(Array.from(output.data));
  }
  return rows;
}

self.onmessage = async (event) => {
  const { id, type, texts, device, embedder } = event.data;
  try {
    if (embedder && !extractorPromise) settings = { ...settings, ...embedder };
    if (type === "warmup") {
      if (!extractorPromise) extractorPromise = loadExtractor(device);
      await extractorPromise;
      postMessage({ id, type: "ready" });
    } else if (type === "embed") {
      postMessage({ id, type: "result", rows: await embed(texts) });
    }
  } catch (err) {
    postMessage({ id, type: "error", message: err?.message || String(err) });
  }
};
