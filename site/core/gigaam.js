/* Page side of the GigaAM endpoint pass (ADR D64): owns the dedicated worker
   (site/core/gigaam-worker.js) and turns its messages into promises. No network primitive here:
   the worker loads the self-hosted runtime and model itself; audio goes in by postMessage
   (transferred, never copied to anything but the worker) and only text comes back. */

export const DECODE_TIMEOUT_BASE_MS = 4000;
export const DECODE_TIMEOUT_PER_AUDIO_S = 2000; // 2x real time on top of the base: RTF 0.23 measured on a laptop

/** Absolute same-origin URLs the worker needs, from `asr.endpoint` of qorgan-config.json. */
export function endpointUrls(endpoint, pageUrl, modelsBase = "models/") {
  const at = (path) => new URL(path, pageUrl).href;
  return {
    runtimeBase: at(endpoint.runtime),
    modelUrl: at(modelsBase + endpoint.model.url),
    tokensUrl: at(modelsBase + endpoint.model.tokens),
    cacheName: endpoint.cache,
  };
}

/** Start the worker and load the model. Returns `{ready, decode(samples, sampleRate),
    dispose()}`; `ready` resolves with the worker's load report or rejects (the caller then
    falls back to the Vosk vote). `onProgress({loaded, total})` reports the first download. */
export function createGigaamClient({ endpoint, pageUrl, onProgress = () => {}, WorkerImpl = globalThis.Worker } = {}) {
  const worker = new WorkerImpl(new URL("./gigaam-worker.js", import.meta.url));
  const pending = new Map();
  let nextId = 0;
  let readyResolve;
  let readyReject;
  const ready = new Promise((resolve, reject) => { readyResolve = resolve; readyReject = reject; });

  worker.onmessage = (event) => {
    const msg = event.data || {};
    if (msg.type === "progress") onProgress({ loaded: msg.loaded, total: msg.total });
    else if (msg.type === "ready") readyResolve(msg);
    else if (msg.type === "error" && msg.stage === "init") readyReject(new Error(msg.message));
    else if (msg.type === "result" || msg.type === "error") {
      const entry = pending.get(msg.id);
      if (!entry) return; // timed out already
      pending.delete(msg.id);
      clearTimeout(entry.timer);
      if (msg.type === "result") entry.resolve({ text: msg.text, decodeMs: msg.decodeMs, audioS: msg.audioS });
      else entry.reject(Object.assign(new Error(msg.message), { reason: "failed" }));
    }
  };
  worker.onerror = (event) => {
    readyReject(new Error(event?.message || "the GigaAM worker failed to start"));
    for (const [id, entry] of pending) { clearTimeout(entry.timer); entry.reject(Object.assign(new Error("worker error"), { reason: "failed" })); pending.delete(id); }
  };
  worker.postMessage({ type: "init", ...endpointUrls(endpoint, pageUrl) });

  return {
    ready,
    decode(samples, sampleRate) {
      const id = nextId++;
      const audioS = samples.length / sampleRate;
      return new Promise((resolve, reject) => {
        const timer = setTimeout(() => {
          pending.delete(id);
          reject(Object.assign(new Error(`GigaAM took longer than ${Math.round(DECODE_TIMEOUT_BASE_MS + audioS * DECODE_TIMEOUT_PER_AUDIO_S)} ms`), { reason: "timeout" }));
        }, DECODE_TIMEOUT_BASE_MS + audioS * DECODE_TIMEOUT_PER_AUDIO_S);
        pending.set(id, { resolve, reject, timer });
        worker.postMessage({ type: "decode", id, samples, sampleRate }, [samples.buffer]);
      });
    },
    dispose() {
      for (const entry of pending.values()) clearTimeout(entry.timer);
      pending.clear();
      worker.terminate();
    },
  };
}
