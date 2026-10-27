/* Page side of the GigaAM endpoint pass (ADR D64): owns the dedicated worker
   (site/core/gigaam-worker.js) and turns its messages into promises. No network primitive here:
   the worker loads the self-hosted runtime and model itself; audio goes in by postMessage
   (transferred, never copied to anything but the worker) and only text comes back. */

export const DECODE_TIMEOUT_BASE_MS = 4000;
export const DECODE_TIMEOUT_PER_AUDIO_S = 2000; // 2x real time on top of the base: RTF 0.23 measured on a laptop

/** How long one phrase's decode may take once the worker has it (ADR D78: queue time never counts). */
export function decodeTimeoutMs(audioS, { baseMs = DECODE_TIMEOUT_BASE_MS, perAudioS = DECODE_TIMEOUT_PER_AUDIO_S } = {}) {
  return baseMs + audioS * perAudioS;
}

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

/** Start the worker and load the model. Returns `{ready, decode(samples, sampleRate), backlog,
    dispose()}`; `ready` resolves with the worker's load report or rejects (the caller then
    falls back to the Vosk vote). `onProgress({loaded, total})` reports the first download.
    The worker decodes one phrase at a time, so phrases wait HERE and go to it one by one: a
    phrase's timeout starts when the worker gets it (ADR D78). On a phone slower than real time
    phrases queue, and a queue-blind timer would drop every phrase that waited too long. */
export function createGigaamClient({ endpoint, pageUrl, onProgress = () => {}, WorkerImpl = globalThis.Worker, timeout = {} } = {}) {
  const worker = new WorkerImpl(new URL("./gigaam-worker.js", import.meta.url));
  const waiting = []; // [{id, samples, sampleRate, resolve, reject}], oldest first
  let current = null; // the phrase the worker has: {id, resolve, reject, timer}
  let nextId = 0;
  let readyResolve;
  let readyReject;
  const ready = new Promise((resolve, reject) => { readyResolve = resolve; readyReject = reject; });

  const failed = (message, reason) => Object.assign(new Error(message), { reason });

  const pump = () => {
    if (current || !waiting.length) return;
    const job = waiting.shift();
    const audioS = job.samples.length / job.sampleRate;
    const limitMs = decodeTimeoutMs(audioS, timeout);
    const timer = setTimeout(() => {
      if (current?.id !== job.id) return;
      current = null; // a late answer for this id is ignored below
      job.reject(failed(`GigaAM took longer than ${Math.round(limitMs)} ms`, "timeout"));
      pump();
    }, limitMs);
    current = { id: job.id, resolve: job.resolve, reject: job.reject, timer };
    worker.postMessage({ type: "decode", id: job.id, samples: job.samples, sampleRate: job.sampleRate }, [job.samples.buffer]);
  };

  const settle = (id, settleWith) => {
    if (current?.id !== id) return; // timed out already
    const job = current;
    current = null;
    clearTimeout(job.timer);
    settleWith(job);
    pump();
  };

  worker.onmessage = (event) => {
    const msg = event.data || {};
    if (msg.type === "progress") onProgress({ loaded: msg.loaded, total: msg.total });
    else if (msg.type === "ready") readyResolve(msg);
    else if (msg.type === "error" && msg.stage === "init") readyReject(new Error(msg.message));
    else if (msg.type === "result") settle(msg.id, (job) => job.resolve({ text: msg.text, decodeMs: msg.decodeMs, audioS: msg.audioS }));
    else if (msg.type === "error") settle(msg.id, (job) => job.reject(failed(msg.message, "failed")));
  };
  const rejectAll = (message) => {
    if (current) { clearTimeout(current.timer); current.reject(failed(message, "failed")); current = null; }
    for (const job of waiting.splice(0)) job.reject(failed(message, "failed"));
  };
  worker.onerror = (event) => {
    readyReject(new Error(event?.message || "the GigaAM worker failed to start"));
    rejectAll("worker error");
  };
  worker.postMessage({ type: "init", ...endpointUrls(endpoint, pageUrl) });

  return {
    ready,
    /** Phrases sent to `decode` and not yet answered (the one in the worker included). */
    get backlog() { return waiting.length + (current ? 1 : 0); },
    decode(samples, sampleRate) {
      return new Promise((resolve, reject) => {
        waiting.push({ id: nextId++, samples, sampleRate, resolve, reject });
        pump();
      });
    },
    dispose() {
      rejectAll("GigaAM was closed");
      worker.terminate();
    },
  };
}
