/* Main-thread handle on the embedding worker: `createWorkerEmbedder()` returns an
   `embed(texts)` matching the signature `score.js` expects, plus `warmup()` with
   progress events for the model-download UI. */

// Per text: the worker embeds a batch one text at a time, milliseconds to a few seconds each on a slow
// phone, so this only catches a hung worker. The warmup has no such limit: it may be a 278 MB download.
export const EMBED_TIMEOUT_PER_TEXT_MS = 60_000;

export function createWorkerEmbedder({ workerUrl = new URL("./embed-worker.js", import.meta.url), onProgress, embedder, WorkerImpl = globalThis.Worker, embedTimeoutMs = EMBED_TIMEOUT_PER_TEXT_MS } = {}) {
  const worker = new WorkerImpl(workerUrl, { type: "module" });
  const pending = new Map();
  let nextId = 1;
  let dead = null; // the error every later request gets, once the worker is gone

  const failAll = (error) => {
    dead = error;
    for (const { reject, timer } of pending.values()) {
      clearTimeout(timer);
      reject(error);
    }
    pending.clear();
  };

  worker.onmessage = (event) => {
    const msg = event.data;
    if (msg.type === "progress" || msg.type === "info") {
      onProgress?.(msg);
      return;
    }
    const entry = pending.get(msg.id);
    if (!entry) return;
    pending.delete(msg.id);
    clearTimeout(entry.timer);
    if (msg.type === "error") entry.reject(new Error(msg.message));
    else entry.resolve(msg.type === "result" ? msg.rows : true);
  };
  worker.onerror = (event) => {
    // A module worker that fails to load reports a bare ErrorEvent; say so usefully.
    const detail = event?.message || (event?.filename ? `${event.filename}:${event.lineno}` : "embedding worker failed to load (check the model files and the script imports)");
    failAll(new Error(detail));
  };

  const request = (payload, timeoutMs = null) =>
    new Promise((resolve, reject) => {
      if (dead) {
        reject(dead);
        return;
      }
      const id = nextId++;
      const timer = timeoutMs === null ? null : setTimeout(() => {
        worker.terminate();
        failAll(new Error("the embedding worker stopped answering"));
      }, timeoutMs);
      pending.set(id, { resolve, reject, timer });
      worker.postMessage({ id, ...payload });
    });

  return {
    warmup: (device) => request({ type: "warmup", device, embedder }),
    embed: (texts) => request({ type: "embed", texts, embedder }, embedTimeoutMs * Math.max(1, texts.length)),
    /** False once the worker is gone: every later request fails, so the page builds a new runtime. */
    alive: () => dead === null,
    terminate: () => {
      worker.terminate();
      failAll(new Error("the embedding worker was closed"));
    },
  };
}
