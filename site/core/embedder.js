/* Main-thread handle on the embedding worker: `createWorkerEmbedder()` returns an
   `embed(texts)` matching the signature `score.js` expects, plus `warmup()` with
   progress events for the model-download UI. */

export function createWorkerEmbedder({ workerUrl = new URL("./embed-worker.js", import.meta.url), onProgress } = {}) {
  const worker = new Worker(workerUrl, { type: "module" });
  const pending = new Map();
  let nextId = 1;

  worker.onmessage = (event) => {
    const msg = event.data;
    if (msg.type === "progress" || msg.type === "info") {
      onProgress?.(msg);
      return;
    }
    const entry = pending.get(msg.id);
    if (!entry) return;
    pending.delete(msg.id);
    if (msg.type === "error") entry.reject(new Error(msg.message));
    else entry.resolve(msg.type === "result" ? msg.rows : true);
  };
  worker.onerror = (event) => {
    // A module worker that fails to load reports a bare ErrorEvent; say so usefully.
    const detail = event?.message || (event?.filename ? `${event.filename}:${event.lineno}` : "embedding worker failed to load (check the model files and the script imports)");
    const error = new Error(detail);
    for (const { reject } of pending.values()) reject(error);
    pending.clear();
  };

  const request = (payload) =>
    new Promise((resolve, reject) => {
      const id = nextId++;
      pending.set(id, { resolve, reject });
      worker.postMessage({ id, ...payload });
    });

  return {
    warmup: (device) => request({ type: "warmup", device }),
    embed: (texts) => request({ type: "embed", texts }),
    terminate: () => worker.terminate(),
  };
}
