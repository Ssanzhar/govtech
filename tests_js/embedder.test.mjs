/* The embedding worker's handle: no promise may stay unsettled, or the Stop button waits for ever. */
import test from "node:test";
import assert from "node:assert/strict";
import { createWorkerEmbedder } from "../site/core/embedder.js";

function fakeWorker({ answer = true } = {}) {
  return class FakeWorker {
    constructor() { FakeWorker.instance = this; FakeWorker.terminated = 0; }
    postMessage(msg) {
      if (answer) queueMicrotask(() => this.onmessage({ data: { id: msg.id, type: msg.type === "embed" ? "result" : "ready", rows: [[1]] } }));
    }
    terminate() { FakeWorker.terminated += 1; }
  };
}

const URL_STUB = "x";

test("a healthy worker answers", async () => {
  const embedder = createWorkerEmbedder({ workerUrl: URL_STUB, WorkerImpl: fakeWorker() });
  await embedder.warmup();
  assert.deepEqual(await embedder.embed(["a"]), [[1]]);
  embedder.terminate();
});

test("a worker that dies rejects what was pending and everything after", async () => {
  const WorkerImpl = fakeWorker({ answer: false });
  const embedder = createWorkerEmbedder({ workerUrl: URL_STUB, WorkerImpl });
  const pending = embedder.embed(["a"]);
  WorkerImpl.instance.onerror({ message: "out of memory" });
  await assert.rejects(pending, /out of memory/);
  await assert.rejects(embedder.embed(["b"]), /out of memory/, "a new request must not wait on a dead worker");
});

test("closing the embedder settles what was pending", async () => {
  const embedder = createWorkerEmbedder({ workerUrl: URL_STUB, WorkerImpl: fakeWorker({ answer: false }) });
  const pending = embedder.embed(["a"]);
  embedder.terminate();
  await assert.rejects(pending, /closed/);
});

test("a worker that stops answering is given up on; the model download is not timed", async () => {
  const WorkerImpl = fakeWorker({ answer: false });
  const embedder = createWorkerEmbedder({ workerUrl: URL_STUB, WorkerImpl, embedTimeoutMs: 5 });
  const warmup = embedder.warmup();
  await assert.rejects(embedder.embed(["a"]), /stopped answering/);
  assert.equal(WorkerImpl.terminated, 1);
  await assert.rejects(warmup, /stopped answering/, "the worker is gone, so its other requests end too");
});

// The worker embeds one text at a time, so a long call is a long batch from a healthy worker: a flat
// limit killed it on a slow phone, for good.
test("the patience grows with the batch", async () => {
  const WorkerImpl = fakeWorker({ answer: false });
  const embedder = createWorkerEmbedder({ workerUrl: URL_STUB, WorkerImpl, embedTimeoutMs: 20 });
  const started = Date.now();
  await assert.rejects(embedder.embed(Array.from({ length: 5 }, (_, i) => `text ${i}`)), /stopped answering/);
  assert.ok(Date.now() - started >= 90, "20 ms a text, 5 texts");
});

test("the runtime can tell its worker is gone, so the page can build a new one", async () => {
  const WorkerImpl = fakeWorker();
  const embedder = createWorkerEmbedder({ workerUrl: URL_STUB, WorkerImpl });
  assert.equal(embedder.alive(), true);
  WorkerImpl.instance.onerror({ message: "out of memory" });
  assert.equal(embedder.alive(), false);
});
