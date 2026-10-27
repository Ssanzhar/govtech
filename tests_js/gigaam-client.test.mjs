/* The GigaAM worker client (ADR D64, D78): the worker decodes one phrase at a time, so the client
   posts one at a time and a phrase's timeout counts only its own decode, never the time it waited
   behind earlier phrases (a slow phone queues phrases; light mode drops a timed-out one). */
import test from "node:test";
import assert from "node:assert/strict";
import { createGigaamClient, decodeTimeoutMs } from "../site/core/gigaam.js";

const endpoint = { id: "g", model: { url: "m", tokens: "t" }, runtime: "r/", cache: "c" };

/** A worker that decodes serially, `decodeMs` per phrase, like the single-threaded sherpa build. */
function slowWorker(decodeMs) {
  const posted = [];
  class FakeWorker {
    constructor() { this.busy = Promise.resolve(); FakeWorker.instance = this; }
    postMessage(msg) {
      posted.push({ type: msg.type, at: Date.now() });
      if (msg.type === "init") { setTimeout(() => this.onmessage({ data: { type: "ready", loadMs: 1, fromCache: true } }), 0); return; }
      this.busy = this.busy.then(() => new Promise((resolve) => setTimeout(() => {
        this.onmessage({ data: { type: "result", id: msg.id, text: `phrase ${msg.id}`, decodeMs, audioS: msg.samples.length / msg.sampleRate } });
        resolve();
      }, decodeMs)));
    }
    terminate() {}
  }
  return { FakeWorker, posted };
}

test("the timeout grows with the phrase's audio", () => {
  assert.equal(decodeTimeoutMs(0, { baseMs: 100, perAudioS: 50 }), 100);
  assert.equal(decodeTimeoutMs(2, { baseMs: 100, perAudioS: 50 }), 200);
});

test("phrases queued behind a slow decode do not time out while they wait", async () => {
  // Each decode takes 60 ms; a phrase may take 100 ms. Posted all at once, the 2nd and 3rd would
  // reach 120 / 180 ms of wall time before their answer: a queue-blind timer drops them.
  const { FakeWorker, posted } = slowWorker(60);
  const client = createGigaamClient({ endpoint, pageUrl: "http://x/live.html", WorkerImpl: FakeWorker, timeout: { baseMs: 100, perAudioS: 0 } });
  await client.ready;
  const results = await Promise.allSettled([1, 2, 3].map(() => client.decode(new Float32Array(16), 16)));
  assert.deepEqual(results.map((r) => r.status), ["fulfilled", "fulfilled", "fulfilled"]);
  assert.deepEqual(results.map((r) => r.value.text), ["phrase 0", "phrase 1", "phrase 2"]);
  const decodes = posted.filter((p) => p.type === "decode");
  assert.equal(decodes.length, 3);
  assert.ok(decodes[2].at - decodes[0].at >= 100, "the worker gets the next phrase only after the previous answer");
  client.dispose();
});

test("a decode that really hangs still times out, and the next phrase then goes to the worker", async () => {
  const posted = [];
  class HangOnce {
    constructor() { HangOnce.instance = this; }
    postMessage(msg) {
      posted.push(msg.type);
      if (msg.type === "init") { setTimeout(() => this.onmessage({ data: { type: "ready", loadMs: 1, fromCache: true } }), 0); return; }
      if (msg.id === 0) return; // never answers
      setTimeout(() => this.onmessage({ data: { type: "result", id: msg.id, text: "ok", decodeMs: 1, audioS: 1 } }), 5);
    }
    terminate() {}
  }
  const client = createGigaamClient({ endpoint, pageUrl: "http://x/live.html", WorkerImpl: HangOnce, timeout: { baseMs: 30, perAudioS: 0 } });
  await client.ready;
  const [first, second] = await Promise.allSettled([client.decode(new Float32Array(16), 16), client.decode(new Float32Array(16), 16)]);
  assert.equal(first.status, "rejected");
  assert.equal(first.reason.reason, "timeout");
  assert.equal(second.status, "fulfilled");
  assert.equal(second.value.text, "ok");
  client.dispose();
});

test("backlog counts the phrases waiting for or in the worker", async () => {
  const { FakeWorker } = slowWorker(20);
  const client = createGigaamClient({ endpoint, pageUrl: "http://x/live.html", WorkerImpl: FakeWorker, timeout: { baseMs: 1000, perAudioS: 0 } });
  await client.ready;
  assert.equal(client.backlog, 0);
  const jobs = [1, 2, 3].map(() => client.decode(new Float32Array(16), 16));
  assert.equal(client.backlog, 3);
  await Promise.all(jobs);
  assert.equal(client.backlog, 0);
  client.dispose();
});

test("dispose rejects the phrases still waiting", async () => {
  const { FakeWorker } = slowWorker(50);
  const client = createGigaamClient({ endpoint, pageUrl: "http://x/live.html", WorkerImpl: FakeWorker, timeout: { baseMs: 1000, perAudioS: 0 } });
  await client.ready;
  const jobs = [1, 2].map(() => client.decode(new Float32Array(16), 16));
  client.dispose();
  const results = await Promise.allSettled(jobs);
  assert.deepEqual(results.map((r) => r.status), ["rejected", "rejected"]);
});
