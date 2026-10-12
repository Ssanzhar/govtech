/* The endpoint pass's browser wiring with a fake GigaAM client (ADR D64): the audio span of
   each utterance is cut from the ring and decoded; failure to load means the Vosk vote, marked. */
import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { createWordLid } from "../site/core/word-lid.js";
import { AUDIO_PAD_S, createEndpointPass } from "../site/core/endpoint-runtime.js";

const lid = createWordLid(JSON.parse(readFileSync(new URL("../site/core/asr-wordlevel.json", import.meta.url), "utf8")).lid);
const endpoint = { id: "g", model: { url: "m", tokens: "t" }, runtime: "r/", cache: "c" };
const tick = () => new Promise((resolve) => setTimeout(resolve, 0));

function fakeClient({ fail = false, answer = () => "назовите код из смс" } = {}) {
  const calls = [];
  let resolveReady;
  let rejectReady;
  const ready = new Promise((res, rej) => { resolveReady = res; rejectReady = rej; });
  return {
    calls,
    open: () => (fail ? rejectReady(new Error("model 404")) : resolveReady({ loadMs: 10, fromCache: true })),
    factory: () => ({
      ready,
      decode: async (samples, rate) => { calls.push({ n: samples.length, rate }); return { text: answer(calls.length), decodeMs: 5 }; },
      dispose() {},
    }),
  };
}

test("an utterance's padded audio span goes to GigaAM and its text replaces the Vosk text", async () => {
  const fake = fakeClient();
  const finals = [];
  const statuses = [];
  const pass = createEndpointPass({ endpoint, tagText: lid.tagText, pageUrl: "http://x/live.html", onUtterance: (u, info) => finals.push({ u, info }), onStatus: (s) => statuses.push(s.status), createClient: fake.factory });
  pass.reset(100); // 100 Hz keeps the arithmetic visible
  pass.push(new Float32Array(500)); // 5 s
  fake.open();
  await tick();
  assert.equal(pass.status, "ready");
  pass.utterance({ type: "utterance", language: "ru", text: "назови те кот", confidence: 0.6, start: 1, end: 3 });
  await pass.drain();
  assert.deepEqual(fake.calls, [{ n: Math.round((3 - 1 + 2 * AUDIO_PAD_S) * 100), rate: 100 }]);
  assert.equal(finals.length, 1);
  assert.equal(finals[0].u.text, "назовите код из смс");
  assert.equal(finals[0].u.source, "gigaam");
  assert.ok(finals[0].info.latencyMs >= 0);
  assert.deepEqual(statuses, ["loading", "ready"]);
});

test("GigaAM failing to load: every utterance (pending or new) falls back to the Vosk vote, marked", async () => {
  const fake = fakeClient({ fail: true });
  const finals = [];
  const statuses = [];
  const pass = createEndpointPass({ endpoint, tagText: lid.tagText, onUtterance: (u) => finals.push(u), onStatus: (s) => statuses.push(s), createClient: fake.factory });
  pass.reset(100);
  pass.utterance({ type: "utterance", language: "kk", text: "кодты айтыңыз", confidence: 0.9, start: 0, end: 1 }); // while loading
  fake.open();
  await tick();
  pass.utterance({ type: "utterance", language: "ru", text: "назовите код", confidence: 0.8, start: 1, end: 2 });
  assert.deepEqual(finals.map((u) => [u.text, u.source, u.fallback]), [["кодты айтыңыз", "vosk", "loading"], ["назовите код", "vosk", "unavailable"]]);
  assert.equal(statuses.at(-1).status, "unavailable");
  assert.match(statuses.at(-1).reason, /404/);
});

test("no endpoint configured: the Vosk vote passes straight through", () => {
  const finals = [];
  const pass = createEndpointPass({ endpoint: null, tagText: lid.tagText, onUtterance: (u) => finals.push(u), createClient: () => assert.fail("no client") });
  pass.reset(16000);
  pass.utterance({ type: "utterance", language: "ru", text: "алло", confidence: 1 });
  assert.equal(finals[0].text, "алло");
  assert.equal(finals[0].source, "vosk");
});
