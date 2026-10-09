/* The WebView-safe voice mode's wiring (ADR D67) with a fake GigaAM client: the VAD cuts the
   audio into phrases, each phrase goes through the D64 endpoint pass, and only GigaAM's text
   reaches the page. Nothing here touches a real AudioContext. */
import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { createWordLid } from "../site/core/word-lid.js";
import { AUDIO_PAD_S } from "../site/core/endpoint-runtime.js";
import { LIGHT_CONFIDENCE, createLightPass } from "../site/core/voice-light.js";

const lid = createWordLid(JSON.parse(readFileSync(new URL("../site/core/asr-wordlevel.json", import.meta.url), "utf8")).lid);
const endpoint = { id: "g", model: { url: "m", tokens: "t" }, runtime: "r/", cache: "c" };
const RATE = 16000;

function phrases(spec) {
  let s = 1;
  const rnd = () => { s = (Math.imul(s, 1664525) + 1013904223) >>> 0; return s / 2147483648 - 1; };
  const n = spec.reduce((a, [, sec]) => a + Math.round(sec * RATE), 0);
  const out = new Float32Array(n);
  let at = 0;
  for (const [kind, sec] of spec) {
    const len = Math.round(sec * RATE);
    if (kind === "speech") for (let i = 0; i < len; i++) out[at + i] = 0.3 * (0.6 * Math.sin((2 * Math.PI * 220 * (at + i)) / RATE) + 0.4 * rnd());
    at += len;
  }
  return out;
}

function fakeClient({ fail = false, answers = ["алло здравствуйте", "назовите код из смс"] } = {}) {
  const calls = [];
  let resolveReady;
  let rejectReady;
  const ready = new Promise((res, rej) => { resolveReady = res; rejectReady = rej; });
  return {
    calls,
    open: () => (fail ? rejectReady(new Error("model 404")) : resolveReady({ loadMs: 10, fromCache: true })),
    factory: () => ({
      ready,
      decode: async (samples, rate) => {
        calls.push({ seconds: samples.length / rate, rate });
        const answer = answers[(calls.length - 1) % answers.length];
        if (answer instanceof Error) throw Object.assign(answer, { reason: "timeout" });
        return { text: answer, decodeMs: 5 };
      },
      dispose() {},
    }),
  };
}

function feed(pass, samples, chunk = 2048) {
  for (let i = 0; i < samples.length; i += chunk) pass.push(samples.subarray(i, i + chunk));
}

test("each phrase the VAD finds is decoded by GigaAM and reaches the page as GigaAM's text", async () => {
  const fake = fakeClient();
  const finals = [];
  const speech = [];
  const pass = createLightPass({ endpoint, tagText: lid.tagText, pageUrl: "http://x/live.html", onUtterance: (u, info) => finals.push({ u, info }), onSpeech: (s) => speech.push(s.type), createClient: fake.factory });
  fake.open();
  await pass.ready;
  pass.reset(RATE);
  feed(pass, phrases([["silence", 1], ["speech", 1.5], ["silence", 1], ["speech", 2], ["silence", 1]]));
  await pass.stop();
  assert.equal(fake.calls.length, 2);
  // The endpoint pass pads each phrase by AUDIO_PAD_S on both sides (the VAD adds its pre-roll).
  assert.ok(Math.abs(fake.calls[0].seconds - (1.5 + 0.15 + 2 * AUDIO_PAD_S)) < 0.08, `first decode ${fake.calls[0].seconds}`);
  assert.deepEqual(finals.map((f) => f.u.text), ["алло здравствуйте", "назовите код из смс"]);
  assert.ok(finals.every((f) => f.u.source === "gigaam" && f.u.confidence === LIGHT_CONFIDENCE && f.u.language === "ru"));
  assert.ok(finals[0].u.words.length > 0, "words are tagged by the word LID");
  assert.deepEqual(speech, ["speech_start", "segment", "speech_start", "segment"]);
  assert.equal(pass.segments.length, 2);
  assert.ok(pass.segments.every((s) => s.end > s.start));
});

test("a failed or empty decode is dropped (reported), never sent to the meter as empty text", async () => {
  const fake = fakeClient({ answers: [new Error("slow"), "", "переведите деньги"] });
  const finals = [];
  const dropped = [];
  const pass = createLightPass({ endpoint, tagText: lid.tagText, onUtterance: (u) => finals.push(u), onDropped: (d) => dropped.push(d.reason), createClient: fake.factory });
  fake.open();
  await pass.ready;
  pass.reset(RATE);
  feed(pass, phrases([["silence", 0.5], ["speech", 1], ["silence", 1], ["speech", 1], ["silence", 1], ["speech", 1], ["silence", 1]]));
  await pass.stop();
  assert.deepEqual(dropped, ["timeout", "empty"]);
  assert.deepEqual(finals.map((u) => u.text), ["переведите деньги"]);
});

test("a phrase still open when the microphone stops is flushed and decoded", async () => {
  const fake = fakeClient({ answers: ["код из смс"] });
  const finals = [];
  const pass = createLightPass({ endpoint, tagText: lid.tagText, onUtterance: (u) => finals.push(u), createClient: fake.factory });
  fake.open();
  await pass.ready;
  pass.reset(RATE);
  feed(pass, phrases([["silence", 0.5], ["speech", 1.2]]));
  assert.equal(fake.calls.length, 0);
  await pass.stop();
  assert.equal(fake.calls.length, 1);
  assert.deepEqual(finals.map((u) => u.text), ["код из смс"]);
});

test("if GigaAM cannot load, `ready` rejects: the light mode has nothing to fall back to", async () => {
  const fake = fakeClient({ fail: true });
  const statuses = [];
  const pass = createLightPass({ endpoint, tagText: lid.tagText, onUtterance() {}, onStatus: (s) => statuses.push(s.status), createClient: fake.factory });
  fake.open();
  await assert.rejects(pass.ready, /model 404/);
  assert.deepEqual(statuses, ["loading", "unavailable"]);
});

test("without the endpoint model or the word LID the light mode is unavailable at once", async () => {
  await assert.rejects(createLightPass({ endpoint: null, tagText: lid.tagText, onUtterance() {} }).ready, /word-level model/);
  await assert.rejects(createLightPass({ endpoint, tagText: null, onUtterance() {} }).ready, /word-level model/);
});
