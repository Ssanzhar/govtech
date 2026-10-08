/* The endpoint pass's pure reducer (ADR D63): GigaAM's text replaces the Vosk utterance, in
   commit order; anything that keeps GigaAM from answering falls back to the Vosk vote, marked. */
import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { REPO } from "./helpers.mjs";
import { createWordLid } from "../site/core/word-lid.js";
import { initialEndpointState, majorityLanguage, reduceEndpoint } from "../site/core/endpoint.js";

const lid = createWordLid(JSON.parse(readFileSync(join(REPO, "site", "core", "asr-wordlevel.json"), "utf8")).lid);
const opts = { tagText: lid.tagText };
const vosk = (text, language = "ru", confidence = 0.8, start = 1, end = 3) => ({ type: "utterance", language, text, confidence, start, end });

const run = (state, events) => {
  const emits = [];
  const requests = [];
  for (const event of events) {
    const out = reduceEndpoint(state, event, opts);
    state = out.state;
    emits.push(...out.emits);
    requests.push(...out.requests);
  }
  return { state, emits, requests };
};

test("ready: the utterance waits for GigaAM, whose text replaces the Vosk text with per-word tags", () => {
  let out = reduceEndpoint(initialEndpointState("ready"), { type: "utterance", utterance: vosk("назови те кот из смс", "ru", 0.7, 2.0, 4.5) }, opts);
  assert.deepEqual(out.emits, []);
  assert.deepEqual(out.requests, [{ type: "decode", seq: 0, start: 2.0, end: 4.5 }]);
  out = reduceEndpoint(out.state, { type: "decoded", seq: 0, text: "назовите код из смс", decodeMs: 520 }, opts);
  assert.equal(out.emits.length, 1);
  const [u] = out.emits;
  assert.equal(u.text, "назовите код из смс");
  assert.equal(u.source, "gigaam");
  assert.equal(u.language, "ru");
  assert.equal(u.confidence, 0.7); // the meter keeps the vote's confidence
  assert.deepEqual(u.words.map((w) => w.lang), ["ru", "ru", "ru", "ru"]);
  assert.deepEqual(u.vosk, { language: "ru", text: "назови те кот из смс", confidence: 0.7 });
  assert.equal(u.decodeMs, 520);
  assert.deepEqual(out.state.queue, []);
});

test("code-switched GigaAM text is tagged word by word; the utterance language is the majority", () => {
  const { emits } = run(initialEndpointState("ready"), [
    { type: "utterance", utterance: vosk("сәлеметсіз бе мен банктің", "kk") },
    { type: "decoded", seq: 0, text: "сәлеметсіз бе мен банктің службы безопасности қызметкерімін" },
  ]);
  assert.equal(emits[0].language, "kk");
  assert.ok(emits[0].words.some((w) => w.lang === "ru"), "the Russian insertion keeps its own tag");
  const en = run(initialEndpointState("ready"), [{ type: "utterance", utterance: vosk("зе код", "ru") }, { type: "decoded", seq: 0, text: "read me the code now" }]);
  assert.equal(en.emits[0].language, "en");
});

test("answers out of order are emitted in commit order", () => {
  const { emits, requests } = run(initialEndpointState("ready"), [
    { type: "utterance", utterance: vosk("бір", "kk") },
    { type: "utterance", utterance: vosk("два", "ru", 0.8, 4, 5) },
    { type: "decoded", seq: 1, text: "два три" },
  ]);
  assert.equal(requests.length, 2);
  assert.deepEqual(emits, []); // seq 1 is done but seq 0 is not
  const out = reduceEndpoint(run(initialEndpointState("ready"), [
    { type: "utterance", utterance: vosk("бір", "kk") },
    { type: "utterance", utterance: vosk("два", "ru", 0.8, 4, 5) },
    { type: "decoded", seq: 1, text: "два три" },
  ]).state, { type: "decoded", seq: 0, text: "бір екі" }, opts);
  assert.deepEqual(out.emits.map((e) => [e.seq, e.text]), [[0, "бір екі"], [1, "два три"]]);
});

test("fallback: while GigaAM loads, the Vosk vote is used immediately and marked", () => {
  const { emits, requests } = run(initialEndpointState("loading"), [{ type: "utterance", utterance: vosk("назовите код") }]);
  assert.deepEqual(requests, []);
  assert.equal(emits[0].source, "vosk");
  assert.equal(emits[0].fallback, "loading");
  assert.equal(emits[0].text, "назовите код");
  assert.equal(emits[0].confidence, 0.8);
  assert.deepEqual(emits[0].words.map((w) => w.word), ["назовите", "код"]);
});

test("fallback: GigaAM becoming unavailable releases everything pending, in order", () => {
  const { state } = run(initialEndpointState("ready"), [{ type: "utterance", utterance: vosk("один") }, { type: "utterance", utterance: vosk("два") }]);
  const out = reduceEndpoint(state, { type: "status", status: "unavailable", reason: "load failed" }, opts);
  assert.deepEqual(out.emits.map((e) => [e.seq, e.text, e.source, e.fallback]), [[0, "один", "vosk", "unavailable"], [1, "два", "vosk", "unavailable"]]);
  const after = reduceEndpoint(out.state, { type: "utterance", utterance: vosk("три") }, opts);
  assert.equal(after.emits[0].fallback, "unavailable");
  assert.deepEqual(after.requests, []);
});

test("fallback: a failed or timed-out decode, an empty transcript, or an utterance without timing", () => {
  const failed = run(initialEndpointState("ready"), [{ type: "utterance", utterance: vosk("код") }, { type: "failed", seq: 0, reason: "timeout" }]);
  assert.equal(failed.emits[0].fallback, "timeout");
  const empty = run(initialEndpointState("ready"), [{ type: "utterance", utterance: vosk("код") }, { type: "decoded", seq: 0, text: "  " }]);
  assert.equal(empty.emits[0].fallback, "empty");
  assert.equal(empty.emits[0].text, "код");
  const untimed = run(initialEndpointState("ready"), [{ type: "utterance", utterance: { language: "ru", text: "код", confidence: 0.9 } }]);
  assert.equal(untimed.emits[0].fallback, "no_timing");
  assert.deepEqual(untimed.requests, []);
});

test("a late or duplicate answer is ignored; inputs are never mutated", () => {
  const first = run(initialEndpointState("ready"), [{ type: "utterance", utterance: vosk("код") }, { type: "decoded", seq: 0, text: "код" }]);
  const frozen = JSON.stringify(first.state);
  const again = reduceEndpoint(first.state, { type: "decoded", seq: 0, text: "другое" }, opts);
  assert.deepEqual(again.emits, []);
  assert.equal(JSON.stringify(first.state), frozen);
  assert.throws(() => reduceEndpoint(first.state, { type: "nope" }, opts), /unknown endpoint event/);
});

test("majorityLanguage: ties keep the vote's language", () => {
  assert.equal(majorityLanguage([{ lang: "kk" }, { lang: "ru" }], "ru"), "ru");
  assert.equal(majorityLanguage([{ lang: "kk" }, { lang: "ru" }], "en"), "kk");
  assert.equal(majorityLanguage([], "ru"), "ru");
});
