/* The live pass of the voice mode (ADR D64): word timings from Vosk, the span vote rerun on
   every recogniser event (per-word kk/ru/en tags while the person speaks), and the audio span
   each committed utterance carries for the endpoint pass. */
import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { initialAsrState, parseVoskWords, reduceAsrEvent } from "../site/core/asr.js";
import { createWordLid } from "../site/core/word-lid.js";
import { createSpanVoter } from "../site/core/span-vote.js";

const wordlevel = JSON.parse(readFileSync(new URL("../site/core/asr-wordlevel.json", import.meta.url), "utf8"));
const voter = createSpanVoter(wordlevel.span_vote, createWordLid(wordlevel.lid));
const timed = (kind, words) => {
  const list = words.map(([word, start, end, conf]) => ({ word, start, end, conf }));
  const text = words.map((w) => w[0]).join(" ");
  return kind === "result" ? { text, result: list } : { partial: text, partial_result: list };
};
const untimed = (text, conf) => JSON.stringify({ text, result: text.split(" ").map((word) => ({ word, conf })) });

test("parseVoskWords: timed words from results and partials; untimed words are dropped", () => {
  assert.deepEqual(parseVoskWords(JSON.stringify(timed("result", [["код", 0.5, 0.9, 0.8]]))), [["код", 0.5, 0.9, 0.8]]);
  assert.deepEqual(parseVoskWords(timed("partial", [["алло", 0, 0.4, 1]])), [["алло", 0, 0.4, 1]]);
  assert.deepEqual(parseVoskWords(JSON.stringify({ partial: "алло", partial_result: [{ word: "алло", start: 0, end: 0.4 }] })), [["алло", 0, 0.4, 1]]);
  assert.deepEqual(parseVoskWords(untimed("назовите код", 0.9)), []);
  assert.deepEqual(parseVoskWords("not json"), []);
});

test("every partial reruns the span vote and emits per-word language tags", () => {
  const opts = { voter };
  let state = initialAsrState(["kk", "ru", "en"]);
  state = reduceAsrEvent(state, { type: "partial", language: "kk", detail: timed("partial", [["сәлеметсіз", 0, 0.7, 1], ["бе", 0.7, 0.9, 1], ["су", 1.0, 1.2, 0.6]]) }, 100, opts).state;
  state = reduceAsrEvent(state, { type: "partial", language: "ru", detail: timed("partial", [["со", 0, 0.2, 0.6], ["лимит", 0.2, 0.5, 0.6], ["службы", 1.0, 1.5, 1]]) }, 120, opts).state;
  const out = reduceAsrEvent(state, { type: "partial", language: "en", detail: timed("partial", [["sally", 0, 0.3, 0.3], ["meets", 0.3, 0.6, 0.5]]) }, 140, opts);
  const live = out.emits.find((e) => e.type === "live");
  assert.ok(live, "a live emit");
  assert.ok(live.words.length >= 3);
  assert.deepEqual([live.words[0].word, live.words[0].lang], ["сәлеметсіз", "kk"]);
  for (const w of live.words) {
    assert.ok(["kk", "ru", "en"].includes(w.lang));
    assert.equal(typeof w.frozen, "boolean");
  }
});

test("results accumulate into the live hypotheses until the vote commits", () => {
  const opts = { voter };
  let out = reduceAsrEvent(initialAsrState(["kk", "ru"]), { type: "partial", language: "ru", detail: timed("partial", [["назовите", 1.2, 1.8, 0.9]]) }, 100, opts);
  out = reduceAsrEvent(out.state, { type: "result", language: "ru", detail: timed("result", [["назовите", 1.2, 1.8, 0.9], ["код", 1.8, 2.3, 0.9]]) }, 200, opts);
  assert.deepEqual(out.state.live.finals.ru.map((w) => w[0]), ["назовите", "код"]);
  assert.deepEqual(out.state.live.partials.ru, []);
  assert.ok(out.emits.some((e) => e.type === "live"));
});

test("the committed utterance carries its audio span; the live hypotheses and the freeze reset", () => {
  const opts = { voter };
  let out = reduceAsrEvent(initialAsrState(["kk", "ru"]), { type: "result", language: "ru", detail: timed("result", [["назовите", 1.2, 1.8, 0.9], ["код", 1.8, 2.3, 0.9]]) }, 200, opts);
  out = reduceAsrEvent(out.state, { type: "result", language: "kk", detail: timed("result", [["назови", 1.1, 1.7, 0.4], ["те", 1.7, 1.9, 0.4]]) }, 300, opts);
  const utterance = out.emits.find((e) => e.type === "utterance");
  assert.equal(utterance.text, "назовите код");
  assert.equal(utterance.start, 1.1);
  assert.equal(utterance.end, 2.3);
  assert.deepEqual(out.state.live, { finals: {}, partials: {}, freeze: { spans: [], until: 0 }, after: 2.3 });
});

test("without a voter the reducer behaves exactly as before: no live emits, no timing on untimed words", () => {
  const out = reduceAsrEvent(initialAsrState(["kk", "ru"]), { type: "partial", language: "ru", detail: timed("partial", [["алло", 0, 0.4, 1]]) }, 100);
  assert.deepEqual(out.emits, [{ type: "partial", language: "ru", text: "алло" }]);
  let s = reduceAsrEvent(initialAsrState(["kk", "ru"]), { type: "result", language: "ru", detail: untimed("код", 0.9) }, 1).state;
  const done = reduceAsrEvent(s, { type: "result", language: "kk", detail: untimed("кот", 0.3) }, 2);
  assert.deepEqual(done.emits, [{ type: "utterance", language: "ru", text: "код", confidence: 0.9 }]);
});

// Found in the Chromium run: a recogniser that has not endpointed keeps ONE partial across the
// committed utterance and the next, so the next utterance's span reached back over the previous
// one and GigaAM transcribed both. Words of an already committed stretch are dropped.
test("a recogniser that never endpointed cannot drag the previous utterance into the next span", () => {
  const opts = { voter };
  let state = initialAsrState(["kk", "ru", "en"]);
  const step = (event, now) => { const out = reduceAsrEvent(state, event, now, opts); state = out.state; return out.emits; };
  step({ type: "partial", language: "en", detail: timed("partial", [["sally", 0.1, 0.6, 0.4], ["meets", 0.6, 1.4, 0.4]]) }, 100);
  step({ type: "result", language: "kk", detail: timed("result", [["сәлем", 0.1, 0.7, 0.9], ["бе", 0.7, 1.4, 0.9]]) }, 200);
  step({ type: "result", language: "ru", detail: timed("result", [["салем", 0.1, 0.7, 0.5], ["бе", 0.7, 1.4, 0.5]]) }, 250);
  const first = step({ type: "tick" }, 1500).find((e) => e.type === "utterance");
  assert.deepEqual([first.start, first.end], [0.1, 1.4]);
  // en still has not endpointed: its next partial spans both utterances
  step({ type: "partial", language: "en", detail: timed("partial", [["sally", 0.1, 0.6, 0.4], ["meets", 0.6, 1.4, 0.4], ["code", 3.0, 3.5, 0.5]]) }, 3000);
  step({ type: "result", language: "kk", detail: timed("result", [["код", 3.0, 3.5, 0.9]]) }, 3100);
  step({ type: "result", language: "ru", detail: timed("result", [["код", 3.0, 3.5, 0.9]]) }, 3150);
  const second = step({ type: "tick" }, 4500).find((e) => e.type === "utterance");
  assert.deepEqual([second.start, second.end], [3.0, 3.5]);
});

test("a suppressed duplicate result never leaks into the next utterance's live hypotheses", () => {
  const opts = { voter };
  let state = initialAsrState(["kk", "ru"]);
  state = reduceAsrEvent(state, { type: "partial", language: "kk", detail: timed("partial", [["сәлем", 0, 0.5, 0.95]]) }, 900, opts).state;
  let out = reduceAsrEvent(state, { type: "result", language: "ru", detail: timed("result", [["салем", 0, 0.5, 0.4]]) }, 1000, opts);
  out = reduceAsrEvent(out.state, { type: "tick" }, 2300, opts);
  assert.equal(out.emits[0].text, "сәлем");
  out = reduceAsrEvent(out.state, { type: "result", language: "kk", detail: timed("result", [["сәлем", 0, 0.5, 0.95]]) }, 2400, opts);
  assert.deepEqual(out.state.live.finals, {});
});
