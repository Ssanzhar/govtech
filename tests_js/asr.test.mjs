/* Tests for site/core/asr.js — the pure part of the on-device ASR port (PLAN B9): the
   result parser and voting rule (1:1 with tests/asr/test_vosk_stream.py) and the
   endpoint-alignment state machine over Vosklet events. */
import test from "node:test";
import assert from "node:assert/strict";
import { SUPPORT_REASONS, initialAsrState, isSupported, parseVoskResult, reduceAsrEvent, supportIssues, voteFinal } from "../site/core/asr.js";

const result = (text, ...confs) =>
  JSON.stringify({ text, result: text.split(" ").map((word, i) => (confs[i] === undefined ? { word } : { word, conf: confs[i] })) });
const partial = (text, ...confs) =>
  JSON.stringify({ partial: text, partial_result: text.split(" ").map((word, i) => (confs[i] === undefined ? { word } : { word, conf: confs[i] })) });

test("parseVoskResult: mean word confidence, missing confs default to full, clamped", () => {
  assert.deepEqual(parseVoskResult(result("назовите код", 0.9, 0.7)), { text: "назовите код", confidence: 0.8 });
  assert.deepEqual(parseVoskResult(result("код")), { text: "код", confidence: 1 });
  assert.equal(parseVoskResult(result("а б", 1.4, 0.8)).confidence, 1);
  assert.deepEqual(parseVoskResult("{}"), { text: "", confidence: 0 });
  assert.deepEqual(parseVoskResult("not json"), { text: "", confidence: 0 });
  assert.deepEqual(parseVoskResult({ text: " " }), { text: "", confidence: 0 });
  assert.deepEqual(parseVoskResult(partial("кодты айт", 0.5, 0.5)), { text: "кодты айт", confidence: 0.5 });
});

test("voteFinal: the more confident language wins; empty text always loses", () => {
  const kk = { text: "кодты айтыңыз", confidence: 0.9 };
  const ru = { text: "назовите код", confidence: 0.6 };
  assert.deepEqual(voteFinal({ kk, ru }, "ru"), { language: "kk", text: "кодты айтыңыз", confidence: 0.9 });
  assert.deepEqual(voteFinal({ kk: { text: "", confidence: 1 }, ru }, "kk").language, "ru");
  assert.equal(voteFinal({ kk: { text: "", confidence: 0 }, ru: { text: "", confidence: 0 } }, "kk"), null);
});

test("voteFinal: on a near-tie the script decides for kk/ru — few Kazakh letters in the KK hypothesis means Russian speech", () => {
  // Russian speech: the KK model produced Russian loanwords with a sprinkle of Kazakh letters (measured in the B9 harness)
  const ruSpeech = {
    kk: { text: "алло здравствуйте этой службы безопасности по вашей кортизол тексеріп операцияны", confidence: 0.95 },
    ru: { text: "алло здравствуйте это служба безопасности вашего банка по вашей карте", confidence: 0.95 },
  };
  assert.equal(voteFinal(ruSpeech, "kk").language, "ru");
  // Kazakh speech: the KK hypothesis is full of Kazakh letters
  const kkSpeech = {
    kk: { text: "сәлеметсіз бе бұл банктің қауіпсіздік қызметі ешкімге айтпаңыз", confidence: 0.93 },
    ru: { text: "соли миссис би был бантом хаус", confidence: 0.94 },
  };
  assert.equal(voteFinal(kkSpeech, "ru").language, "kk");
  // beyond the epsilon it is not a tie
  assert.equal(voteFinal({ ...kkSpeech, ru: { ...kkSpeech.ru, confidence: 0.99 } }, "ru").language, "ru");
  // other language pairs: the tie goes to the preferred language
  const other = { en: { text: "hello", confidence: 0.9 }, de: { text: "hallo", confidence: 0.9 } };
  assert.equal(voteFinal(other, "de").language, "de");
  assert.equal(voteFinal(other, "en").language, "en");
});

test("kazakhLetterShare", async () => {
  const { kazakhLetterShare } = await import("../site/core/asr.js");
  assert.equal(kazakhLetterShare("здравствуйте это банк"), 0);
  assert.ok(kazakhLetterShare("сәлеметсіз бе бұл банктің қауіпсіздік қызметі") > 0.12);
  assert.equal(kazakhLetterShare("123 ..."), 0);
});

test("aligned endpoints: both results inside the window → one voted utterance, preference follows the winner", () => {
  let state = initialAsrState(["kk", "ru"]);
  let out = reduceAsrEvent(state, { type: "result", language: "ru", detail: result("назовите код из смс", 0.9, 0.9, 0.9, 0.9) }, 1000);
  assert.deepEqual(out.emits, []); // waiting for kk
  out = reduceAsrEvent(out.state, { type: "result", language: "kk", detail: result("назовите кот из мс", 0.5, 0.5, 0.5, 0.5) }, 1100);
  assert.deepEqual(out.emits, [{ type: "utterance", language: "ru", text: "назовите код из смс", confidence: 0.9 }]);
  assert.equal(out.state.preferred, "ru");
  assert.deepEqual(out.state.pending, {});
  assert.equal(out.state.windowOpenedAt, null);
  assert.deepEqual(state.pending, {}); // inputs untouched
});

test("unaligned endpoints: the window closes on a tick, the silent language contributes its partial and its late result is dropped", () => {
  let state = initialAsrState(["kk", "ru"]);
  state = reduceAsrEvent(state, { type: "partial", language: "kk", detail: partial("сәлеметсіз бе", 0.95, 0.95) }, 900).state;
  let out = reduceAsrEvent(state, { type: "result", language: "ru", detail: result("салем этси бе", 0.4, 0.4, 0.4) }, 1000);
  assert.deepEqual(out.emits, []);
  out = reduceAsrEvent(out.state, { type: "tick" }, 2000);
  assert.deepEqual(out.emits, []); // window still open (1200 ms)
  out = reduceAsrEvent(out.state, { type: "tick" }, 2250);
  assert.deepEqual(out.emits, [{ type: "utterance", language: "kk", text: "сәлеметсіз бе", confidence: 0.95 }]);
  // kk's own endpoint arrives late: a duplicate, suppressed -- and so is a second one inside the window
  out = reduceAsrEvent(out.state, { type: "result", language: "kk", detail: result("сәлеметсіз бе", 0.95, 0.95) }, 2600);
  assert.deepEqual(out.emits, []);
  out = reduceAsrEvent(out.state, { type: "result", language: "kk", detail: result("сәлеметсіз бе", 0.95, 0.95) }, 3000);
  assert.deepEqual(out.emits, []);
  assert.ok(out.state.suppressUntil.kk > 3000);
  // ...the entry expires on a tick, and a new kk result is a new utterance
  out = reduceAsrEvent(out.state, { type: "tick" }, 3900);
  assert.deepEqual(out.state.suppressUntil, {});
  out = reduceAsrEvent(out.state, { type: "result", language: "kk", detail: result("кодты айтыңыз", 0.9, 0.9) }, 4000);
  out = reduceAsrEvent(out.state, { type: "tick" }, 5300);
  assert.equal(out.emits[0]?.text, "кодты айтыңыз");
});

test("partials stream between endpoints, deduplicated, preferred language first with fallback", () => {
  let state = initialAsrState(["kk", "ru"]);
  let out = reduceAsrEvent(state, { type: "partial", language: "ru", detail: partial("алло") }, 100);
  assert.deepEqual(out.emits, [{ type: "partial", language: "ru", text: "алло" }]); // kk silent → ru shown
  out = reduceAsrEvent(out.state, { type: "partial", language: "ru", detail: partial("алло") }, 200);
  assert.deepEqual(out.emits, []); // identical consecutive partial
  out = reduceAsrEvent(out.state, { type: "partial", language: "kk", detail: partial("алло сәлем") }, 300);
  assert.deepEqual(out.emits, [{ type: "partial", language: "kk", text: "алло сәлем" }]); // preferred (kk) now speaks
});

test("silence endpoints emit nothing and unknown events throw", () => {
  let out = reduceAsrEvent(initialAsrState(["kk", "ru"]), { type: "result", language: "kk", detail: "{}" }, 10);
  out = reduceAsrEvent(out.state, { type: "result", language: "ru", detail: "{}" }, 20);
  assert.deepEqual(out.emits, []);
  assert.throws(() => reduceAsrEvent(out.state, { type: "bogus" }, 30), /unknown asr event/);
});

test("isSupported names every missing capability and excludes phones for now", () => {
  const ok = { crossOriginIsolated: true, SharedArrayBuffer: function () {}, isSecureContext: true, AudioWorkletNode: function () {}, navigator: { mediaDevices: { getUserMedia() {} }, userAgent: "Chrome desktop" } };
  assert.deepEqual(isSupported(ok), { ok: true, reasons: [] });
  const phone = { ...ok, navigator: { ...ok.navigator, userAgent: "Mozilla/5.0 (Linux; Android 14) Mobile" } };
  assert.equal(isSupported(phone).ok, false);
  assert.match(isSupported(phone).reasons[0], /phones/);
  const bare = { navigator: {} };
  assert.equal(isSupported(bare).reasons.length, 5);
});

test("supportIssues gives stable codes (the page localises them) in isSupported's order", () => {
  const ok = { crossOriginIsolated: true, SharedArrayBuffer: function () {}, isSecureContext: true, AudioWorkletNode: function () {}, navigator: { mediaDevices: { getUserMedia() {} }, userAgent: "Chrome desktop" } };
  assert.deepEqual(supportIssues(ok), []);
  assert.deepEqual(supportIssues({ ...ok, navigator: { ...ok.navigator, userAgent: "iPhone" } }), ["phone"]);
  const bare = { navigator: {} };
  assert.deepEqual(supportIssues(bare), ["isolation", "shared_memory", "secure_context", "audio_worklet", "capture"]);
  assert.deepEqual(isSupported(bare).reasons, supportIssues(bare).map((code) => SUPPORT_REASONS[code]));
});

test("a language that endpoints twice before the other commits the first window instead of overwriting it", () => {
  let state = initialAsrState(["kk", "ru"]);
  state = reduceAsrEvent(state, { type: "partial", language: "ru", detail: partial("бир", 0.3) }, 900).state;
  let out = reduceAsrEvent(state, { type: "result", language: "kk", detail: result("бір", 0.9) }, 1000);
  assert.deepEqual(out.emits, []);
  out = reduceAsrEvent(out.state, { type: "result", language: "kk", detail: result("екі", 0.9) }, 1300);
  assert.deepEqual(out.emits, [{ type: "utterance", language: "kk", text: "бір", confidence: 0.9 }]); // first window closed, nothing lost
  assert.deepEqual(out.state.pending, { kk: { text: "екі", confidence: 0.9 } });
  assert.equal(out.state.windowOpenedAt, 1300);
  assert.ok(out.state.suppressUntil.ru > 1300); // ru's partial was used for the vote; its late result is a duplicate
  out = reduceAsrEvent(out.state, { type: "result", language: "ru", detail: result("бир", 0.3) }, 1400);
  assert.deepEqual(out.emits, []); // suppressed
  out = reduceAsrEvent(out.state, { type: "tick" }, 2600);
  assert.deepEqual(out.emits, [{ type: "utterance", language: "kk", text: "екі", confidence: 0.9 }]);
});

// --- language locking (STT Tier B, ADR D40) -------------------------------------------------
// Two recognisers running for a whole call is the memory half of the phone gate (D25). Once
// the language is settled the loser can stop being fed; a confidence drop reopens both,
// because that is what a language switch looks like.

const bothFire = (state, now, kk, ru, opts) => {
  let emits = [];
  for (const [language, hyp] of [["kk", kk], ["ru", ru]]) {
    const step = reduceAsrEvent(state, { type: "result", language, detail: { text: hyp.text, result: [{ conf: hyp.confidence }] } }, now, opts);
    state = step.state;
    emits = emits.concat(step.emits);
  }
  return { state, emits };
};

test("locking: after the configured number of utterances only the winning language stays active", () => {
  const opts = { lock: { after: 2, confFloor: 0 } };
  let state = initialAsrState(["kk", "ru"]);
  assert.deepEqual(state.languages, ["kk", "ru"]);
  for (let i = 0; i < 2; i += 1) {
    state = bothFire(state, 1000 * (i + 1), { text: "жоқ", confidence: 0.4 }, { text: "здравствуйте это банк", confidence: 0.95 }, opts).state;
  }
  assert.deepEqual(state.languages, ["ru"], "the winner should be the only active recognizer");
  assert.equal(state.locked, "ru");
});

test("locking: a locked call commits on the winner's result alone, without waiting for the window", () => {
  const opts = { lock: { after: 1, confFloor: 0 } };
  let state = initialAsrState(["kk", "ru"]);
  state = bothFire(state, 1000, { text: "жоқ", confidence: 0.3 }, { text: "это банк", confidence: 0.99 }, opts).state;
  assert.deepEqual(state.languages, ["ru"]);
  const step = reduceAsrEvent(state, { type: "result", language: "ru", detail: { text: "назовите код", result: [{ conf: 0.9 }] } }, 2000, opts);
  assert.equal(step.emits.length, 1, "no second language to wait for");
  assert.equal(step.emits[0].text, "назовите код");
});

test("locking: a confidence drop reopens both recognizers", () => {
  const opts = { lock: { after: 1, confFloor: 0.8 } };
  let state = initialAsrState(["kk", "ru"]);
  state = bothFire(state, 1000, { text: "жоқ", confidence: 0.3 }, { text: "это банк", confidence: 0.99 }, opts).state;
  assert.deepEqual(state.languages, ["ru"]);
  const step = reduceAsrEvent(state, { type: "result", language: "ru", detail: { text: "мага шорт", result: [{ conf: 0.55 }] } }, 2000, opts);
  assert.deepEqual(step.state.languages, ["kk", "ru"], "a low-confidence utterance looks like a language switch");
  assert.equal(step.state.locked, null);
});

test("locking: off by default — the shipped behaviour is unchanged", () => {
  let state = initialAsrState(["kk", "ru"]);
  for (let i = 0; i < 5; i += 1) {
    state = bothFire(state, 1000 * (i + 1), { text: "жоқ", confidence: 0.4 }, { text: "это банк", confidence: 0.95 }).state;
  }
  assert.deepEqual(state.languages, ["kk", "ru"]);
});
