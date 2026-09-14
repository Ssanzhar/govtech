import test from "node:test";
import assert from "node:assert/strict";
import { compileReassurance, hardSignalFeatures, matchCues, reassuranceFeature } from "../site/core/lexicon.js";
import { config, weights } from "./helpers.mjs";

const cues = weights.lexicon.cues;
const hardIds = config.taxonomy.hard_signal_ids;
const matcher = compileReassurance(weights.lexicon.reassurance);

test("cue block fires on a verbatim OTP request and stays all-zero on a legit reassurance", () => {
  const scam = hardSignalFeatures("Продиктуйте код из SMS, никому не говорите", cues, hardIds);
  assert.equal(scam.length, hardIds.length);
  assert.equal(scam[hardIds.indexOf("otp_request")], 1);
  assert.equal(scam[hardIds.indexOf("secrecy")], 1);
  const legit = hardSignalFeatures("Код называть не нужно, это служба банка", cues, hardIds);
  assert.deepEqual(legit, hardIds.map(() => 0));
});

test("matchCues returns verbatim spans that slice back from the original text", () => {
  const text = "Алло. Продиктуйте код из SMS сейчас.";
  const matches = matchCues(text, cues, hardIds);
  assert.ok(matches.length >= 1);
  for (const { span } of matches) assert.equal(text.slice(span.start, span.end), span.text);
});

test("matching is case-insensitive like Python's str.lower()", () => {
  const upper = hardSignalFeatures("ПРОДИКТУЙТЕ КОД ИЗ SMS", cues, hardIds);
  assert.equal(upper[hardIds.indexOf("otp_request")], 1);
});

test("reassurance fires when a sensitive term sits near a negation-of-need", () => {
  assert.equal(reassuranceFeature("Код из SMS называть не нужно.", matcher), 1);
  assert.equal(reassuranceFeature("Продиктуйте код из SMS.", matcher), 0);
});
