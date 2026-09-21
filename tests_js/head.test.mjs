import test from "node:test";
import assert from "node:assert/strict";
import { decodeTactics, hybridRow, riskProba, tacticProba } from "../site/core/head.js";
import { compileReassurance, hardSignalFeatures, reassuranceFeature } from "../site/core/lexicon.js";
import { config, fixtureEmbed, parity, weights } from "./helpers.mjs";

test("risk head reproduces the Python score for every fixture transcript (same embeddings)", async () => {
  const matcher = compileReassurance(weights.lexicon.reassurance);
  const hardIds = config.taxonomy.hard_signal_ids;
  for (const c of parity.cases) {
    const [embedding] = await fixtureEmbed([c.transcript]);
    const row = hybridRow(embedding, hardSignalFeatures(c.transcript, weights.lexicon.cues, hardIds), reassuranceFeature(c.transcript, matcher));
    const risk = riskProba(row, weights.risk_head);
    assert.ok(Math.abs(risk - c.score.risk) < 1e-6, `${c.id}: ${risk} vs ${c.score.risk}`);
  }
});

test("tactic head + decode reproduce the tag set (before cue merge) ordering rules", async () => {
  const [embedding] = await fixtureEmbed([parity.cases[0].transcript]);
  const probs = tacticProba(embedding, weights.tactic_head, weights.label_space);
  assert.equal(probs.length, weights.label_space.length);
  const decoded = decodeTactics(probs, weights.label_space, weights.tactic_head.threshold);
  for (let i = 1; i < decoded.length; i += 1) {
    assert.ok(decoded[i - 1][1] > decoded[i][1] || (decoded[i - 1][1] === decoded[i][1] && decoded[i - 1][0] < decoded[i][0]));
  }
});

test("risk head rejects a row of the wrong width", () => {
  assert.throws(() => riskProba([0, 1, 2], weights.risk_head), RangeError);
});

test("decodeTactics honours per-tactic thresholds (1:1 with labels.decode_tactics, ADR D30)", () => {
  const probs = [0.7, 0.7, 0.55];
  const space = ["otp_request", "urgency", "secrecy"];
  assert.deepEqual(decodeTactics(probs, space, 0.5), [["otp_request", 0.7], ["urgency", 0.7], ["secrecy", 0.55]]);
  assert.deepEqual(decodeTactics(probs, space, 0.5, { otp_request: 0.8, secrecy: 0.5 }), [["urgency", 0.7], ["secrecy", 0.55]]);
});
