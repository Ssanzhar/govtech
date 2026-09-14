import test from "node:test";
import assert from "node:assert/strict";
import { rollingWindow } from "../site/core/session.js";
import { band, initialMeter, updateMeter } from "../site/core/meter.js";
import { config } from "./helpers.mjs";

const w = config.window;
const m = config.meter;

test("rolling window keeps head + tail once over the cap and always includes the newest turn", () => {
  const utterances = Array.from({ length: 40 }, (_, i) => `turn ${i} ` + "x".repeat(300));
  const text = rollingWindow(utterances, w);
  assert.ok(text.length <= w.max_chars + 310);
  assert.ok(text.startsWith("turn 0 "));
  assert.ok(text.includes(`turn ${utterances.length - 1} `));
  assert.ok(!text.includes("turn 10 "));
});

test("rolling window returns the full text under the cap", () => {
  assert.equal(rollingWindow(["a", "b"], w), "a\nb");
});

test("meter rises fast, decays slowly, floors on hard signals and latches with hysteresis", () => {
  let s = updateMeter(initialMeter(), { risk: 0.9 }, m);
  assert.equal(s.score, 45);
  s = updateMeter(s, { risk: 0.9 }, m);
  assert.equal(band(s.score, m), "high");
  assert.equal(s.latched, true);
  s = updateMeter(s, { risk: 0.0 }, m);
  assert.ok(s.score > 50 && s.latched, "slow decay keeps the latch");
  const floored = updateMeter(initialMeter(), { risk: 0.1, hardSignals: { otp_request: 1, secrecy: 0.95 } }, m);
  assert.equal(floored.score, m.double_hard_signal_floor);
  assert.deepEqual(floored.hard_signal_ids, ["otp_request", "secrecy"]);
});

test("meter validates its inputs", () => {
  assert.throws(() => updateMeter(initialMeter(), { risk: 1.5 }, m), RangeError);
  assert.throws(() => band(101, m), RangeError);
});
