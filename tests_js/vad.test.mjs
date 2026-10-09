/* The WebView-safe voice mode's segmenter (ADR D67): a deterministic energy + zero-crossing VAD.
   Synthetic signals only: digital silence, a seeded "speech" proxy (a 220 Hz tone plus white
   noise, which has speech-like energy and crossing rate), mains hum, a steady noise floor. */
import test from "node:test";
import assert from "node:assert/strict";
import { VAD_DEFAULTS, createVad, flushVad, frameFeatures, initialVadState, reduceVad } from "../site/core/vad.js";

/** Seeded uniform noise in [-1, 1) (an LCG: the same samples on every run). */
function noise(seed) {
  let s = seed >>> 0;
  return () => {
    s = (Math.imul(s, 1664525) + 1013904223) >>> 0;
    return s / 2147483648 - 1;
  };
}

/** Concatenate parts: {s: seconds, kind: "silence" | "speech" | "hum" | "noise", amp}. */
function signal(rate, parts, seed = 7) {
  const rnd = noise(seed);
  const total = parts.reduce((n, p) => n + Math.round(p.s * rate), 0);
  const out = new Float32Array(total);
  let at = 0;
  for (const p of parts) {
    const n = Math.round(p.s * rate);
    for (let i = 0; i < n; i++) {
      const t = (at + i) / rate;
      if (p.kind === "speech") out[at + i] = (p.amp ?? 0.3) * (0.6 * Math.sin(2 * Math.PI * 220 * t) + 0.4 * rnd());
      else if (p.kind === "hum") out[at + i] = (p.amp ?? 0.5) * Math.sin(2 * Math.PI * 50 * t);
      else if (p.kind === "noise") out[at + i] = (p.amp ?? 0.01) * rnd();
    }
    at += n;
  }
  return out;
}

/** Every event of a whole signal, pushed in `chunk`-sample pieces, then flushed. */
function run(rate, samples, { chunk = 128, options } = {}) {
  const vad = createVad(rate, options);
  const events = [];
  for (let i = 0; i < samples.length; i += chunk) events.push(...vad.push(samples.subarray(i, i + chunk)));
  events.push(...vad.flush());
  return events;
}
const segments = (events) => events.filter((e) => e.type === "segment");
const near = (actual, expected, tol, label) => assert.ok(Math.abs(actual - expected) <= tol, `${label}: ${actual} vs ${expected} (±${tol})`);

test("frame features: decibels of the RMS and zero crossings per second, rate-independent", () => {
  const rate = 16000;
  const silent = frameFeatures(new Float32Array(480), rate);
  assert.equal(silent.db, VAD_DEFAULTS.floorDb);
  assert.equal(silent.zcrPerS, 0);
  const sine = Float32Array.from({ length: 1600 }, (_, i) => 0.5 * Math.sin(2 * Math.PI * 1000 * (i / rate) + 0.1));
  const f = frameFeatures(sine, rate);
  near(f.db, 20 * Math.log10(0.5 / Math.SQRT2), 0.05, "dB");
  near(f.zcrPerS, 2000, 20, "crossings/s");
  const f48 = frameFeatures(Float32Array.from({ length: 4800 }, (_, i) => 0.5 * Math.sin(2 * Math.PI * 1000 * (i / 48000) + 0.1)), 48000);
  near(f48.zcrPerS, 2000, 20, "crossings/s at 48 kHz");
});

test("digital silence produces no events", () => {
  assert.deepEqual(run(16000, signal(16000, [{ s: 5, kind: "silence" }])), []);
});

for (const rate of [16000, 48000]) {
  test(`two phrases separated by a pause are two segments with tight bounds (${rate} Hz)`, () => {
    const samples = signal(rate, [{ s: 1, kind: "silence" }, { s: 1.5, kind: "speech" }, { s: 1, kind: "silence" }, { s: 2, kind: "speech" }, { s: 1, kind: "silence" }]);
    const events = run(rate, samples);
    const segs = segments(events);
    assert.equal(segs.length, 2);
    const frame = VAD_DEFAULTS.frameMs / 1000;
    near(segs[0].start, 1 - VAD_DEFAULTS.preRollMs / 1000, frame, "first start");
    near(segs[0].end, 2.5, frame, "first end");
    near(segs[1].start, 3.5 - VAD_DEFAULTS.preRollMs / 1000, frame, "second start");
    near(segs[1].end, 5.5, frame, "second end");
    assert.equal(segs[0].cut, "pause");
    // A segment closes one hangover after the speech ends (that wait is part of the latency).
    near(segs[0].at, 2.5 + VAD_DEFAULTS.hangoverMs / 1000, 2 * frame, "closed at");
    assert.deepEqual(events.map((e) => e.type), ["speech_start", "segment", "speech_start", "segment"]);
  });
}

test("a click and a burst shorter than the minimum phrase are dropped", () => {
  const samples = signal(16000, [{ s: 1, kind: "silence" }, { s: 0.05, kind: "speech" }, { s: 1, kind: "silence" }, { s: 0.22, kind: "speech" }, { s: 1, kind: "silence" }]);
  const events = run(16000, samples);
  assert.deepEqual(segments(events), []);
  assert.ok(events.some((e) => e.type === "speech_end" && e.dropped), "the short burst is reported as dropped");
});

test("a pause shorter than the hangover stays inside one phrase", () => {
  const samples = signal(16000, [{ s: 0.5, kind: "silence" }, { s: 1, kind: "speech" }, { s: 0.3, kind: "silence" }, { s: 1, kind: "speech" }, { s: 1, kind: "silence" }]);
  const segs = segments(run(16000, samples));
  assert.equal(segs.length, 1);
  near(segs[0].end, 2.8, 0.03, "end");
});

test("loud mains hum (50 Hz) is not speech: too few zero crossings", () => {
  assert.deepEqual(segments(run(16000, signal(16000, [{ s: 4, kind: "hum", amp: 0.5 }]))), []);
});

test("the noise floor adapts: phrases over steady background noise are still separated", () => {
  const samples = signal(16000, [
    { s: 3, kind: "noise", amp: 0.02 }, { s: 1.2, kind: "speech", amp: 0.4 }, { s: 1.2, kind: "noise", amp: 0.02 },
    { s: 1.2, kind: "speech", amp: 0.4 }, { s: 1.5, kind: "noise", amp: 0.02 },
  ]);
  // Background noise alone sits above the absolute minimum, so only the adaptive floor separates it.
  assert.ok(frameFeatures(signal(16000, [{ s: 0.03, kind: "noise", amp: 0.02 }]), 16000).db > VAD_DEFAULTS.absMinDb);
  const segs = segments(run(16000, samples));
  assert.equal(segs.length, 2);
  near(segs[0].end, 4.2, 0.06, "first end");
  near(segs[1].end, 6.6, 0.06, "second end");
});

test("an unbroken phrase is cut at the hard maximum and continues as the next segment", () => {
  const segs = segments(run(16000, signal(16000, [{ s: 20, kind: "speech" }, { s: 1, kind: "silence" }])));
  assert.equal(segs.length, 2);
  assert.equal(segs[0].cut, "max");
  near(segs[0].end - segs[0].start, VAD_DEFAULTS.maxMs / 1000, 0.031, "first length");
  near(segs[1].start, segs[0].end, 1e-9, "contiguous");
  near(segs[1].end, 20, 0.03, "second end");
});

test("after the soft maximum a short pause is enough to close the phrase", () => {
  const short = { s: 0.3, kind: "silence" }; // longer than softHangoverMs, shorter than hangoverMs
  const segs = segments(run(16000, signal(16000, [{ s: 9, kind: "speech" }, short, { s: 2, kind: "speech" }, { s: 1, kind: "silence" }])));
  assert.equal(segs.length, 2);
  near(segs[0].end, 9, 0.03, "cut at the pause");
  assert.equal(segs[0].cut, "pause");
});

test("flush closes a phrase that is still open when the microphone stops", () => {
  const events = run(16000, signal(16000, [{ s: 0.5, kind: "silence" }, { s: 1.5, kind: "speech" }]));
  const segs = segments(events);
  assert.equal(segs.length, 1);
  assert.equal(segs[0].cut, "flush");
  near(segs[0].end, 2, 0.03, "end");
});

test("events do not depend on how the audio is chunked", () => {
  const samples = signal(48000, [{ s: 0.7, kind: "silence" }, { s: 1.3, kind: "speech" }, { s: 0.9, kind: "silence" }, { s: 0.8, kind: "speech" }, { s: 0.9, kind: "silence" }]);
  const a = run(48000, samples, { chunk: 128 });
  const b = run(48000, samples, { chunk: 4096 });
  const c = run(48000, samples, { chunk: samples.length });
  assert.deepEqual(a, b);
  assert.deepEqual(a, c);
  assert.equal(segments(a).length, 2);
});

test("the reducer never mutates its input state", () => {
  const state = initialVadState(VAD_DEFAULTS);
  const frozen = JSON.stringify(state);
  const out = reduceVad(state, { db: -10, zcrPerS: 3000 }, 0.03);
  assert.equal(JSON.stringify(state), frozen);
  assert.notEqual(out.state, state);
  const flushed = flushVad(out.state, 0.06);
  assert.equal(flushed.state.phase, "silence");
});

test("options override the defaults; bad sample rates are refused", () => {
  const segs = segments(run(16000, signal(16000, [{ s: 0.5, kind: "silence" }, { s: 1, kind: "speech" }, { s: 0.3, kind: "silence" }, { s: 1, kind: "speech" }, { s: 1, kind: "silence" }]), { options: { hangoverMs: 200 } }));
  assert.equal(segs.length, 2);
  assert.throws(() => createVad(0), /sample rate/);
});
