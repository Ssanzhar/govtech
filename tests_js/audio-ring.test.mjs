import test from "node:test";
import assert from "node:assert/strict";
import { createAudioRing } from "../site/core/audio-ring.js";

const ramp = (from, n) => Float32Array.from({ length: n }, (_, i) => from + i);

test("slice returns exactly the samples of a stream-time span", () => {
  const ring = createAudioRing(10, 5); // 10 Hz, 50 samples
  ring.push(ramp(0, 7));
  ring.push(ramp(7, 13));
  assert.equal(ring.seconds, 2);
  assert.deepEqual(Array.from(ring.slice(0.5, 1.2)), [5, 6, 7, 8, 9, 10, 11]);
  assert.deepEqual(Array.from(ring.slice(-1, 0.3)), [0, 1, 2]);
  assert.deepEqual(Array.from(ring.slice(1.9, 9)), [19]);
  assert.equal(ring.slice(3, 4).length, 0);
});

test("old audio is dropped once the ring wraps; times keep counting from the stream start", () => {
  const ring = createAudioRing(10, 2); // 20 samples
  for (let i = 0; i < 5; i++) ring.push(ramp(i * 10, 10)); // 50 samples, 0..49
  assert.deepEqual(Array.from(ring.slice(0, 5)), Array.from(ramp(30, 20)));
  assert.deepEqual(Array.from(ring.slice(3.5, 4.2)), [35, 36, 37, 38, 39, 40, 41]);
  ring.push(ramp(50, 45)); // a chunk larger than the ring
  assert.equal(ring.seconds, 9.5);
  assert.deepEqual(Array.from(ring.slice(7.5, 9.5)), Array.from(ramp(75, 20)));
});

test("a bad sample rate is refused", () => {
  assert.throws(() => createAudioRing(0), /sample rate/);
});
