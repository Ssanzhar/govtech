/* A context that stops must say so: the page used to keep saying "listening" over silence. */
import test from "node:test";
import assert from "node:assert/strict";
import { startAudioContext } from "../site/core/audio-context.js";

class FakeContext {
  constructor(state = "running") { this.state = state; this.resumes = 0; this.listeners = []; FakeContext.last = this; }
  addEventListener(_type, fn) { this.listeners.push(fn); }
  set(state) { this.state = state; for (const fn of this.listeners) fn(); }
  async resume() { this.resumes += 1; }
}

test("the interruption is reported, resumed, and its end reported", async () => {
  const seen = [];
  const ctx = await startAudioContext((stopped) => seen.push(stopped), FakeContext);
  ctx.set("suspended");
  assert.equal(ctx.resumes, 1);
  ctx.set("running");
  ctx.set("interrupted"); // Safari's own state
  assert.equal(ctx.resumes, 2);
  assert.deepEqual(seen, [true, false, true]);
});

test("a context that starts suspended is asked to run, without waiting for ever", async () => {
  class Suspended extends FakeContext {
    constructor() { super("suspended"); }
    resume() { this.resumes += 1; return new Promise(() => {}); } // pending for as long as the interruption lasts
  }
  const started = Date.now();
  const ctx = await startAudioContext(() => {}, Suspended);
  assert.equal(ctx.resumes, 1);
  assert.ok(Date.now() - started < 3000);
});

test("closing is not an interruption", async () => {
  const seen = [];
  const ctx = await startAudioContext((stopped) => seen.push(stopped), FakeContext);
  ctx.set("closed");
  assert.deepEqual(seen, []);
});

// Started after a long download the click no longer counts as a gesture: no statechange ever fires,
// and the page said "listening" over silence.
test("a context still suspended after the wait is reported", async () => {
  class Stuck extends FakeContext {
    constructor() { super("suspended"); }
    async resume() { this.resumes += 1; } // resolves, and the context stays suspended
  }
  const seen = [];
  await startAudioContext((stopped) => seen.push(stopped), Stuck);
  assert.deepEqual(seen, [true]);
});

test("a context that resumes in time reports nothing", async () => {
  class Wakes extends FakeContext {
    constructor() { super("suspended"); }
    async resume() { this.state = "running"; }
  }
  const seen = [];
  await startAudioContext((stopped) => seen.push(stopped), Wakes);
  assert.deepEqual(seen, []);
});
