/* Golden parity: the JS core must reproduce the Python implementation on every fixture
   case -- score(), explain() for ru + kk, and the whole live-session trajectory. */
import test from "node:test";
import assert from "node:assert/strict";
import { createScorer } from "../site/core/score.js";
import { explain } from "../site/core/explain.js";
import { advance, initialSession } from "../site/core/session.js";
import { config, fixtureEmbed, parity, weights } from "./helpers.mjs";

const score = createScorer({ weights, config, embed: fixtureEmbed });
// Meter scores are 100x risk. Two noise sources below any display/decision resolution:
// dot-product summation order (numpy vs JS, ~1e-9 in risk) and sentence-transformers'
// batch-padding nondeterminism (Python embeds the window text twice per turn, alone and
// padded in the utterance batch, ~1e-6 apart). Bands, latches and evidence stay exact.
const TOL = 1e-3;

test("fixture sanity: config knobs match what Python used", () => {
  assert.equal(config.scoring.risk_threshold, parity.config.risk_threshold);
  assert.equal(config.meter.enter, parity.config.enter);
  assert.equal(config.meter.exit, parity.config.exit);
  assert.equal(config.meter.alpha_up, parity.config.alpha_up);
  assert.equal(config.meter.alpha_down, parity.config.alpha_down);
  assert.ok(parity.cases.length >= 20);
});

test("score(): risk, tags, attributions match Python on every case", async () => {
  for (const c of parity.cases) {
    const r = await score(c.transcript);
    assert.ok(Math.abs(r.risk - c.score.risk) < 1e-6, `${c.id} risk ${r.risk} vs ${c.score.risk}`);
    assert.ok(Math.abs(r.raw_confidence - c.score.raw_confidence) < 1e-6, `${c.id} confidence`);
    assert.deepEqual(r.tags.map((t) => t.id), c.score.tags.map((t) => t.id), `${c.id} tag ids`);
    r.tags.forEach((t, i) => assert.ok(Math.abs(t.weight - c.score.tags[i].weight) < 1e-6, `${c.id} weight ${t.id}`));
    assert.deepEqual(r.attributions, c.score.attributions, `${c.id} attributions`);
  }
});

test("explain(): identical strings for ru and kk on every case", async () => {
  for (const c of parity.cases) {
    const r = await score(c.transcript);
    for (const locale of ["ru", "kk"]) {
      const e = explain(r, c.transcript, locale, config);
      const want = c.explanations[locale];
      assert.equal(e.reason, want.reason, `${c.id}/${locale} reason`);
      assert.equal(e.confidence_label, want.confidence_label, `${c.id}/${locale} confidence label`);
      assert.equal(e.caveat, want.caveat, `${c.id}/${locale} caveat`);
      assert.equal(e.human_note, want.human_note, `${c.id}/${locale} human note`);
      assert.deepEqual(e.highlights.map((s) => s.text), want.highlights, `${c.id}/${locale} highlights`);
    }
  }
});

test("live session: meter trajectory, bands, evidence and advice match Python turn by turn", async () => {
  for (const c of parity.cases) {
    let state = initialSession(c.locale, config);
    for (let i = 0; i < c.utterances.length; i += 1) {
      const want = c.live.turns[i];
      const out = await advance(state, { text: c.utterances[i], confidence: 1 }, { score, config });
      state = out.state;
      const u = out.update;
      assert.equal(u.window_text, want.window_text, `${c.id} turn ${i + 1} window`);
      assert.ok(Math.abs(u.meter.score - want.score) < TOL, `${c.id} turn ${i + 1} score ${u.meter.score} vs ${want.score}`);
      assert.equal(u.meter.latched, want.latched, `${c.id} turn ${i + 1} latched`);
      assert.deepEqual(u.meter.hard_signal_ids, want.hard_signal_ids, `${c.id} turn ${i + 1} hard signals`);
      assert.equal(u.band, want.band, `${c.id} turn ${i + 1} band`);
      assert.ok(Math.abs(u.result.risk - want.risk) < 1e-6, `${c.id} turn ${i + 1} risk`);
      assert.deepEqual(u.new_evidence.map((s) => s.text), want.new_evidence, `${c.id} turn ${i + 1} evidence`);
      assert.deepEqual(u.recommendation.advices, want.recommendation.advices, `${c.id} turn ${i + 1} advices`);
      assert.deepEqual(u.recommendation.verification_questions, want.recommendation.verification_questions, `${c.id} turn ${i + 1} questions`);
      assert.equal(u.recommendation.softened, want.recommendation.softened, `${c.id} turn ${i + 1} softened`);
      assert.equal(u.recommendation.note, want.recommendation.note, `${c.id} turn ${i + 1} note`);
    }
    assert.deepEqual(state.tags.map((t) => t.id).sort(), c.live.tags.map((t) => t.id).sort(), `${c.id} accumulated tags`);
  }
});
