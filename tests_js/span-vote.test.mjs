/* Golden parity for site/core/span-vote.js against the spike's span vote + freeze loop (ADR D63):
   tests_js/fixtures/span_vote.json replays DEV rows (built from train text) step by step. */
import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { REPO } from "./helpers.mjs";
import { createWordLid } from "../site/core/word-lid.js";
import { cleanWords, createSpanVoter, initialFreeze, spanVote } from "../site/core/span-vote.js";

const bundle = JSON.parse(readFileSync(join(REPO, "site", "core", "asr-wordlevel.json"), "utf8"));
const fixture = JSON.parse(readFileSync(join(REPO, "tests_js", "fixtures", "span_vote.json"), "utf8"));
const voter = createSpanVoter(bundle.span_vote, createWordLid(bundle.lid));
const TOL = 1e-9;

const closeSpans = (got, want, label) => {
  assert.equal(got.length, want.length, `${label}: span count`);
  got.forEach((g, i) => {
    assert.ok(Math.abs(g[0] - want[i][0]) < TOL && Math.abs(g[1] - want[i][1]) < TOL, `${label}: span ${i} bounds`);
    assert.equal(g[2], want[i][2], `${label}: span ${i} language`);
  });
};

test("fixture sanity: the measured parameters and English in the replay", () => {
  assert.equal(bundle.span_vote.switch_penalty, 0.25);
  assert.equal(bundle.span_vote.missing_logit, -2.0);
  assert.equal(bundle.span_vote.freeze_s, 1.0);
  assert.equal(bundle.span_vote.features.length, 13);
  assert.ok(fixture.rows.length >= 30);
  const langs = new Set(fixture.rows.flatMap((r) => r.steps.flatMap((s) => s.words.map((w) => w[3]))));
  for (const l of ["kk", "ru", "en"]) assert.ok(langs.has(l), `no ${l} word in the replay`);
});

test("every live step matches Python: probabilities, words, spans and the freeze", () => {
  for (const row of fixture.rows) {
    let freeze = initialFreeze();
    for (const step of row.steps) {
      const label = `${row.id}/${row.condition}/${row.channel}@${step.t}`;
      const out = voter.label(step.hyps, freeze);
      for (const l of ["kk", "ru", "en"]) {
        assert.equal(out.probs[l].length, step.probs[l].length, `${label}: ${l} word count`);
        out.probs[l].forEach((p, i) => assert.ok(Math.abs(p - step.probs[l][i]) < TOL, `${label}: ${l}[${i}] p ${p} vs ${step.probs[l][i]}`));
      }
      assert.deepEqual(out.words.map((w) => [w[0], w[3]]), step.words.map((w) => [w[0], w[3]]), `${label}: words`);
      closeSpans(out.labelled, step.labelled, label);
      closeSpans(out.freeze.spans, step.frozen_after, `${label} frozen`);
      freeze = out.freeze;
    }
  }
});

test("spanVote: nothing decoded -> nothing shown; inputs are not mutated", () => {
  assert.deepEqual(spanVote({ kk: [], ru: [], en: [] }, { kk: [], ru: [], en: [] }, { switch: 0.25, missing: -2 }), { words: [], labelled: [] });
  const hyps = { kk: [["сәлем", 0, 0.5, 0.9]], ru: [], en: [] };
  const copy = JSON.parse(JSON.stringify(hyps));
  voter.label(hyps);
  assert.deepEqual(hyps, copy);
});

test("cleanWords accepts Vosk word objects and drops empty words", () => {
  assert.deepEqual(cleanWords([{ word: "Сәлем,", start: 0, end: 0.4, conf: 0.8 }, { word: "--", start: 0.4, end: 0.5 }, { word: "код", start: 0.5, end: 0.9 }]),
    [["сәлем", 0, 0.4, 0.8], ["код", 0.5, 0.9, 1]]);
});
