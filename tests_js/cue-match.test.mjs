/* The cue matcher must decide exactly what Python decides -- it computes a model feature.
   Fixture: `python scripts/export_cue_match_fixture.py` (real recogniser output, ADR D39). */

import { strict as assert } from "node:assert";
import { readFileSync } from "node:fs";
import { test } from "node:test";

import { MATCHER_VERSION, budgetFor, findCue, normalize } from "../site/core/cue-match.js";

const fixture = JSON.parse(readFileSync(new URL("./fixtures/cue_match.json", import.meta.url), "utf8"));

test("matcher version matches the fixture's", () => {
  assert.equal(MATCHER_VERSION, fixture.matcher_version);
});

test("every fixture case resolves to the same span as Python", () => {
  let hits = 0;
  for (const { text, cue, span } of fixture.cases) {
    const got = findCue(text, cue);
    assert.deepEqual(got, span, `cue ${JSON.stringify(cue)} in ${JSON.stringify(text.slice(0, 70))}`);
    if (span) hits += 1;
  }
  assert.ok(hits > 0, "fixture must contain at least one match");
});

test("a fuzzy hit is still a verbatim slice of the original text", () => {
  for (const { text, span } of fixture.cases) {
    if (span) assert.equal(text.slice(span[0], span[1]).length, span[1] - span[0]);
  }
});

test("normalisation de-spaces and folds yo", () => {
  assert.equal(normalize("Продиктуйте КОД из SMS, пожалуйста!")[0], "продиктуйтекодизsmsпожалуйста");
  assert.equal(normalize("Счёт")[0], "счет");
});

test("budget is zero for short cues and grows monotonically", () => {
  assert.equal(budgetFor(7), 0);
  const budgets = [0, 5, 11, 12, 20, 21, 30, 31, 60].map(budgetFor);
  assert.deepEqual(budgets, [...budgets].sort((a, b) => a - b));
});
