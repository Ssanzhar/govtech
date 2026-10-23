/* Golden parity for site/core/word-align.js + word-lid.js tagWithHints against the spike's
   `spanvote/inherit.py` (ADR D70): GigaAM's words inherit the live span-vote tags.
   tests_js/fixtures/word_align.json comes from scripts/export_asr_wordlevel.py (DEV rows). */
import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { REPO } from "./helpers.mjs";
import { createWordLid } from "../site/core/word-lid.js";
import { alignWords, createInheritingTagger, hintsFor, levenshtein, similarity } from "../site/core/word-align.js";

const bundle = JSON.parse(readFileSync(join(REPO, "site", "core", "asr-wordlevel.json"), "utf8"));
const fixture = JSON.parse(readFileSync(join(REPO, "tests_js", "fixtures", "word_align.json"), "utf8"));
const lid = createWordLid(bundle.lid);

test("fixture sanity: the bundle ships the parameters chosen on DEV", () => {
  assert.deepEqual(bundle.inherit, { min_sim: fixture.inherit.min_sim, weight: fixture.inherit.weight });
  assert.ok(fixture.cases.length >= 30);
  assert.ok(fixture.cases.some((c) => c.hints.some((h) => h === null)), "some words have no partner");
});

test("alignment, hints and inherited tags match Python on every DEV case", () => {
  const { min_sim: minSim, weight } = bundle.inherit;
  for (const c of fixture.cases) {
    const label = c.giga.join(" ");
    const pairs = alignWords(c.giga, c.live.map((w) => w.word));
    assert.equal(pairs.length, c.pairs.length, `${label}: pair count`);
    pairs.forEach(([i, j, s], k) => {
      assert.deepEqual([i, j], c.pairs[k].slice(0, 2), `${label}: pair ${k}`);
      assert.ok(Math.abs(s - c.pairs[k][2]) < 1e-12, `${label}: similarity ${k}`);
    });
    assert.deepEqual(hintsFor(c.giga, c.live, minSim), c.hints, `${label}: hints`);
    assert.deepEqual(lid.tagWithHints(c.giga, c.hints, weight), c.tags, `${label}: tags`);
  }
});

test("similarity is normalised Levenshtein by code point", () => {
  assert.equal(levenshtein("код", "кот"), 1);
  assert.equal(levenshtein("", "абв"), 3);
  assert.equal(similarity("", ""), 1);
  assert.ok(Math.abs(similarity("қызмет", "кызмет") - 5 / 6) < 1e-12);
});

test("the inheriting tagger keeps the live tag of an aligned word; letters still decide", () => {
  const tag = createInheritingTagger(lid, bundle.inherit);
  const live = [{ word: "назовите", lang: "ru" }, { word: "код", lang: "ru" }];
  assert.deepEqual(tag("назовите код", live).map((w) => w.lang), ["ru", "ru"]);
  // a Kazakh-only letter still decides, whatever the live tag
  assert.equal(tag("қызмет", [{ word: "кызмет", lang: "ru" }])[0].lang, "kk");
  // without live words (light mode) or parameters it is the plain word LID
  assert.deepEqual(tag("сіздің код келді ме"), lid.tagText("сіздің код келді ме"));
  assert.deepEqual(createInheritingTagger(lid, null)("назовите код", live), lid.tagText("назовите код"));
});

test("edge cases: empty sides, ties prefer the later new word, a hint against the LID wins at weight 16", () => {
  assert.deepEqual(alignWords([], ["код"]), []);
  assert.deepEqual(alignWords(["код"], []), []);
  assert.deepEqual(alignWords(["аа", "аа"], ["аа"]), [[1, 0, 1]]);
  assert.deepEqual(hintsFor(["зzz"], [{ word: "код", lang: "ru" }], 0.34), [null]); // too dissimilar
  const tags = lid.tagWithHints(["сіздің", "код", "келді"], [null, "ru", null], 16);
  assert.equal(tags[1], "ru");
  assert.deepEqual(lid.tagWithHints([], [], 16), []);
});
