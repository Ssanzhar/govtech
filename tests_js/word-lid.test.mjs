/* Golden parity for site/core/word-lid.js against the spike's WordLID (ADR D63):
   tests_js/fixtures/word_lid.json comes from scripts/export_asr_wordlevel.py. */
import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { REPO } from "./helpers.mjs";
import { createWordLid, letterLanguage, normalizeText, normalizeWords } from "../site/core/word-lid.js";

const bundle = JSON.parse(readFileSync(join(REPO, "site", "core", "asr-wordlevel.json"), "utf8"));
const fixture = JSON.parse(readFileSync(join(REPO, "tests_js", "fixtures", "word_lid.json"), "utf8"));
const lid = createWordLid(bundle.lid);

test("fixture sanity: real transcripts in all three languages", () => {
  assert.ok(fixture.cases.length >= 60);
  const tags = new Set(fixture.cases.flatMap((c) => c.tags));
  for (const lang of ["kk", "ru", "en"]) assert.ok(tags.has(lang), `no ${lang} tag in the fixture`);
});

test("normalisation matches Python word for word", () => {
  for (const c of fixture.cases) assert.deepEqual(normalizeWords(c.text), c.words, c.text);
});

test("tags match Python on every case", () => {
  for (const c of fixture.cases) assert.deepEqual(lid.tag(c.words), c.tags, c.text);
});

test("tagText pairs each normalised word with its tag", () => {
  assert.deepEqual(lid.tagText("Hello, СӘЛЕМ!"), [{ word: "hello", lang: "en" }, { word: "сәлем", lang: "kk" }]);
  assert.deepEqual(lid.tagText(""), []);
});

test("letters decide first: Latin -> en, Kazakh-only letters -> kk, plain Cyrillic -> undecided", () => {
  assert.equal(letterLanguage("code"), "en");
  assert.equal(letterLanguage("қызмет"), "kk");
  assert.equal(letterLanguage("код"), null);
  assert.equal(normalizeText("Ёлка_ёж"), "елка еж");
});

test("context: a plain-Cyrillic word follows Kazakh neighbours", () => {
  const tags = lid.tag(normalizeWords("сіздің код келді ме"));
  assert.equal(tags[1], "kk");
});
