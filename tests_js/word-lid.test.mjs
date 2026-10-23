/* Golden parity for site/core/word-lid.js against the spike's WordLID (ADR D64):
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

test("fixture sanity: real transcripts tagged kk and ru only (ADR D70)", () => {
  assert.ok(fixture.cases.length >= 60);
  const tags = new Set(fixture.cases.flatMap((c) => c.tags));
  assert.deepEqual([...tags].sort(), ["kk", "ru"]);
});

test("normalisation matches Python word for word", () => {
  for (const c of fixture.cases) assert.deepEqual(normalizeWords(c.text), c.words, c.text);
});

test("tags match Python on every case", () => {
  for (const c of fixture.cases) assert.deepEqual(lid.tag(c.words), c.tags, c.text);
});

test("tagText pairs each normalised word with its tag", () => {
  assert.deepEqual(lid.tagText("Hello, СӘЛЕМ!"), [{ word: "hello", lang: "kk" }, { word: "сәлем", lang: "kk" }]);
  assert.deepEqual(lid.tagText(""), []);
});

test("letters decide first: Latin -> en, Kazakh-only letters -> kk, plain Cyrillic -> undecided", () => {
  assert.equal(letterLanguage("code"), "en");
  assert.equal(letterLanguage("қызмет"), "kk");
  assert.equal(letterLanguage("код"), null);
  assert.equal(normalizeText("Ёлка_ёж"), "елка еж");
});

test("a Latin word is neutral: it follows its neighbours (no English state, ADR D70)", () => {
  assert.deepEqual(lid.tag(normalizeWords("kaspi арқылы жіберіңіз")), ["kk", "kk", "kk"]);
  assert.deepEqual(lid.tag(normalizeWords("переведите через kaspi")).at(-1), lid.tag(normalizeWords("переведите через kaspi")).at(-2));
  assert.ok(!lid.tag(normalizeWords("the code please")).includes("en"));
});

test("context: a plain-Cyrillic word follows Kazakh neighbours", () => {
  const tags = lid.tag(normalizeWords("сіздің код келді ме"));
  assert.equal(tags[1], "kk");
});
