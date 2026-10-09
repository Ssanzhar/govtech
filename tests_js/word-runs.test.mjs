import test from "node:test";
import assert from "node:assert/strict";
import { alignTokens, evidenceRanges, overlaps, runsOf } from "../site/core/word-runs.js";

test("alignTokens keeps verbatim tokens with offsets and their tags", () => {
  const tokens = alignTokens("сәлем код келді", [{ word: "сәлем", lang: "kk" }, { word: "код", lang: "kk" }, { word: "келді", lang: "kk" }]);
  assert.deepEqual(tokens.map((t) => [t.text, t.start, t.end, t.lang]), [["сәлем", 0, 5, "kk"], ["код", 6, 9, "kk"], ["келді", 10, 15, "kk"]]);
  assert.equal(alignTokens("Ёлка", [{ word: "елка", lang: "ru" }])[0].text, "Ёлка"); // normalised comparison
});

test("alignTokens refuses to guess when the words do not line up", () => {
  assert.equal(alignTokens("из-за кода", [{ word: "из", lang: "ru" }, { word: "за", lang: "ru" }, { word: "кода", lang: "ru" }]), null);
  assert.equal(alignTokens("код", []), null);
});

test("runsOf groups consecutive words of one language", () => {
  const runs = runsOf([{ lang: "kk" }, { lang: "kk" }, { lang: "ru" }, { lang: "kk" }]);
  assert.deepEqual(runs.map((r) => [r.lang, r.items.length]), [["kk", 2], ["ru", 1], ["kk", 1]]);
  assert.deepEqual(runsOf([]), []);
});

test("evidence ranges mark the tokens they overlap", () => {
  const text = "назовите код из смс";
  const ranges = evidenceRanges(text, ["код из смс", "", "нет такого"]);
  assert.deepEqual(ranges, [[9, 19]]);
  const tokens = alignTokens(text, text.split(" ").map((word) => ({ word, lang: "ru" })));
  assert.deepEqual(tokens.map((t) => overlaps(t, ranges)), [false, true, true, true]);
});
