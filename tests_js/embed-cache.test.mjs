/* The embedding cache (ADR D78): a live call re-scores every utterance's evidence span at every
   turn, so phrase n cost n+1 e5 runs. Texts are embedded one at a time (embed-worker.js), so a
   vector depends only on its text and a cached one is exactly what a fresh run returns. */
import test from "node:test";
import assert from "node:assert/strict";
import { cachedEmbed } from "../site/core/embed-cache.js";

function countingEmbed() {
  const seen = [];
  const embed = async (texts) => { seen.push([...texts]); return texts.map((t) => [t.length, t.charCodeAt(0)]); };
  return { embed, seen };
}

test("only texts never embedded before reach the embedder, and rows keep the input order", async () => {
  const { embed, seen } = countingEmbed();
  const cached = cachedEmbed(embed);
  assert.deepEqual(await cached(["ab", "c"]), [[2, 97], [1, 99]]);
  assert.deepEqual(await cached(["ab c", "ab", "c", "d"]), [[4, 97], [2, 97], [1, 99], [1, 100]]);
  assert.deepEqual(seen, [["ab", "c"], ["ab c", "d"]]);
});

test("a call of n utterances embeds 2 texts per turn instead of n + 1 (1 on the first: window = utterance)", async () => {
  const { embed, seen } = countingEmbed();
  const cached = cachedEmbed(embed);
  const utterances = [];
  for (let i = 0; i < 10; i += 1) {
    utterances.push(`u${i}`);
    await cached([utterances.join("\n"), ...utterances]);
  }
  assert.deepEqual(seen.map((batch) => batch.length), [1, 2, 2, 2, 2, 2, 2, 2, 2, 2]);
});

test("a repeated text inside one request is embedded once", async () => {
  const { embed, seen } = countingEmbed();
  const cached = cachedEmbed(embed);
  assert.deepEqual(await cached(["да", "да"]), [[2, 1076], [2, 1076]]);
  assert.deepEqual(seen, [["да"]]);
});

test("the cache is bounded: the least recently used text is evicted", async () => {
  const { embed, seen } = countingEmbed();
  const cached = cachedEmbed(embed, { maxEntries: 2 });
  await cached(["a", "b"]);
  await cached(["a"]); // a is now the most recent
  await cached(["c"]); // evicts b
  await cached(["a", "b"]);
  assert.deepEqual(seen, [["a", "b"], ["c"], ["b"]]);
});

test("a failed embed caches nothing and the error reaches the caller", async () => {
  let fail = true;
  const seen = [];
  const cached = cachedEmbed(async (texts) => { seen.push([...texts]); if (fail) throw new Error("worker died"); return texts.map(() => [0]); });
  await assert.rejects(cached(["a"]), /worker died/);
  fail = false;
  assert.deepEqual(await cached(["a"]), [[0]]);
  assert.deepEqual(seen, [["a"], ["a"]]);
});

test("returned rows are copies: a caller mutating one cannot corrupt the cache", async () => {
  const cached = cachedEmbed(async (texts) => texts.map(() => [1, 2]));
  const [row] = await cached(["a"]);
  row[0] = 99;
  assert.deepEqual(await cached(["a"]), [[1, 2]]);
});

test("a request still gets its rows when a concurrent one evicts them while it waits", async () => {
  let release;
  const gate = new Promise((resolve) => { release = resolve; });
  const cached = cachedEmbed(async (texts) => { if (texts.includes("slow")) await gate; return texts.map((t) => [t.length]); }, { maxEntries: 1 });
  await cached(["a"]);
  const first = cached(["a", "slow"]); // "a" is cached when it starts
  await cached(["bb"]); // evicts "a" while the first request waits
  release();
  assert.deepEqual(await first, [[1], [4]]);
});
