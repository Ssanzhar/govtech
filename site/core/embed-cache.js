/* A bounded in-memory cache in front of `embed(texts)` (ADR D78). A live call scores the rolling
   window AND every utterance in it (the evidence spans, score.js) at every turn, so without it
   phrase n costs n+1 e5 runs and a slow phone falls further behind with each phrase. The worker
   embeds texts one at a time, never batched (embed-worker.js), so a vector depends only on its
   text: a cached row is exactly what a fresh run returns. Memory only, never persisted; the
   texts are already in the page's memory as the transcript. */

export const EMBED_CACHE_MAX_ENTRIES = 256; // ~0.8 MB of 768-d rows; a long call has ~100 utterances

/** Wraps `embed`; same signature. Rows come back as copies, in input order. */
export function cachedEmbed(embed, { maxEntries = EMBED_CACHE_MAX_ENTRIES } = {}) {
  const rows = new Map(); // text -> row; insertion order = recency (oldest first)

  const remember = (text, row) => {
    rows.delete(text);
    rows.set(text, row);
    while (rows.size > maxEntries) rows.delete(rows.keys().next().value);
  };

  return async function embedCached(texts) {
    // This request's rows are held here, not looked up later: a concurrent request or this one's
    // own stores may evict a row it still needs.
    const found = new Map(texts.filter((text) => rows.has(text)).map((text) => [text, rows.get(text)]));
    const missing = [...new Set(texts.filter((text) => !found.has(text)))];
    const fresh = missing.length ? await embed(missing) : [];
    missing.forEach((text, i) => found.set(text, fresh[i]));
    for (const [text, row] of found) remember(text, row);
    return texts.map((text) => Array.from(found.get(text)));
  };
}
