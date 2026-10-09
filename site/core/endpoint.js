/* The endpoint pass (ADR D64): every utterance the Vosk vote commits is decoded once more by
   GigaAM-Multilingual, whose text REPLACES the Vosk text -- only the replaced text reaches the
   session and the meter. Per-word kk/ru/en tags come from site/core/word-lid.js.

   `reduceEndpoint` is a pure state machine over:
     {type: "status", status: "loading" | "ready" | "unavailable", reason?}
     {type: "utterance", utterance}   -- a Vosk vote result ({language, text, confidence, start?, end?})
     {type: "decoded", seq, text, decodeMs?}
     {type: "failed", seq, reason}
   and returns `{state, emits, requests}`:
     emits    -- final utterances, strictly in commit order: {seq, text, language, confidence,
                 words: [{word, lang}], source: "gigaam" | "vosk", fallback?, vosk, decodeMs?}
     requests -- {type: "decode", seq, start, end} for the runtime to cut the audio and post.
   Fallback (the shipped per-utterance Vosk vote, clearly marked `source: "vosk"` with a reason)
   whenever GigaAM is not ready ("loading" / "unavailable"), the utterance has no timing
   ("no_timing"), the decode failed or timed out, or GigaAM returned nothing ("empty").
   Inputs are never mutated. */

import { LANGS } from "./word-lid.js";

export const FALLBACK_REASONS = Object.freeze(["loading", "unavailable", "no_timing", "failed", "timeout", "empty"]);

export function initialEndpointState(status = "loading") {
  return { status, nextSeq: 0, queue: [] }; // queue: [{seq, utterance, final | null}]
}

/** The language most of the words carry; ties go to the Vosk vote's language, then LANGS order. */
export function majorityLanguage(words, preferred = null) {
  const counts = Object.fromEntries(LANGS.map((l) => [l, 0]));
  for (const w of words) if (counts[w.lang] !== undefined) counts[w.lang] += 1;
  const best = Math.max(...Object.values(counts));
  if (best === 0) return preferred;
  if (preferred && counts[preferred] === best) return preferred;
  return LANGS.find((l) => counts[l] === best);
}

const voskView = (u) => ({ language: u.language, text: u.text, confidence: u.confidence });

/** The Vosk vote's utterance as the final one (the fallback), with its own words tagged. */
export function fallbackUtterance(utterance, reason, tagText) {
  const words = tagText ? tagText(utterance.text) : [];
  return {
    text: utterance.text,
    language: utterance.language,
    // The meter's step size keeps the recogniser's confidence, as before (ADR D26).
    confidence: utterance.confidence,
    words,
    source: "vosk",
    fallback: reason,
    vosk: voskView(utterance),
  };
}

/** GigaAM's transcript replaces the Vosk text; the meter keeps the vote's confidence (GigaAM's
    CTC decode exposes none), so the meter moves exactly as fast as it did before (FPR-first). */
export function replacedUtterance(utterance, text, tagText, decodeMs = null) {
  const words = tagText(text);
  return {
    text: String(text).trim().replace(/\s+/g, " "),
    language: majorityLanguage(words, utterance.language),
    confidence: utterance.confidence,
    words,
    source: "gigaam",
    vosk: voskView(utterance),
    ...(decodeMs !== null ? { decodeMs } : {}),
  };
}

function flush(queue) {
  const emits = [];
  let i = 0;
  while (i < queue.length && queue[i].final) {
    emits.push({ seq: queue[i].seq, ...queue[i].final });
    i += 1;
  }
  return { queue: queue.slice(i), emits };
}

function resolve(state, seq, final) {
  const queue = state.queue.map((item) => (item.seq === seq && !item.final ? { ...item, final } : item));
  const flushed = flush(queue);
  return { state: { ...state, queue: flushed.queue }, emits: flushed.emits, requests: [] };
}

export function reduceEndpoint(state, event, { tagText } = {}) {
  if (event.type === "status") {
    if (event.status !== "unavailable") return { state: { ...state, status: event.status }, emits: [], requests: [] };
    // GigaAM is gone: everything still waiting falls back, in order.
    const queue = state.queue.map((item) => (item.final ? item : { ...item, final: fallbackUtterance(item.utterance, "unavailable", tagText) }));
    const flushed = flush(queue);
    return { state: { ...state, status: "unavailable", queue: flushed.queue }, emits: flushed.emits, requests: [] };
  }
  if (event.type === "utterance") {
    const seq = state.nextSeq;
    const u = event.utterance;
    const timed = Number.isFinite(u.start) && Number.isFinite(u.end) && u.end > u.start;
    const reason = state.status !== "ready" ? state.status : timed ? null : "no_timing";
    const final = reason ? fallbackUtterance(u, reason, tagText) : null;
    const flushed = flush([...state.queue, { seq, utterance: u, final }]);
    return {
      state: { ...state, nextSeq: seq + 1, queue: flushed.queue },
      emits: flushed.emits,
      requests: final ? [] : [{ type: "decode", seq, start: u.start, end: u.end }],
    };
  }
  if (event.type === "decoded" || event.type === "failed") {
    const item = state.queue.find((q) => q.seq === event.seq && !q.final);
    if (!item) return { state, emits: [], requests: [] }; // late or duplicate answer
    if (event.type === "failed") return resolve(state, event.seq, fallbackUtterance(item.utterance, event.reason || "failed", tagText));
    const text = String(event.text || "").trim();
    const final = text && tagText(text).length
      ? replacedUtterance(item.utterance, text, tagText, event.decodeMs ?? null)
      : fallbackUtterance(item.utterance, "empty", tagText);
    return resolve(state, event.seq, final);
  }
  throw new Error(`unknown endpoint event type: ${event.type}`);
}
