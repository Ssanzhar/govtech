/* On-device speech recognition for the live page (PLAN_2026-09 B9, ADR D25/D26).

   A 1:1 port of the July server-side design (`src/qorgan/asr/vosk_stream.py`): the same
   audio feeds a Kazakh and a Russian Vosk recognizer in parallel; at an utterance endpoint
   the hypothesis with the higher mean word confidence wins (ties go to the language of the
   last committed utterance). Here the recognizers are Vosklet (Vosk compiled to WASM) —
   nothing leaves the browser; the server never sees audio (ADR D12).

   The prebuilt Vosklet has ONE worker thread per module instance, so each language gets
   its own `loadVosklet()` instance (measured: two instances decode concurrently at a
   combined RTF ≈ 0.07 on a laptop). Vosklet's endpointer fires per recognizer, so
   `reduceAsrEvent` aligns the two streams: the first `result` opens a short window, the
   vote happens when the other language's result arrives or the window closes (using its
   current partial as the hypothesis), and a late duplicate is suppressed.

   Pure functions (`parseVoskResult`, `voteFinal`, `reduceAsrEvent`, `isSupported`) are
   unit-tested in Node; `createDeviceAsr` is the browser runtime. */

// Self-hosted and hash-pinned by deploy_bootstrap (asr/web_models.py): this script sees the
// raw microphone audio, so it is never loaded live from a third party.
export const VOSKLET_SCRIPT_URL = new URL("../vendor/vosklet/Vosklet.js", import.meta.url).href;
export const FULL_CONFIDENCE = 1.0;
// Endpointers differ per model: measured 0.3-0.5 s apart on the same pause. The first
// endpoint waits this long for the other language before voting with its partial instead.
export const ALIGN_WINDOW_MS = 1200;
export const DUPLICATE_SUPPRESS_MS = 1500; // a late `result` from the flushed language is dropped within this
export const TRANSFERER_BUFFER = 128 * 150; // Vosklet's recommended AudioWorklet buffer (~0.4 s at 48 kHz)
const SETTLE_MS = 250; // after teardown, time for the last `result` events to arrive before the flush
// Vosklet stores each model under its own `path` and refetches when the `id` changes; one
// namespace per model, so two languages never collide in the cache.
const storagePathFor = (language, id) => `qorgan-${language}-${id}-v2`; // flat: Vosklet's FS has no nested dirs

/** `(text, mean word confidence)` from a Vosk result payload (string or object). Vosk
    omits `conf` when certain, so missing confidences default to full; clamped to [0, 1]. */
export function parseVoskResult(detail) {
  let payload;
  try {
    payload = typeof detail === "string" ? JSON.parse(detail) : detail;
  } catch {
    return { text: "", confidence: 0 };
  }
  if (!payload || typeof payload !== "object") return { text: "", confidence: 0 };
  const text = String(payload.text ?? payload.partial ?? "").trim();
  if (!text) return { text: "", confidence: 0 };
  const words = Array.isArray(payload.result) ? payload.result : Array.isArray(payload.partial_result) ? payload.partial_result : [];
  const confidences = words.map((w) => (typeof w?.conf === "number" ? w.conf : FULL_CONFIDENCE));
  const mean = confidences.length ? confidences.reduce((a, b) => a + b, 0) / confidences.length : FULL_CONFIDENCE;
  return { text, confidence: Math.min(FULL_CONFIDENCE, Math.max(0, mean)) };
}

export const TIE_EPSILON = 0.02; // confidences this close are a tie (clean speech ties both models)
const KAZAKH_LETTERS = /[әғқңөұүһі]/g;
// Kazakh texts carry >= 12 % Kazakh-only letters, Russian ones ~0 (measured on the corpus);
// the KK model decoding *Russian* speech lands in between, with few of them.
const KAZAKH_SHARE_MIN = 0.08;

/** Share of Kazakh-only letters among the letters of `text`. */
export function kazakhLetterShare(text) {
  const letters = (text.toLowerCase().match(/\p{L}/gu) || []).length;
  if (!letters) return 0;
  return (text.toLowerCase().match(KAZAKH_LETTERS) || []).length / letters;
}

/** The `{language, text, confidence}` with the highest mean word confidence among
    `hypotheses` (`{language: {text, confidence}}`); empty texts always lose. Confidences
    within `TIE_EPSILON` are a tie: for the kk/ru pair the script decides (the RU model
    cannot emit Kazakh letters; a KK hypothesis with almost none means Russian speech),
    otherwise the tie goes to `preferred`. */
export function voteFinal(hypotheses, preferred = null) {
  const candidates = Object.entries(hypotheses)
    .filter(([, hyp]) => hyp && hyp.text)
    .map(([language, hyp]) => ({ language, text: hyp.text, confidence: Math.max(0, hyp.confidence) }))
    .sort((a, b) => b.confidence - a.confidence);
  if (!candidates.length) return null;
  const tied = candidates.filter((c) => candidates[0].confidence - c.confidence <= TIE_EPSILON);
  if (tied.length < 2) return candidates[0];
  const kk = tied.find((c) => c.language === "kk");
  const ru = tied.find((c) => c.language === "ru");
  if (kk && ru) return kazakhLetterShare(kk.text) >= KAZAKH_SHARE_MIN ? kk : ru;
  return tied.find((c) => c.language === preferred) || tied[0];
}

/** Initial reducer state for `languages` (the first is the initial partial preference). */
export function initialAsrState(languages) {
  return {
    languages: [...languages],
    allLanguages: [...languages],
    locked: null,      // the single language left running, once it is settled
    committed: 0,      // utterances voted so far (the lock arms on this)
    wins: {},          // language -> utterances won, the tally the lock reads
    preferred: languages[0],
    pending: {}, // language -> {text, confidence} finals waiting for the vote
    partials: {}, // language -> {text, confidence} latest partials
    windowOpenedAt: null,
    suppressUntil: {}, // language -> timestamp: drop a late duplicate `result`
    lastPartialText: "",
  };
}

/** Pure state machine over recognizer events. `event` is `{type: "result"|"partial"|
    "tick", language?, detail?}`, `now` a timestamp (ms). Returns `{state, emits}` where
    emits are `{type: "partial", language, text}` and `{type: "utterance", language,
    text, confidence}`. Inputs are never mutated. */
export function reduceAsrEvent(state, event, now, { alignWindowMs = ALIGN_WINDOW_MS, suppressMs = DUPLICATE_SUPPRESS_MS, lock = null } = {}) {
  if (event.type === "partial") {
    const hyp = parseVoskResult(event.detail);
    const next = { ...state, partials: { ...state.partials, [event.language]: hyp } };
    const shown = bestPartial(next);
    if (shown.text && shown.text !== state.lastPartialText && state.windowOpenedAt === null) {
      return { state: { ...next, lastPartialText: shown.text }, emits: [{ type: "partial", language: shown.language, text: shown.text }] };
    }
    return { state: next, emits: [] };
  }
  if (event.type === "result") {
    if ((state.suppressUntil[event.language] ?? 0) > now) {
      // A duplicate of an utterance already committed from this language's partial. The
      // entry stays until it expires, so a second duplicate inside the window is dropped too.
      return { state: { ...state, partials: { ...state.partials, [event.language]: null } }, emits: [] };
    }
    if (state.pending[event.language] !== undefined) {
      // This language endpointed twice before the other fired: commit the open window first
      // (voting against the other language's partial), then open a new one -- never overwrite.
      const closed = commit(state, now, suppressMs, lock);
      const reopened = reduceAsrEvent(closed.state, event, now, { alignWindowMs, suppressMs, lock });
      return { state: reopened.state, emits: [...closed.emits, ...reopened.emits] };
    }
    const hyp = parseVoskResult(event.detail);
    const next = {
      ...state,
      pending: { ...state.pending, [event.language]: hyp },
      partials: { ...state.partials, [event.language]: null },
      windowOpenedAt: state.windowOpenedAt ?? now,
    };
    const everyLanguageFired = state.languages.every((l) => next.pending[l] !== undefined);
    return everyLanguageFired ? commit(next, now, suppressMs, lock) : { state: next, emits: [] };
  }
  if (event.type === "tick") {
    const pruned = pruneSuppressions(state, now);
    if (pruned.windowOpenedAt !== null && now - pruned.windowOpenedAt >= alignWindowMs) return commit(pruned, now, suppressMs, lock);
    return { state: pruned, emits: [] };
  }
  throw new Error(`unknown asr event type: ${event.type}`);
}

/** Vote on what is pending; a language that has not fired contributes its current
    partial (like `FinalResult()` at the other's boundary in the Python design) and its
    next `result` is suppressed as a duplicate. */
function commit(state, now, suppressMs, lock = null) {
  const hypotheses = {};
  const suppressUntil = { ...state.suppressUntil };
  for (const language of state.languages) {
    if (state.pending[language] !== undefined) {
      hypotheses[language] = state.pending[language];
    } else {
      hypotheses[language] = state.partials[language] ?? { text: "", confidence: 0 };
      if (hypotheses[language].text) suppressUntil[language] = now + suppressMs;
    }
  }
  const winner = voteFinal(hypotheses, state.preferred);
  const next = applyLock({
    ...state,
    pending: {},
    partials: Object.fromEntries(state.languages.map((l) => [l, null])),
    windowOpenedAt: null,
    suppressUntil,
    lastPartialText: "",
    preferred: winner ? winner.language : state.preferred,
    committed: state.committed + (winner ? 1 : 0),
    wins: winner ? { ...state.wins, [winner.language]: (state.wins[winner.language] ?? 0) + 1 } : state.wins,
  }, winner, lock);
  return { state: next, emits: winner ? [{ type: "utterance", ...winner }] : [] };
}

/** Narrow `languages` to the settled winner, or widen it back when the locked recognizer
    loses confidence -- which is what a speaker switching language looks like. Locking is
    expressed purely as the active language set, so every alignment rule above still holds:
    with one language `everyLanguageFired` is immediate and the vote has one candidate. */
function applyLock(state, winner, lock) {
  if (!lock) return state;
  const { after = 3, confFloor = 0 } = lock;
  if (state.locked) {
    if (winner && winner.confidence < confFloor) {
      return { ...state, locked: null, committed: 0, wins: {}, languages: [...state.allLanguages] };
    }
    return state;
  }
  if (state.committed < after) return state;
  const ranked = Object.entries(state.wins).sort((a, b) => b[1] - a[1] || (a[0] < b[0] ? -1 : 1));
  if (!ranked.length) return state;
  const settled = ranked[0][0];
  return { ...state, locked: settled, languages: [settled] };
}

function pruneSuppressions(state, now) {
  const live = Object.fromEntries(Object.entries(state.suppressUntil).filter(([, until]) => until > now));
  return Object.keys(live).length === Object.keys(state.suppressUntil).length ? state : { ...state, suppressUntil: live };
}

function bestPartial(state) {
  const ordered = [state.preferred, ...state.languages.filter((l) => l !== state.preferred)];
  for (const language of ordered) {
    const hyp = state.partials[language];
    if (hyp && hyp.text) return { language, text: hyp.text };
  }
  return { language: state.preferred, text: "" };
}

/** Can this browser run on-device recognition? `{ok, reasons}`; each reason is
    user-facing. Phones are excluded until the B9 Android bench passes (ADR D25). */
export function isSupported(env = globalThis) {
  const reasons = [];
  if (!env.crossOriginIsolated) reasons.push("this page is not served cross-origin isolated (COOP/COEP headers)");
  if (typeof env.SharedArrayBuffer === "undefined") reasons.push("SharedArrayBuffer is unavailable");
  if (!env.isSecureContext) reasons.push("a secure context (https or localhost) is required");
  if (typeof env.AudioWorkletNode === "undefined") reasons.push("AudioWorklet is unavailable");
  if (!env.navigator?.mediaDevices?.getUserMedia) reasons.push("microphone capture is unavailable");
  if (/Android|iPhone|iPad|Mobile/i.test(env.navigator?.userAgent || "")) reasons.push("phones are not enabled yet (the on-device model has not been measured there)");
  return { ok: reasons.length === 0, reasons };
}

// ── browser runtime ────────────────────────────────────────────────────────────

let scriptPromise = null;
/** Inject the (same-origin, pinned) Vosklet script once. */
export function loadVoskletScript(url = VOSKLET_SCRIPT_URL, doc = globalThis.document) {
  if (typeof globalThis.loadVosklet === "function") return Promise.resolve(globalThis.loadVosklet);
  if (!scriptPromise) {
    scriptPromise = new Promise((resolve, reject) => {
      const tag = doc.createElement("script");
      tag.src = url;
      tag.onload = () => (typeof globalThis.loadVosklet === "function" ? resolve(globalThis.loadVosklet) : reject(new Error("Vosklet did not initialise")));
      tag.onerror = () => reject(new Error("could not load the speech-recognition runtime"));
      doc.head.appendChild(tag);
    });
  }
  return scriptPromise;
}

/** Dual-language on-device ASR. `models` = `{language: {url, id}}` (self-hosted USTAR
    tarballs). Calls `onPartial({language, text})`, `onUtterance({language, text,
    confidence})`, `onStatus(message)`. `start(stream)` takes a microphone MediaStream. */
export async function createDeviceAsr({ models, onPartial, onUtterance, onStatus = () => {}, onError = () => {}, onRaw = null, lock = null }) {
  const languages = Object.keys(models);
  if (!languages.length) throw new Error("no ASR models configured");
  const loadVosklet = await loadVoskletScript();
  onStatus("loading the speech-recognition runtime…");
  const modules = {};
  const loaded = {};
  for (const language of languages) {
    modules[language] = await loadVosklet(); // one module instance (= one worker thread) per language
    onStatus(`loading the ${language.toUpperCase()} speech model… (cached after the first time)`);
    loaded[language] = await modules[language].createModel(models[language].url, storagePathFor(language, models[language].id), models[language].id);
  }

  let ctx = null;
  let recognizers = {};
  let transferer = null;
  let source = null;
  let ticker = null;
  let state = initialAsrState(languages);

  const dispatch = (event) => {
    if (onRaw && event.type !== "tick") onRaw({ ...event, at: performance.now() }); // diagnostics: every recognizer event
    try {
      const out = reduceAsrEvent(state, event, performance.now(), { lock });
      state = out.state;
      for (const emit of out.emits) {
        if (emit.type === "partial") onPartial(emit);
        else onUtterance(emit);
      }
    } catch (err) {
      onError(err);
    }
  };

  return {
    languages,
    async start(stream) {
      ctx = new AudioContext();
      state = initialAsrState(languages);
      for (const language of languages) {
        const rec = await modules[language].createRecognizer(loaded[language], ctx.sampleRate);
        rec.setWords(true);
        rec.setPartialWords(true);
        rec.addEventListener("result", (ev) => dispatch({ type: "result", language, detail: ev.detail }));
        rec.addEventListener("partialResult", (ev) => dispatch({ type: "partial", language, detail: ev.detail }));
        recognizers[language] = rec;
      }
      transferer = await modules[languages[0]].createTransferer(ctx, TRANSFERER_BUFFER);
      transferer.port.onmessage = (ev) => {
        // `state.languages`, not `languages`: once the reducer locks a language the other
        // recognizer stops being fed, which is the whole compute win (ADR D40). Its instance
        // stays loaded, so this saves decode work, not memory.
        for (const language of state.languages) recognizers[language].acceptWaveform(ev.data.slice()); // a copy each: buffers may be transferred
      };
      source = ctx.createMediaStreamSource(stream);
      source.connect(transferer);
      ticker = setInterval(() => dispatch({ type: "tick" }), 100);
      onStatus("listening — audio stays on this device");
    },
    async stop() {
      if (ticker) clearInterval(ticker);
      ticker = null;
      try { source?.disconnect(); } catch {}
      try { transferer?.disconnect(); } catch {}
      // The recognizers finish their queued audio (`delete(true)` waits for the current
      // block) and may still fire `result` events while doing so; those land in `state`
      // through `dispatch` because the listeners stay attached until here.
      for (const rec of Object.values(recognizers)) { try { await rec.delete(true); } catch {} }
      await new Promise((resolve) => setTimeout(resolve, SETTLE_MS));
      recognizers = {};
      // Flush: whatever is pending or still a partial is committed now, so every utterance
      // has been handed to `onUtterance` before `stop()` resolves.
      const flushed = reduceAsrEvent(
        { ...state, windowOpenedAt: performance.now() - ALIGN_WINDOW_MS }, { type: "tick" }, performance.now()
      );
      state = flushed.state;
      for (const emit of flushed.emits) if (emit.type === "utterance") onUtterance(emit);
      if (ctx) { try { await ctx.close(); } catch {} }
      ctx = null;
      onStatus("stopped");
    },
    async dispose() {
      for (const language of languages) { try { await modules[language].cleanUp(); } catch {} }
    },
  };
}
