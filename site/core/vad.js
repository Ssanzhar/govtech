/* The WebView-safe voice mode's segmenter (ADR D67): a deterministic energy + zero-crossing
   voice-activity detector that cuts the microphone audio into phrases for GigaAM.

   Why not Vosk's endpointer: inside an Android WebView there is no cross-origin isolation, so
   Vosklet (WASM threads + SharedArrayBuffer) cannot start, and the phrase boundaries it gave
   the D64 endpoint pass are gone. This file replaces exactly that one job.

   Per 30 ms frame: the RMS level in dBFS and the zero crossings per second (rate-independent).
   A frame is ACTIVE when its level clears the adaptive noise floor by a margin (larger to start
   a phrase than to stay in one) and the absolute minimum, and it crosses zero often enough to
   not be mains hum. Inside a phrase, a quiet frame with a very high crossing rate (a fricative:
   "с", "ш", "ф") also counts. The noise floor is the minimum level over the last 2.5 s, capped
   so that a steady loud signal is never mistaken for background.

   A phrase opens after `onsetMs` of consecutive active frames (`speech_start`) and closes after
   `hangoverMs` of inactive ones (`segment`, with `cut: "pause"`); past `softMaxMs` a shorter
   pause closes it, and at `maxMs` it is cut (`cut: "max"`) and continues as the next phrase.
   Phrases shorter than `minSpeechMs` are dropped (`speech_end`, `dropped: true`). Segment
   times are stream seconds from the first sample pushed, as the audio ring counts them.

   Pure: `frameFeatures`, `reduceVad` and `flushVad` never mutate their inputs; `createVad` only
   buffers samples into frames. No network primitive (tests/test_architecture.py). */

export const VAD_DEFAULTS = Object.freeze({
  frameMs: 30,
  floorDb: -100,           // the level reported for digital silence
  absMinDb: -50,           // never speech below this, whatever the floor
  startMarginDb: 12,       // above the floor, to open a phrase
  stayMarginDb: 6,         // above the floor, to stay in one
  floorWindowMs: 2500,     // the noise floor is the minimum level over this window...
  floorCapDb: -35,         // ...but never above this (a steady loud signal is not background)
  minZcrPerS: 150,         // 50/60 Hz hum crosses zero 100/120 times a second
  fricativeZcrPerS: 2500,  // inside a phrase, a quiet frame this "hissy" still counts
  onsetMs: 150,
  hangoverMs: 600,
  softMaxMs: 8000,
  softHangoverMs: 250,
  maxMs: 15000,
  minSpeechMs: 300,
  preRollMs: 150,
});

/** `{db, zcrPerS}` of one frame of samples at `sampleRate`. */
export function frameFeatures(frame, sampleRate, floorDb = VAD_DEFAULTS.floorDb) {
  let energy = 0;
  let crossings = 0;
  for (let i = 0; i < frame.length; i++) {
    energy += frame[i] * frame[i];
    if (i > 0 && frame[i - 1] * frame[i] < 0) crossings += 1;
  }
  const rms = frame.length ? Math.sqrt(energy / frame.length) : 0;
  const db = rms > 0 ? Math.max(floorDb, 20 * Math.log10(rms)) : floorDb;
  return { db, zcrPerS: frame.length ? (crossings * sampleRate) / frame.length : 0 };
}

export function initialVadState(options = VAD_DEFAULTS) {
  return {
    options: { ...VAD_DEFAULTS, ...options },
    frames: 0,           // frames consumed so far
    window: [],          // recent frame levels, for the noise floor
    phase: "silence",    // silence | onset | speech
    onsetStart: null,
    onsetFrames: 0,
    segStart: null,      // where the open phrase starts (with pre-roll)
    speechStart: null,   // where its first active frame starts (for the minimum length)
    lastActiveEnd: null,
    silenceS: 0,         // inactive time since lastActiveEnd
    prevEnd: 0,          // end of the last emitted segment (pre-roll never reaches back past it)
  };
}

const ms = (x) => x / 1000;

function isActive(features, floor, inSpeech, o) {
  if (features.zcrPerS < o.minZcrPerS) return false;
  const margin = inSpeech ? o.stayMarginDb : o.startMarginDb;
  if (features.db >= Math.max(o.absMinDb, floor + margin)) return true;
  return inSpeech && features.zcrPerS >= o.fricativeZcrPerS && features.db >= Math.max(o.absMinDb, floor + o.stayMarginDb / 2);
}

/** Close the open phrase at `lastActiveEnd`: a segment, or a drop when it is too short. */
function close(state, at, cut) {
  const o = state.options;
  const base = { ...state, phase: "silence", onsetStart: null, onsetFrames: 0, segStart: null, speechStart: null, silenceS: 0 };
  if (state.lastActiveEnd - state.speechStart < ms(o.minSpeechMs)) {
    return { state: { ...base, lastActiveEnd: null }, events: [{ type: "speech_end", dropped: true, start: state.segStart, end: state.lastActiveEnd, at }] };
  }
  return {
    state: { ...base, prevEnd: state.lastActiveEnd, lastActiveEnd: null },
    events: [{ type: "segment", start: state.segStart, end: state.lastActiveEnd, cut, at }],
  };
}

/** One frame (`features`, lasting `frameS` seconds) -> `{state, events}`. */
export function reduceVad(state, features, frameS) {
  const o = state.options;
  const t0 = state.frames * frameS;
  const t1 = (state.frames + 1) * frameS;
  const keep = Math.max(1, Math.round(o.floorWindowMs / o.frameMs));
  const window = [...state.window.slice(-(keep - 1)), features.db];
  const floor = Math.min(o.floorCapDb, Math.min(...window));
  const next = { ...state, frames: state.frames + 1, window };
  const active = isActive(features, floor, state.phase === "speech", o);

  if (state.phase === "silence") {
    if (!active) return { state: next, events: [] };
    return reduceOnset({ ...next, phase: "onset", onsetStart: t0, onsetFrames: 1 }, t1);
  }
  if (state.phase === "onset") {
    if (!active) return { state: { ...next, phase: "silence", onsetStart: null, onsetFrames: 0 }, events: [] };
    return reduceOnset({ ...next, onsetFrames: state.onsetFrames + 1 }, t1);
  }
  // speech
  if (active) {
    const open = { ...next, lastActiveEnd: t1, silenceS: 0 };
    if (t1 - state.segStart >= ms(o.maxMs) - 1e-9) {
      const cut = close(open, t1, "max");
      // The phrase goes on: the next segment starts exactly where this one was cut.
      return { state: { ...cut.state, phase: "speech", segStart: t1, speechStart: t1, lastActiveEnd: t1 }, events: cut.events };
    }
    return { state: open, events: [] };
  }
  const silenceS = state.silenceS + frameS;
  const hangover = state.lastActiveEnd - state.segStart >= ms(o.softMaxMs) ? o.softHangoverMs : o.hangoverMs;
  if (silenceS >= ms(hangover) - 1e-9) return close({ ...next, silenceS }, t1, "pause");
  return { state: { ...next, silenceS }, events: [] };
}

function reduceOnset(state, t1) {
  const o = state.options;
  if (state.onsetFrames * o.frameMs < o.onsetMs) return { state, events: [] };
  const segStart = Math.max(state.prevEnd, state.onsetStart - ms(o.preRollMs), 0);
  return {
    state: { ...state, phase: "speech", segStart, speechStart: state.onsetStart, lastActiveEnd: t1, silenceS: 0, onsetStart: null, onsetFrames: 0 },
    events: [{ type: "speech_start", start: segStart, at: t1 }],
  };
}

/** The microphone stopped: close a phrase that is still open (`cut: "flush"`). */
export function flushVad(state) {
  if (state.phase !== "speech") return { state: { ...state, phase: "silence", onsetStart: null, onsetFrames: 0 }, events: [] };
  const at = state.lastActiveEnd + state.silenceS;
  return close(state, at, "flush");
}

/** Streaming wrapper: `push(chunk)` -> events, `flush()` -> events. Frames are cut from the
    concatenated stream, so the events do not depend on the chunk sizes. */
export function createVad(sampleRate, options = {}) {
  if (!(sampleRate > 0)) throw new RangeError(`bad sample rate: ${sampleRate}`);
  let state = initialVadState({ ...VAD_DEFAULTS, ...options });
  const frameLen = Math.round((sampleRate * state.options.frameMs) / 1000);
  const frameS = frameLen / sampleRate;
  const frame = new Float32Array(frameLen);
  let filled = 0;
  return {
    get state() { return state; },
    push(chunk) {
      const events = [];
      for (let i = 0; i < chunk.length; ) {
        const n = Math.min(frameLen - filled, chunk.length - i);
        frame.set(chunk.subarray(i, i + n), filled);
        filled += n;
        i += n;
        if (filled === frameLen) {
          const out = reduceVad(state, frameFeatures(frame, sampleRate, state.options.floorDb), frameS);
          state = out.state;
          events.push(...out.events);
          filled = 0;
        }
      }
      return events;
    },
    flush() {
      const out = flushVad(state);
      state = out.state;
      filled = 0;
      return out.events;
    },
  };
}
