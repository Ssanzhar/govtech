/* The last N seconds of microphone audio, addressable by stream time (ADR D63).

   The endpoint pass re-decodes each committed utterance with GigaAM, so the runtime keeps the
   PCM the recognisers were fed. Vosk word times count from the first sample fed; so does this
   ring, so `slice(start, end)` returns exactly the audio a Vosk span refers to. Memory stays
   bounded (60 s at 48 kHz = 11.5 MB of Float32). The audio never leaves the device. */

export const DEFAULT_RING_SECONDS = 60;

export function createAudioRing(sampleRate, maxSeconds = DEFAULT_RING_SECONDS) {
  if (!(sampleRate > 0)) throw new RangeError(`bad sample rate: ${sampleRate}`);
  const capacity = Math.ceil(sampleRate * maxSeconds);
  const buffer = new Float32Array(capacity);
  let written = 0; // samples pushed since the start of the stream

  return {
    sampleRate,
    /** Samples pushed so far, as stream seconds. */
    get seconds() { return written / sampleRate; },
    push(chunk) {
      const data = chunk.length > capacity ? chunk.subarray(chunk.length - capacity) : chunk;
      const skipped = chunk.length - data.length;
      written += skipped;
      for (let i = 0; i < data.length; ) {
        const at = (written + i) % capacity;
        const n = Math.min(data.length - i, capacity - at);
        buffer.set(data.subarray(i, i + n), at);
        i += n;
      }
      written += data.length;
    },
    /** A copy of [startS, endS) clipped to what is still held; empty if nothing is. */
    slice(startS, endS) {
      const oldest = Math.max(0, written - capacity);
      const from = Math.max(oldest, Math.floor(Math.max(0, startS) * sampleRate));
      const to = Math.min(written, Math.ceil(endS * sampleRate));
      if (to <= from) return new Float32Array(0);
      const out = new Float32Array(to - from);
      for (let i = 0; i < out.length; ) {
        const at = (from + i) % capacity;
        const n = Math.min(out.length - i, capacity - at);
        out.set(buffer.subarray(at, at + n), i);
        i += n;
      }
      return out;
    },
  };
}
