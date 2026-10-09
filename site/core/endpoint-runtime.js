/* Browser wiring of the endpoint pass (ADR D64): keeps the recent microphone audio, sends each
   committed utterance's audio to GigaAM, and hands the final utterances (GigaAM's text, or the
   Vosk vote as a marked fallback) to the page in order. The decisions live in the pure
   `reduceEndpoint` (endpoint.js); this file only does I/O between it, the audio ring and the
   worker client. No network primitive. */

import { createAudioRing } from "./audio-ring.js";
import { initialEndpointState, reduceEndpoint } from "./endpoint.js";
import { createGigaamClient } from "./gigaam.js";

// Context around the recogniser's word span: Vosk word times are tight, and the spike decoded
// whole utterances with ~0.3 s of lead-in.
export const AUDIO_PAD_S = 0.3;

/** `endpoint` = `asr.endpoint` from qorgan-config.json (null = off). `tagText` = the word-LID's
    `tagText`. `onUtterance(final, {latencyMs})` gets every utterance once, in commit order;
    `onStatus({status, reason?, loadMs?, fromCache?, loaded?, total?})` reports loading
    progress, readiness, or unavailability (then every utterance falls back to the Vosk vote). */
export function createEndpointPass({ endpoint, tagText, pageUrl = globalThis.location?.href, onUtterance, onStatus = () => {}, createClient = createGigaamClient, now = () => performance.now() }) {
  const usable = Boolean(endpoint && tagText);
  let state = initialEndpointState(usable ? "loading" : "unavailable");
  let client = null;
  let ring = null;
  const committedAt = new Map(); // seq -> time the Vosk vote committed it
  const inFlight = new Set();
  let seq = 0;

  const feed = (event) => {
    const out = reduceEndpoint(state, event, { tagText });
    state = out.state;
    for (const final of out.emits) {
      const started = committedAt.get(final.seq);
      committedAt.delete(final.seq);
      onUtterance(final, { latencyMs: started === undefined ? null : now() - started });
    }
    for (const request of out.requests) decode(request);
  };

  const decode = ({ seq: id, start, end }) => {
    const samples = ring ? ring.slice(start - AUDIO_PAD_S, end + AUDIO_PAD_S) : new Float32Array(0);
    if (!samples.length || !client) {
      feed({ type: "failed", seq: id, reason: "failed" });
      return;
    }
    const job = client.decode(samples, ring.sampleRate).then(
      (r) => feed({ type: "decoded", seq: id, text: r.text, decodeMs: r.decodeMs }),
      (err) => feed({ type: "failed", seq: id, reason: err?.reason || "failed" })
    );
    inFlight.add(job);
    job.finally(() => inFlight.delete(job));
  };

  if (usable) {
    try {
      client = createClient({ endpoint, pageUrl, onProgress: (p) => onStatus({ status: "loading", ...p }) });
      onStatus({ status: "loading" });
      client.ready.then(
        (info) => { feed({ type: "status", status: "ready" }); onStatus({ status: "ready", loadMs: info.loadMs, fromCache: info.fromCache }); },
        (err) => {
          try { client?.dispose(); } catch {}
          client = null;
          feed({ type: "status", status: "unavailable" });
          onStatus({ status: "unavailable", reason: String(err?.message || err) });
        }
      );
    } catch (err) {
      client = null;
      state = initialEndpointState("unavailable");
      onStatus({ status: "unavailable", reason: String(err?.message || err) });
    }
  } else if (endpoint) {
    onStatus({ status: "unavailable", reason: "the word-level model is missing" });
  }

  return {
    get status() { return state.status; },
    /** A new stream: audio time restarts at 0 (the recognisers are recreated too). */
    reset(sampleRate) { ring = usable ? createAudioRing(sampleRate) : null; },
    push(chunk) { ring?.push(chunk); },
    utterance(voskUtterance) {
      const { type, ...utterance } = voskUtterance;
      committedAt.set(seq, now());
      seq += 1;
      feed({ type: "utterance", utterance });
    },
    /** Resolves once every committed utterance has been delivered (decodes have timeouts). */
    async drain() {
      while (inFlight.size) await Promise.allSettled([...inFlight]);
    },
    dispose() { try { client?.dispose(); } catch {} client = null; },
  };
}
