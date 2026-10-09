/* The WebView-safe voice mode (ADR D67): GigaAM only, no Vosklet.

   Android WebView (eGov Mobile's included) never has cross-origin isolation, so Vosklet's WASM
   threads cannot start and the D64 mode dies there. GigaAM's single-threaded sherpa-onnx build
   needs no isolation. This mode keeps the D64 endpoint pass exactly as it is and replaces only
   what fed it phrase boundaries: Vosk's endpointer becomes the energy VAD (vad.js).

     microphone -> AudioWorklet (capture-worklet.js) -> audio ring + VAD
     VAD segment -> createEndpointPass (the D64 path: cut the padded span, GigaAM decodes it,
                    the word LID tags it) -> onUtterance -> the page's session and meter

   There are no live words while the person speaks (no streaming recogniser), and nothing to
   fall back to: a phrase whose decode fails, times out or comes back empty is dropped and
   reported (`onDropped`), never scored as empty text. The audio stays on the device: it goes
   from the worklet to this module to the GigaAM worker by postMessage, and no module on that
   path has a network primitive (tests/test_architecture.py). */

import { createEndpointPass } from "./endpoint-runtime.js";
import { createGigaamClient } from "./gigaam.js";
import { createVad } from "./vad.js";
import { createWordLid } from "./word-lid.js";

export const CAPTURE_WORKLET_URL = new URL("./capture-worklet.js", import.meta.url).href;
export const CAPTURE_PROCESSOR = "qorgan-capture";
// GigaAM's CTC decode exposes no confidence. The meter then steps exactly as it does for a
// pasted or replayed transcript (session.js defaults to 1), never faster.
export const LIGHT_CONFIDENCE = 1;

const MISSING = "the word-level model is missing";

/** VAD + endpoint pass, without any browser audio API (unit-tested in Node).
    `ready` resolves once GigaAM is loaded and rejects if it cannot be (or is not configured);
    `reset(sampleRate)` starts a stream, `push(chunk)` feeds it, `stop()` flushes the open phrase
    and resolves once every phrase has been delivered or dropped. */
export function createLightPass({ endpoint, tagText, pageUrl = globalThis.location?.href, onUtterance, onDropped = () => {}, onSpeech = () => {}, onStatus = () => {}, createClient = createGigaamClient, now = () => performance.now() }) {
  let resolveReady;
  let rejectReady;
  const ready = new Promise((resolve, reject) => { resolveReady = resolve; rejectReady = reject; });
  ready.catch(() => {}); // the caller awaits it; this only keeps an unobserved copy quiet
  const segments = [];
  let vad = null;

  if (!endpoint || !tagText) rejectReady(new Error(MISSING));
  const pass = endpoint && tagText
    ? createEndpointPass({
      endpoint,
      tagText,
      pageUrl,
      createClient,
      now,
      onUtterance: (final, info) => {
        if (final.source === "gigaam") onUtterance(final, info);
        else onDropped({ seq: final.seq, reason: final.fallback || "failed", latencyMs: info.latencyMs });
      },
      onStatus: (status) => {
        onStatus(status);
        if (status.status === "ready") resolveReady(status);
        else if (status.status === "unavailable") rejectReady(new Error(status.reason || "GigaAM is unavailable"));
      },
    })
    : null;

  const handle = (events) => {
    for (const event of events) {
      onSpeech(event);
      if (event.type !== "segment") continue;
      segments.push({ start: event.start, end: event.end, cut: event.cut, streamAt: event.at, wallAt: now() });
      pass.utterance({ type: "utterance", text: "", language: null, confidence: LIGHT_CONFIDENCE, start: event.start, end: event.end });
    }
  };

  return {
    ready,
    /** Every segment the VAD closed this stream: `{start, end, cut, streamAt, wallAt}`. */
    get segments() { return segments; },
    reset(sampleRate) {
      vad = createVad(sampleRate);
      segments.length = 0;
      pass?.reset(sampleRate);
    },
    push(chunk) {
      if (!vad || !pass) return;
      pass.push(chunk);
      handle(vad.push(chunk));
    },
    async stop() {
      if (vad && pass) handle(vad.flush());
      await pass?.drain();
    },
    dispose() { pass?.dispose(); },
  };
}

/** The browser runtime: same shape as `createDeviceAsr` (asr.js) so the page drives both alike.
    `wordLevel` = site/core/asr-wordlevel.json (the word LID tags GigaAM's words), `endpoint` =
    `asr.endpoint` of qorgan-config.json. `onStatus(message, {code})`: listening | stopped. */
export async function createLightAsr({ endpoint, wordLevel, onUtterance, onDropped, onSpeech, onStatus = () => {}, onEndpointStatus = () => {}, createClient = undefined }) {
  const lid = wordLevel ? createWordLid(wordLevel.lid) : null;
  const pass = createLightPass({
    endpoint,
    tagText: lid?.tagText,
    onUtterance,
    onDropped,
    onSpeech,
    onStatus: onEndpointStatus,
    ...(createClient ? { createClient } : {}),
  });
  try {
    await pass.ready;
  } catch (err) {
    pass.dispose();
    throw err;
  }

  let ctx = null;
  let node = null;
  let source = null;

  return {
    languages: lid ? ["kk", "ru", "en"] : [],
    get segments() { return pass.segments; },
    async start(stream) {
      ctx = new AudioContext();
      await ctx.audioWorklet.addModule(CAPTURE_WORKLET_URL);
      node = new AudioWorkletNode(ctx, CAPTURE_PROCESSOR, { numberOfInputs: 1, numberOfOutputs: 0, channelCount: 1 });
      pass.reset(ctx.sampleRate); // VAD times and the audio ring both count from here
      node.port.onmessage = (event) => pass.push(event.data);
      source = ctx.createMediaStreamSource(stream);
      source.connect(node);
      onStatus("listening — audio stays on this device", { code: "listening" });
    },
    async stop() {
      try { source?.disconnect(); } catch {}
      if (node) node.port.onmessage = null; // the worklet's last partial batch (< 50 ms) is not needed
      source = null;
      node = null;
      await pass.stop();
      if (ctx) { try { await ctx.close(); } catch {} }
      ctx = null;
      onStatus("stopped", { code: "stopped" });
    },
    async dispose() { pass.dispose(); },
  };
}
