/* Qorğan microphone capture — getUserMedia → downsampled 16 kHz PCM16 mono →
   WS /api/live/ws. Audio goes only to the origin server; the server answers with
   partial / utterance / summary events (see qorgan.api_live_ws). */
window.QorganMic = (() => {
  "use strict";

  let ws = null;
  let ctx = null;
  let source = null;
  let node = null;
  let stream = null;
  let targetRate = 16000;
  let stopping = false;

  const supported = () =>
    Boolean(
      navigator.mediaDevices?.getUserMedia &&
        (window.AudioContext || window.webkitAudioContext) &&
        window.WebSocket
    );

  // Average-pool the float samples down to the server's rate, then clamp to int16.
  const downsampleToPcm16 = (input, fromRate, toRate) => {
    const ratio = fromRate / toRate;
    const length = Math.floor(input.length / ratio);
    const out = new Int16Array(length);
    for (let i = 0; i < length; i++) {
      const start = Math.floor(i * ratio);
      const end = Math.min(Math.floor((i + 1) * ratio), input.length);
      let sum = 0;
      for (let j = start; j < end; j++) sum += input[j];
      const sample = Math.max(-1, Math.min(1, sum / Math.max(1, end - start)));
      out[i] = sample < 0 ? sample * 0x8000 : sample * 0x7fff;
    }
    return out.buffer;
  };

  const startAudio = async () => {
    stream = await navigator.mediaDevices.getUserMedia({
      audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true },
    });
    const Ctx = window.AudioContext || window.webkitAudioContext;
    ctx = new Ctx();
    source = ctx.createMediaStreamSource(stream);
    node = ctx.createScriptProcessor(4096, 1, 1);
    node.onaudioprocess = (ev) => {
      if (!ws || ws.readyState !== WebSocket.OPEN) return;
      ws.send(downsampleToPcm16(ev.inputBuffer.getChannelData(0), ctx.sampleRate, targetRate));
    };
    source.connect(node);
    node.connect(ctx.destination); // keeps the processor alive; output stays silent
  };

  const teardownAudio = () => {
    if (node) node.disconnect();
    if (source) source.disconnect();
    if (stream) stream.getTracks().forEach((track) => track.stop());
    if (ctx && ctx.state !== "closed") ctx.close();
    node = source = stream = ctx = null;
  };

  // start({locale, onEvent}) — onEvent receives every server frame plus synthetic
  // {type:"error"} frames for local capture/connection failures.
  const start = ({ locale, onEvent }) => {
    stopping = false;
    const scheme = location.protocol === "https:" ? "wss" : "ws";
    ws = new WebSocket(`${scheme}://${location.host}/api/live/ws`);
    ws.binaryType = "arraybuffer";

    ws.addEventListener("open", () => {
      ws.send(JSON.stringify({ type: "start", locale, backend: null }));
    });
    ws.addEventListener("error", () => {
      onEvent({ type: "error", message: "connection to /api/live/ws failed" });
      teardownAudio();
    });
    ws.addEventListener("close", () => {
      if (!stopping) teardownAudio();
    });
    ws.addEventListener("message", async (event) => {
      let msg;
      try {
        msg = JSON.parse(event.data);
      } catch {
        return;
      }
      if (msg.type === "ready") {
        targetRate = msg.sample_rate || targetRate;
        try {
          await startAudio();
        } catch (err) {
          onEvent({ type: "error", message: `microphone unavailable — ${err.message || err}` });
          stop();
          return;
        }
      }
      onEvent(msg);
      if (msg.type === "summary" || msg.type === "error") {
        stopping = true;
        teardownAudio();
        if (ws && ws.readyState === WebSocket.OPEN) ws.close();
        ws = null;
      }
    });
  };

  // User clicked "End call": stop capturing, ask the server to flush; the websocket
  // stays open so the final utterance(s) + summary still arrive.
  const stop = () => {
    stopping = true;
    teardownAudio();
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ type: "stop" }));
    }
  };

  return { supported, start, stop };
})();
