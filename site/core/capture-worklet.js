/* Microphone capture for the WebView-safe voice mode (ADR D67): an AudioWorkletProcessor that
   batches the first input channel into ~43 ms Float32 chunks (2048 samples at 48 kHz) and posts
   them to its node's port (transferred, not copied). It runs in the audio rendering thread of
   this page; the samples go to voice-light.js and from there only to the GigaAM worker. No
   network primitive exists in an AudioWorklet scope, and none is used here
   (tests/test_architecture.py). Vosklet's own transferer does this job in the full mode. */

/* global AudioWorkletProcessor, registerProcessor */
const BATCH = 2048;

class QorganCapture extends AudioWorkletProcessor {
  constructor() {
    super();
    this.buffer = new Float32Array(BATCH);
    this.filled = 0;
  }

  process(inputs) {
    const channel = inputs[0] && inputs[0][0];
    if (!channel) return true; // no input connected yet (or the source ended): keep the node alive
    for (let i = 0; i < channel.length; ) {
      const n = Math.min(BATCH - this.filled, channel.length - i);
      this.buffer.set(channel.subarray(i, i + n), this.filled);
      this.filled += n;
      i += n;
      if (this.filled === BATCH) {
        const out = this.buffer;
        this.port.postMessage(out, [out.buffer]);
        this.buffer = new Float32Array(BATCH);
        this.filled = 0;
      }
    }
    return true;
  }
}

registerProcessor("qorgan-capture", QorganCapture);
