/* Public surface of the on-device core. Everything here is a pure port of the Python
   modules named in each file; `tests_js/` proves parity on golden fixtures. */
export { createScorer, mergeCueEvidence } from "./score.js";
export { explain } from "./explain.js";
export { recommend } from "./recommend.js";
export { band, initialMeter, updateMeter } from "./meter.js";
export { advance, initialSession, rollingWindow, transcriptOf } from "./session.js";
export { buildReport, summarize } from "./summary.js";
export { createDeviceRuntime } from "./device.js";
