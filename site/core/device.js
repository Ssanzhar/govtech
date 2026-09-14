/* On-device runtime: everything the pages need to analyse a call without a server
   (PLAN_2026-09 B3/B4). Loads `qorgan-config.json` + `/models/weights.json`, starts the
   embedding worker, and exposes analyze()/session APIs built on the pure core modules.
   The only network traffic is fetching these static files (cached by the service worker). */

import { createWorkerEmbedder } from "./embedder.js";
import { explain } from "./explain.js";
import { createScorer } from "./score.js";
import { advance, initialSession, transcriptOf } from "./session.js";
import { buildReport, summarize } from "./summary.js";

const CONFIG_URL = new URL("./qorgan-config.json", import.meta.url);
const WEIGHTS_URL = new URL("../models/weights.json", import.meta.url);

export async function createDeviceRuntime({ onProgress } = {}) {
  const [config, weights] = await Promise.all([fetchJson(CONFIG_URL), fetchJson(WEIGHTS_URL)]);
  const embedder = createWorkerEmbedder({ onProgress });
  const score = createScorer({ weights, config, embed: embedder.embed });

  return {
    config,
    weights,
    /** Download/compile the embedder (once; cached afterwards). Resolves when ready. */
    warmup: (device) => embedder.warmup(device),
    /** One-shot verdict with a grounded, localized explanation. */
    async analyze(transcript, locale = config.default_locale) {
      const result = await score(transcript);
      return { result, explanation: explain(result, transcript, locale, config), threshold: config.scoring.risk_threshold };
    },
    /** Live-call session: `advance` returns `{state, update}` for one committed utterance. */
    newSession: (locale = config.default_locale) => initialSession(locale, config),
    advance: (state, text, confidence = 1) => advance(state, { text, confidence }, { score, config }),
    transcriptOf: (state) => transcriptOf(state, config),
    summarize: (state) => summarize(state, config),
    buildReport: (state, opts) => buildReport(state, config, opts),
    dispose: () => embedder.terminate(),
  };
}

async function fetchJson(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`failed to load ${url}: ${res.status}`);
  return res.json();
}
