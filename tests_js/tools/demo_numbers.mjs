/* Per-turn numbers of the demo scenes, measured in a real browser with the shipped on-device
   runtime. The landing's analyzer figure (site/main.js) and evidence card (site/index.html) quote
   them; re-run after a model export and update those two places.

     python -m qorgan.api
     QORGAN_E2E_CHANNEL=chrome node tests_js/tools/demo_numbers.mjs http://127.0.0.1:8000 docs/research/2026-11/demo_scene_numbers.json */
import { writeFileSync } from "node:fs";
import { chromium } from "playwright";

const BASE = process.argv[2] || "http://127.0.0.1:8000";
const OUT = process.argv[3];
const browser = await chromium.launch(process.env.QORGAN_E2E_CHANNEL ? { channel: process.env.QORGAN_E2E_CHANNEL } : {});
const page = await browser.newPage();
await page.goto(`${BASE}/live.html`);
const result = await page.evaluate(async () => {
  const { createDeviceRuntime } = await import("/core/device.js");
  const rt = await createDeviceRuntime({ onProgress: () => {} });
  await rt.warmup();
  const { scenarios } = await (await fetch("/core/scenarios.json")).json();
  const out = [];
  for (const s of scenarios) {
    let state = rt.newSession("ru");
    const turns = [];
    for (const line of s.lines) {
      const t0 = performance.now();
      const o = await rt.advance(state, line, 1);
      state = o.state;
      turns.push({ turn: o.update.meter.turn_index, score: +o.update.meter.score.toFixed(2), band: o.update.band, risk: +o.update.result.risk.toFixed(4), latched: o.update.meter.latched, ms: Math.round(performance.now() - t0), line });
    }
    const whole = await rt.analyze(s.lines.join("\n"), "ru");
    out.push({ id: s.id, turns, whole: { risk: +whole.result.risk.toFixed(4), threshold: whole.threshold, tags: whole.result.tags.map((t) => ({ id: t.id, weight: +t.weight.toFixed(3) })), spans: whole.result.attributions.map((a) => a.text) } });
  }
  return out;
});
console.log(JSON.stringify(result, null, 1));
if (OUT) writeFileSync(OUT, JSON.stringify(result, null, 1) + "\n");
await browser.close();
