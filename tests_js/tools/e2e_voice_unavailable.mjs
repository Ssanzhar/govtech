/* ADR D67: without cross-origin isolation AND without GigaAM, the voice mode turns itself off
   with a clear reason, and replay still scores a call. Against a non-isolated server:

     node tests_js/tools/static_site.mjs --isolation off --port 8410 &
     node tests_js/tools/e2e_voice_unavailable.mjs --base http://127.0.0.1:8410 --out DIR

   The GigaAM model request is answered 404 (as on a deployment without the model). Writes
   DIR/unavailable.json and two screenshots (light-mode setup, voice mode off). */
import { mkdirSync, writeFileSync } from "node:fs";
import { join, resolve } from "node:path";
import { chromium } from "playwright";

const args = process.argv.slice(2);
const opt = (name, fallback) => { const i = args.indexOf(name); return i > -1 ? args[i + 1] : fallback; };
const BASE = opt("--base", "http://127.0.0.1:8410");
const OUT = resolve(opt("--out", "e2e-voice-unavailable"));
mkdirSync(OUT, { recursive: true });

const browser = await chromium.launch({ args: ["--use-fake-ui-for-media-stream", "--use-fake-device-for-media-stream"] });
const context = await browser.newContext({ viewport: { width: 900, height: 1300 } });
await context.grantPermissions(["microphone"], { origin: BASE });
const page = await context.newPage();
await page.route("**/models/gigaam/**", (route) => route.fulfill({ status: 404, body: "" }));
await page.goto(`${BASE}/live.html`);
await page.locator('label:has(input[name="lv-mode"][value="mic"])').click();
const before = await page.evaluate(() => ({
  isolated: globalThis.crossOriginIsolated === true,
  micNote: document.querySelector("#lvMicNote").dataset.i18n,
  modeNote: document.querySelector("#lvVoiceModeNote").hidden ? null : document.querySelector("#lvVoiceModeNote").dataset.i18n,
  speechDownload: document.querySelector("#lvDlMicSpeech").dataset.i18n,
}));
await page.locator("#lvSetupMic").screenshot({ path: join(OUT, "light-setup.png") });
await page.click("#lvMicStart");
await page.waitForFunction(() => document.querySelector("#lvMicNote")?.dataset.i18n === "mic.light_unavailable", null, { timeout: 300000 });
const after = await page.evaluate(() => ({
  micNote: document.querySelector("#lvMicNote").dataset.i18n,
  micNoteText: document.querySelector("#lvMicNote").textContent,
  micRadioDisabled: document.querySelector('input[name="lv-mode"][value="mic"]').disabled,
}));
await page.locator("#lvSetupMic").screenshot({ path: join(OUT, "voice-off.png") });
// Replay still works.
await page.locator('label:has(input[name="lv-mode"][value="replay"])').click();
await page.click("#lvStart");
await page.waitForFunction(() => !document.querySelector("#lvSummaryWrap").hidden, null, { timeout: 300000 });
const replayScore = await page.evaluate(() => document.querySelector(".lv-summary-score")?.textContent ?? null);
await browser.close();
const result = { base: BASE, before, after, replay_summary_score: replayScore };
writeFileSync(join(OUT, "unavailable.json"), JSON.stringify(result, null, 1));
console.log(JSON.stringify(result));
