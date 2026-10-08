/* The word-level voice mode end to end (ADR D63): the real live page in Chromium, a fake
   microphone playing WAV files (`--use-file-for-fake-audio-capture`), against a running server.

     python -m qorgan.api   # serves site/ with COOP/COEP
     node tests_js/tools/e2e_voice_wordlevel.mjs --base http://127.0.0.1:8000 --out DIR \
          --variant after|before --scene NAME=a.wav,b.wav [--scene ...] [--gap 1.2] [--lead 5]

   Per scene: the WAVs are joined (16-bit mono PCM, one sample rate) with `--gap` s of silence
   and `--lead` s before the first, played once through the microphone; the page runs with
   `?debug=asr`, which keeps a per-utterance trace (text, source, fallback, end-to-end latency
   from the Vosk endpoint to the replaced text, GigaAM decode time, meter). Writes
   DIR/<variant>-<scene>.json, a screenshot of the live per-word chips mid-call and one of
   the final transcript. `--variant before` serves the shipped D62 configuration instead
   (per-utterance Vosk vote, English handicap 0.15, no word-level models, no GigaAM). */
import { execFileSync } from "node:child_process";
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { join, resolve } from "node:path";
import { chromium } from "playwright";
import { readWav } from "./gigaam_browser_bench.mjs";

const args = process.argv.slice(2);
const opt = (name, fallback) => { const i = args.indexOf(name); return i > -1 ? args[i + 1] : fallback; };
const BASE = opt("--base", "http://127.0.0.1:8000");
const OUT = resolve(opt("--out", "e2e-voice"));
const VARIANT = opt("--variant", "after");
const GAP_S = Number(opt("--gap", "1.2"));
const LEAD_S = Number(opt("--lead", "5"));
const TAIL_S = 6;
const scenes = args.flatMap((a, i) => (a === "--scene" ? [args[i + 1]] : [])).map((s) => {
  const [name, files] = s.split("=");
  return { name, files: files.split(",") };
});
if (!scenes.length) throw new Error("give at least one --scene NAME=a.wav,b.wav");
mkdirSync(OUT, { recursive: true });

/** Join 16-bit mono WAVs (same rate) with silence; returns {path, seconds, offsets}. */
function joinWavs(files, path) {
  const parts = files.map((f) => ({ f, ...readWav(f) }));
  const rate = parts[0].sampleRate;
  if (parts.some((p) => p.sampleRate !== rate)) throw new Error("all WAVs of a scene need one sample rate");
  const silence = (s) => new Array(Math.round(s * rate)).fill(0);
  let samples = silence(LEAD_S);
  const offsets = [];
  for (const p of parts) {
    offsets.push({ file: p.f, start: samples.length / rate, seconds: p.samples.length / rate });
    samples = samples.concat(p.samples, silence(GAP_S));
  }
  const data = Buffer.alloc(samples.length * 2);
  samples.forEach((v, i) => data.writeInt16LE(Math.max(-32768, Math.min(32767, Math.round(v * 32768))), i * 2));
  const header = Buffer.alloc(44);
  header.write("RIFF", 0); header.writeUInt32LE(36 + data.length, 4); header.write("WAVE", 8);
  header.write("fmt ", 12); header.writeUInt32LE(16, 16); header.writeUInt16LE(1, 20); header.writeUInt16LE(1, 22);
  header.writeUInt32LE(rate, 24); header.writeUInt32LE(rate * 2, 28); header.writeUInt16LE(2, 32); header.writeUInt16LE(16, 34);
  header.write("data", 36); header.writeUInt32LE(data.length, 40);
  writeFileSync(path, Buffer.concat([header, data]));
  return { path, seconds: samples.length / rate, offsets };
}

function treeRssMb(rootPid) {
  const rows = execFileSync("ps", ["-A", "-o", "pid=,ppid=,rss="], { encoding: "utf8" }).trim().split("\n").map((l) => l.trim().split(/\s+/).map(Number));
  const children = new Map();
  for (const [pid, ppid] of rows) children.set(ppid, [...(children.get(ppid) || []), pid]);
  const rss = new Map(rows.map(([pid, , kb]) => [pid, kb]));
  let total = 0;
  const stack = [rootPid];
  while (stack.length) { const pid = stack.pop(); total += rss.get(pid) || 0; stack.push(...(children.get(pid) || [])); }
  return Math.round(total / 1024);
}

/** The shipped D62 voice mode: no word-level models, no endpoint pass, handicap 0.15. */
async function serveBefore(page) {
  await page.route("**/core/qorgan-config.json", async (route) => {
    const response = await route.fetch();
    const config = await response.json();
    config.asr = { ...config.asr, endpoint: null, handicap: { en: 0.15 } };
    await route.fulfill({ response, json: config });
  });
  await page.route("**/core/asr-wordlevel.json", (route) => route.fulfill({ status: 404, body: "" }));
}

async function runScene(scene) {
  const wav = joinWavs(scene.files, join(OUT, `${scene.name}.wav`));
  const server = await chromium.launchServer({
    args: ["--use-fake-ui-for-media-stream", "--use-fake-device-for-media-stream", `--use-file-for-fake-audio-capture=${wav.path}%noloop`, "--autoplay-policy=no-user-gesture-required"],
  });
  const rootPid = server.process().pid;
  const browser = await chromium.connect(server.wsEndpoint());
  const context = await browser.newContext({ viewport: { width: 1280, height: 1400 } });
  await context.grantPermissions(["microphone"], { origin: BASE });
  const page = await context.newPage();
  const errors = [];
  page.on("console", (m) => { if (m.type() === "error") errors.push(m.text()); });
  page.on("pageerror", (e) => errors.push(String(e)));
  if (VARIANT === "before") await serveBefore(page);
  await page.goto(`${BASE}/live.html?debug=asr`);
  await page.locator('label:has(input[name="lv-mode"][value="mic"])').click();
  const clicked = Date.now();
  await page.click("#lvMicStart");
  await page.waitForFunction(() => document.querySelector("#lvMicNote")?.dataset.i18n === "mic.listening", null, { timeout: 300000 });
  const listeningAfterMs = Date.now() - clicked;
  const rssListening = treeRssMb(rootPid);
  const firstLong = wav.offsets.reduce((a, b) => (b.seconds > a.seconds ? b : a));
  const shotAt = (firstLong.start + firstLong.seconds * 0.7) * 1000;
  await page.waitForTimeout(shotAt);
  await page.locator("#lvCallWrap").screenshot({ path: join(OUT, `${VARIANT}-${scene.name}-live.png`) });
  const liveSnapshot = await page.evaluate(() => [...document.querySelectorAll("#lvLiveWords .lv-run")].map((r) => ({ lang: r.dataset.lang, words: [...r.querySelectorAll(".lv-w")].map((w) => w.textContent) })));
  await page.waitForTimeout(Math.max(0, wav.seconds * 1000 - shotAt) + TAIL_S * 1000);
  const rssPeak = treeRssMb(rootPid);
  const heap = await page.evaluate(() => (performance.memory ? Math.round(performance.memory.usedJSHeapSize / 1048576) : null));
  await page.click("#lvMicStop");
  await page.waitForFunction(() => !document.querySelector("#lvSummaryWrap").hidden || document.querySelector("#lvMicNote")?.dataset.i18n === "mic.no_speech", null, { timeout: 120000 });
  await page.locator("#lvCallWrap").screenshot({ path: join(OUT, `${VARIANT}-${scene.name}-final.png`) });
  const trace = await page.evaluate(() => window.__qorganVoice ?? null);
  const summaryScore = await page.evaluate(() => document.querySelector(".lv-summary-score")?.textContent ?? null);
  const lines = await page.evaluate(() => [...document.querySelectorAll("#lvTranscript .lv-line")].map((l) => ({ source: l.dataset.source, text: l.textContent })));
  await browser.close();
  await server.close();
  const latencies = (trace?.utterances || []).map((u) => u.latencyMs).filter((x) => x !== null);
  const result = {
    variant: VARIANT, scene: scene.name, audio_s: wav.seconds, files: wav.offsets, listening_after_ms: listeningAfterMs,
    rss_mb: { listening: rssListening, peak: rssPeak }, js_heap_mb: heap, summary_score: summaryScore,
    final_meter: trace?.utterances?.at(-1)?.meter ?? null, final_band: trace?.utterances?.at(-1)?.band ?? null,
    utterances: trace?.utterances?.map(({ words, ...u }) => ({ ...u, tags: (words || []).map((w) => w.lang).join(" ") })) ?? [],
    latency_ms: latencies.length ? { n: latencies.length, median: [...latencies].sort((a, b) => a - b)[Math.floor(latencies.length / 2)], max: Math.max(...latencies) } : null,
    live_events: trace?.live?.length ?? 0, live_snapshot: liveSnapshot, endpoint: trace?.endpoint ?? [], lines, errors,
  };
  writeFileSync(join(OUT, `${VARIANT}-${scene.name}.json`), JSON.stringify(result, null, 1));
  console.log(JSON.stringify({ scene: scene.name, variant: VARIANT, final_meter: result.final_meter, summary: summaryScore, n: result.utterances.length, latency_ms: result.latency_ms, rss_mb: result.rss_mb, errors: errors.length }));
}

for (const scene of scenes) await runScene(scene);
