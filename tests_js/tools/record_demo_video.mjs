/* A backup video of the demo: the real live page in Chromium listening to a scene through a fake
   microphone, recorded by Playwright, with the scene's audio muxed back in by ffmpeg so the
   video plays what the page heard. For the pitch, in case the room's network or microphone
   fails. Nothing leaves the machine.

     node tests_js/tools/static_site.mjs --isolation off --port 8410 &   # light mode, as in a WebView
     node tests_js/tools/record_demo_video.mjs --base http://127.0.0.1:8410 --out DIR \
          --scene NAME=a.wav,b.wav [--scene ...] [--lang ru|kk] [--phone] [--gap 1.2]

   Per scene: DIR/<scene>.mp4 (H.264 + AAC) and DIR/<scene>.json (the audio offset used, the
   final meter). Desktop viewport (1280 x 800) by default: meter, tags, advice and transcript fit
   on one screen. `--phone` (412 x 915) is taller than a screen, so the meter scrolls out of view
   as the call grows. The first
   visit only fills the model caches and is not recorded. Audio alignment: the fake microphone
   starts when getUserMedia resolves (the page's `captureAt`), so the WAV is delayed by that
   moment's distance from the start of the recording (error: the page's start-up jitter, tens of
   ms). */
import { execFileSync } from "node:child_process";
import { mkdirSync, mkdtempSync, readdirSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { chromium } from "playwright";
import { joinWavs } from "./voice_e2e_lib.mjs";

const args = process.argv.slice(2);
const opt = (name, fallback) => { const i = args.indexOf(name); return i > -1 ? args[i + 1] : fallback; };
const BASE = opt("--base", "http://127.0.0.1:8410");
const OUT = resolve(opt("--out", "demo-video"));
const LANG = opt("--lang", "ru");
const DESKTOP = !args.includes("--phone");
const GAP_S = Number(opt("--gap", "1.2"));
const LEAD_S = 2;
const TAIL_MS = 6000; // after the audio: the last phrase is transcribed and scored
const HOLD_MS = 5000; // how long each closing screen stays on the video
const SIZE = DESKTOP ? { width: 1280, height: 800 } : { width: 412, height: 915 };
const scenes = args.flatMap((a, i) => (a === "--scene" ? [args[i + 1]] : [])).map((s) => {
  const [name, files] = s.split("=");
  return { name, files: files.split(",") };
});
if (!scenes.length) throw new Error("give at least one --scene NAME=a.wav,b.wav");
if (!["ru", "kk"].includes(LANG)) throw new Error("--lang is ru or kk");
mkdirSync(OUT, { recursive: true });

const pageUrl = `${BASE}/live.html?debug=asr`;

function launch(userDataDir, wavPath, videoDir) {
  return chromium.launchPersistentContext(userDataDir, {
    viewport: SIZE,
    locale: LANG === "kk" ? "kk-KZ" : "ru-RU", // the page picks its UI language from the browser's
    ...(DESKTOP ? {} : { deviceScaleFactor: 2, isMobile: true, hasTouch: true }),
    ...(videoDir ? { recordVideo: { dir: videoDir, size: SIZE } } : {}),
    permissions: ["microphone"],
    args: ["--use-fake-ui-for-media-stream", "--use-fake-device-for-media-stream", `--use-file-for-fake-audio-capture=${wavPath}%noloop`, "--autoplay-policy=no-user-gesture-required"],
  });
}

async function startMic(page) {
  await page.locator('label:has(input[name="lv-mode"][value="mic"])').click();
  await page.click("#lvMicStart");
  await page.waitForFunction(() => ["mic.listening", "mic.light_unavailable", "mic.start_failed"].includes(document.querySelector("#lvMicNote")?.dataset.i18n), null, { timeout: 300000 });
  const note = await page.evaluate(() => document.querySelector("#lvMicNote")?.dataset.i18n);
  if (note !== "mic.listening") throw new Error(`the microphone did not start: ${note}`);
}

/** Scrolls so `selector` starts just under the page's sticky header. */
async function pinToTop(page, selector) {
  await page.evaluate((sel) => {
    const header = document.querySelector("header");
    const offset = (header?.getBoundingClientRect().height ?? 0) + 8;
    window.scrollTo({ top: document.querySelector(sel).getBoundingClientRect().top + window.scrollY - offset, behavior: "instant" });
  }, selector);
}

/** The first visit: downloads the models into this profile's caches; not recorded. */
async function warmUp(userDataDir, wavPath) {
  const context = await launch(userDataDir, wavPath, null);
  const page = context.pages()[0] || (await context.newPage());
  await page.goto(pageUrl);
  await startMic(page);
  await page.waitForTimeout(3000);
  await context.close();
}

async function recordScene(scene) {
  const wav = joinWavs(scene.files, join(OUT, `${scene.name}.wav`), { leadS: LEAD_S, gapS: GAP_S });
  const userDataDir = mkdtempSync(join(tmpdir(), "qorgan-video-"));
  const videoDir = mkdtempSync(join(tmpdir(), "qorgan-video-raw-"));
  try {
    await warmUp(userDataDir, wav.path);
    const context = await launch(userDataDir, wav.path, videoDir);
    const videoStart = Date.now(); // the recording starts with the context's first page
    const page = context.pages()[0] || (await context.newPage());
    await page.goto(pageUrl);
    await page.waitForTimeout(1500); // a beat on the start screen
    await startMic(page);
    const captureWall = await page.evaluate(() => performance.timeOrigin + window.__qorganVoice.captureAt);
    await pinToTop(page, "#lvCallWrap"); // the meter heads the panel; keep it on screen all call
    await page.waitForTimeout(wav.seconds * 1000 + TAIL_MS);
    const finalMeter = await page.evaluate(() => window.__qorganVoice?.utterances?.at(-1)?.meter ?? null);
    await page.click("#lvMicStop");
    await page.waitForFunction(() => !document.querySelector("#lvSummaryWrap").hidden, null, { timeout: 120000 });
    await page.locator("#lvSummaryWrap").scrollIntoViewIfNeeded();
    await page.waitForTimeout(HOLD_MS);
    await page.locator("#lvReport").scrollIntoViewIfNeeded();
    await page.waitForTimeout(HOLD_MS);
    await context.close(); // finishes the video file
    const raw = readdirSync(videoDir).find((f) => f.endsWith(".webm"));
    if (!raw) throw new Error("Playwright wrote no video");
    const offsetS = Math.max(0, (captureWall - videoStart) / 1000);
    const mp4 = join(OUT, `${scene.name}.mp4`);
    execFileSync("ffmpeg", ["-y", "-loglevel", "error", "-i", join(videoDir, raw), "-itsoffset", offsetS.toFixed(3), "-i", wav.path,
      "-map", "0:v", "-map", "1:a", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", "-c:a", "aac", "-b:a", "128k", mp4]);
    const result = { scene: scene.name, lang: LANG, viewport: SIZE, audio_s: wav.seconds, audio_offset_s: Number(offsetS.toFixed(3)), final_meter: finalMeter && Math.round(finalMeter), video: mp4 };
    writeFileSync(join(OUT, `${scene.name}.json`), JSON.stringify(result, null, 1));
    console.log(JSON.stringify(result));
  } finally {
    rmSync(userDataDir, { recursive: true, force: true });
    rmSync(videoDir, { recursive: true, force: true });
  }
}

let failed = 0;
for (const scene of scenes) {
  try {
    await recordScene(scene);
  } catch (err) {
    failed += 1;
    console.log(JSON.stringify({ scene: scene.name, failed: String(err?.message || err).split("\n")[0] }));
  }
}
process.exitCode = failed ? 1 : 0;
