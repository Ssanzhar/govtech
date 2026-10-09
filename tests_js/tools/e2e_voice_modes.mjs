/* The voice modes end to end, isolated vs not (ADR D67): the real live page in Chromium, a fake
   microphone playing a scene's WAVs, served by tests_js/tools/static_site.mjs with cross-origin
   isolation ON (the full D64 word-level mode) or OFF (what Android WebView gets: the light
   GigaAM-only mode with the energy VAD).

     node tests_js/tools/static_site.mjs --isolation off --port 8410 &
     node tests_js/tools/static_site.mjs --isolation on  --port 8411 &
     node tests_js/tools/e2e_voice_modes.mjs --base http://127.0.0.1:8410 --label light --out DIR \
          --scene NAME=a.wav,b.wav [--scene ...] [--gap 1.2] [--lead 5] [--voice full|light] [--warm] [--mobile]

   One WAV = one reference utterance. Per scene it writes DIR/<label>-<scene>.json with: the
   mode the page chose and `crossOriginIsolated`; every final utterance (text, source, meter);
   per reference line, the latency from the end of its speech (measured from the WAV's levels)
   to its transcript line + meter on screen; the light mode's VAD segments matched against the
   reference lines (found / split / merged / spurious, boundary error); and the browser's
   resident memory sampled every second (peak and end of audio, by process type; dedicated
   workers live in the renderer). Plus two screenshots. Timing caveat: file time 0 is taken as
   the moment getUserMedia resolved (Chromium starts the fake file then), so absolute times
   carry the page's start-up jitter (tens of ms). */
import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { chromium } from "playwright";
import { joinWavs, processTreeRssMb } from "./voice_e2e_lib.mjs";

const args = process.argv.slice(2);
const opt = (name, fallback) => { const i = args.indexOf(name); return i > -1 ? args[i + 1] : fallback; };
const BASE = opt("--base", "http://127.0.0.1:8410");
const OUT = resolve(opt("--out", "e2e-voice-modes"));
const LABEL = opt("--label", "run");
const VOICE = opt("--voice", null);
const WARM = args.includes("--warm"); // measure a second visit (models already cached)
// `--mobile`: a phone viewport and an Android System WebView user agent ("; wv"), so the page's
// own phone checks see what eGov Mobile's WebView reports. Still desktop Chromium underneath.
const MOBILE = args.includes("--mobile");
const WEBVIEW_UA = "Mozilla/5.0 (Linux; Android 14; Pixel 7 Build/AP2A.240805.005; wv) AppleWebKit/537.36 (KHTML, like Gecko) Version/4.0 Chrome/141.0.0.0 Mobile Safari/537.36";
const GAP_S = Number(opt("--gap", "1.2"));
const LEAD_S = Number(opt("--lead", "5"));
const TAIL_S = 8;
const SETTLE_SAMPLES = 5; // per-second memory samples skipped after "listening" (loading transients)
const scenes = args.flatMap((a, i) => (a === "--scene" ? [args[i + 1]] : [])).map((s) => {
  const [name, files] = s.split("=");
  return { name, files: files.split(",") };
});
if (!scenes.length) throw new Error("give at least one --scene NAME=a.wav,b.wav");
mkdirSync(OUT, { recursive: true });

const median = (xs) => (xs.length ? [...xs].sort((a, b) => a - b)[Math.floor(xs.length / 2)] : null);
const stats = (xs) => (xs.length ? { n: xs.length, median: Math.round(median(xs)), max: Math.round(Math.max(...xs)) } : null);
const overlap = (a, b) => Math.min(a.end, b.end) - Math.max(a.start, b.start) > 0;

/** Per reference line: ms from its speech end to its transcript line + meter on screen. Only
    defined when the run produced exactly one final utterance per line (then the k-th utterance
    is the k-th line's); otherwise every entry is null and the scene reports no latency, rather
    than guessing which piece of a split line completed it. */
function lineLatencies(lines, utterances, zero) {
  if (utterances.length !== lines.length) return lines.map(() => null);
  return lines.map((line, i) => utterances[i].at - (zero + line.speechEnd * 1000));
}

/** The light mode's VAD segments (file seconds) against the reference lines. */
function segmentation(lines, segments) {
  const refs = lines.map((l) => ({ start: l.speechStart, end: l.speechEnd }));
  const perLine = refs.map((r) => segments.filter((s) => overlap(r, s)));
  const perSeg = segments.map((s) => refs.filter((r) => overlap(r, s)).length);
  const oneToOne = refs.map((r, i) => (perLine[i].length === 1 && perSeg[segments.indexOf(perLine[i][0])] === 1 ? { r, s: perLine[i][0] } : null)).filter(Boolean);
  return {
    reference_lines: refs.length,
    segments: segments.length,
    lines_found: perLine.filter((s) => s.length > 0).length,
    lines_split: perLine.filter((s) => s.length > 1).length,
    segments_merging_lines: perSeg.filter((n) => n > 1).length,
    spurious_segments: perSeg.filter((n) => n === 0).length,
    one_to_one: oneToOne.length,
    start_error_ms: stats(oneToOne.map(({ r, s }) => Math.abs(s.start - r.start) * 1000)),
    end_error_ms: stats(oneToOne.map(({ r, s }) => Math.abs(s.end - r.end) * 1000)),
    cuts: segments.map((s) => s.cut),
  };
}

async function runScene(scene) {
  const wav = joinWavs(scene.files, join(OUT, `${scene.name}.wav`), { leadS: LEAD_S, gapS: GAP_S });
  const userDataDir = mkdtempSync(join(tmpdir(), "qorgan-voice-"));
  const launch = () => chromium.launchPersistentContext(userDataDir, {
    ...(MOBILE
      ? { viewport: { width: 412, height: 915 }, deviceScaleFactor: 2.625, isMobile: true, hasTouch: true, userAgent: WEBVIEW_UA }
      : { viewport: { width: 1280, height: 1400 } }),
    permissions: ["microphone"],
    args: ["--use-fake-ui-for-media-stream", "--use-fake-device-for-media-stream", `--use-file-for-fake-audio-capture=${wav.path}%noloop`, "--autoplay-policy=no-user-gesture-required"],
  });
  if (WARM) {
    // A returning user: the first visit downloads the models into the Cache API / HTTP cache,
    // then a NEW browser process on the same profile is measured.
    const warm = await launch();
    const warmPage = warm.pages()[0] || (await warm.newPage());
    await warmPage.goto(`${BASE}/live.html?debug=asr${VOICE ? `&voice=${VOICE}` : ""}`);
    await warmPage.locator('label:has(input[name="lv-mode"][value="mic"])').click();
    await warmPage.click("#lvMicStart");
    await warmPage.waitForFunction(() => document.querySelector("#lvMicNote")?.dataset.i18n === "mic.listening", null, { timeout: 300000 });
    await warmPage.waitForTimeout(3000);
    await warm.close();
  }
  const context = await launch();
  const rss = async () => processTreeRssMb(userDataDir);
  const page = context.pages()[0] || (await context.newPage());
  const errors = [];
  page.on("console", (m) => { if (m.type() === "error") errors.push(m.text()); });
  page.on("pageerror", (e) => errors.push(String(e)));
  await page.goto(`${BASE}/live.html?debug=asr${VOICE ? `&voice=${VOICE}` : ""}`);
  const isolated = await page.evaluate(() => globalThis.crossOriginIsolated === true);
  const rssIdle = await rss();
  await page.locator('label:has(input[name="lv-mode"][value="mic"])').click();
  const clicked = Date.now();
  await page.click("#lvMicStart");
  await page.waitForFunction(() => ["mic.listening", "mic.light_unavailable", "mic.start_failed"].includes(document.querySelector("#lvMicNote")?.dataset.i18n), null, { timeout: 300000 });
  const note = await page.evaluate(() => document.querySelector("#lvMicNote")?.dataset.i18n);
  if (note !== "mic.listening") throw new Error(`the microphone did not start: ${note} ${errors.join(" | ")}`);
  const listeningAfterMs = Date.now() - clicked;
  const rssListening = await rss();

  const samples = [];
  const sampler = setInterval(() => { rss().then((r) => r && samples.push(r), () => {}); }, 1000);
  const longest = wav.offsets.reduce((a, b) => (b.seconds > a.seconds ? b : a));
  const shotAt = (longest.start + longest.seconds * 0.8) * 1000;
  await page.waitForTimeout(shotAt);
  await page.locator("#lvCallWrap").screenshot({ path: join(OUT, `${LABEL}-${scene.name}-live.png`) });
  await page.waitForTimeout(Math.max(0, wav.seconds * 1000 - shotAt) + TAIL_S * 1000);
  clearInterval(sampler);
  const rssEnd = await rss();
  await page.click("#lvMicStop");
  await page.waitForFunction(() => !document.querySelector("#lvSummaryWrap").hidden || document.querySelector("#lvMicNote")?.dataset.i18n === "mic.no_speech", null, { timeout: 120000 });
  await page.locator("#lvCallWrap").screenshot({ path: join(OUT, `${LABEL}-${scene.name}-final.png`) });
  const setupShot = join(OUT, `${LABEL}-${scene.name}-setup.png`);
  await page.locator("#lvSetupMic").screenshot({ path: setupShot }).catch(() => {});
  const trace = await page.evaluate(() => window.__qorganVoice ?? null);
  const summaryScore = await page.evaluate(() => document.querySelector(".lv-summary-score")?.textContent ?? null);
  const modeNote = await page.evaluate(() => { const el = document.querySelector("#lvVoiceModeNote"); return el && !el.hidden ? el.dataset.i18n : null; });
  await context.close();
  rmSync(userDataDir, { recursive: true, force: true });

  const zero = trace.captureAt;
  const streamOffsetS = (trace.listeningAt - trace.captureAt) / 1000; // stream time 0 is when capture began feeding the recognisers
  const utterances = trace.utterances || [];
  const latencies = lineLatencies(wav.offsets, utterances, zero);
  const segments = (trace.speech || []).filter((e) => e.type === "segment").map((e) => ({ start: e.start + streamOffsetS, end: e.end + streamOffsetS, cut: e.cut, closedAfterSpeechMs: null, wallAt: e.wallAt }));
  for (const s of segments) {
    const line = wav.offsets.find((l) => overlap({ start: l.speechStart, end: l.speechEnd }, s));
    if (line) s.closedAfterSpeechMs = Math.round(s.wallAt - (zero + line.speechEnd * 1000));
  }
  const peakBy = (type) => Math.max(0, ...samples.map((r) => r[type] || 0));
  const result = {
    label: LABEL, scene: scene.name, base: BASE, warm_cache: WARM, mobile_webview_ua: MOBILE, capture_at: trace.captureAt, listening_at: trace.listeningAt, cross_origin_isolated: isolated, mode: trace.mode, mode_note: modeNote,
    audio_s: wav.seconds, lines: wav.offsets, listening_after_ms: listeningAfterMs,
    gigaam: (trace.endpoint || []).find((e) => e.status === "ready") ?? (trace.endpoint || []).at(-1) ?? null,
    rss_mb: {
      idle_page: rssIdle, listening: rssListening, end_of_audio: rssEnd, peak_renderer: peakBy("renderer"), peak_total: Math.max(0, ...samples.map((r) => r.total)),
      // During the call, past the model-loading transients (the first SETTLE_SAMPLES seconds after "listening").
      median_renderer_in_call: median(samples.slice(SETTLE_SAMPLES).map((r) => r.renderer || 0)),
      median_total_in_call: median(samples.slice(SETTLE_SAMPLES).map((r) => r.total)),
      renderer_per_second: samples.map((r) => r.renderer || 0),
    },
    summary_score: summaryScore, final_meter: utterances.at(-1)?.meter ?? null, final_band: utterances.at(-1)?.band ?? null,
    utterances: utterances.map(({ words, ...u }) => ({ ...u, tags: (words || []).map((w) => w.lang).join(" ") })),
    line_latency_ms: latencies, latency_ms: stats(latencies.filter((x) => x !== null)),
    decode_ms: stats(utterances.map((u) => u.decodeMs).filter((x) => x !== null)),
    segmentation: trace.mode === "light" ? segmentation(wav.offsets, segments) : null,
    vad_close_after_speech_ms: trace.mode === "light" ? stats(segments.map((s) => s.closedAfterSpeechMs).filter((x) => x !== null)) : null,
    segments, dropped: trace.dropped ?? [], errors,
  };
  writeFileSync(join(OUT, `${LABEL}-${scene.name}.json`), JSON.stringify(result, null, 1));
  console.log(JSON.stringify({
    scene: scene.name, label: LABEL, isolated, mode: result.mode, final_meter: result.final_meter && Math.round(result.final_meter), summary: summaryScore,
    n: utterances.length, lines: wav.offsets.length, latency_ms: result.latency_ms, seg: result.segmentation && { found: result.segmentation.lines_found, segs: result.segmentation.segments, split: result.segmentation.lines_split, merged: result.segmentation.segments_merging_lines, spurious: result.segmentation.spurious_segments },
    peak_renderer: result.rss_mb.peak_renderer, errors: errors.length,
  }));
}

let failed = 0;
for (const scene of scenes) {
  try {
    await runScene(scene);
  } catch (err) {
    failed += 1;
    console.log(JSON.stringify({ scene: scene.name, label: LABEL, failed: String(err?.message || err).split("\n")[0] }));
  }
}
process.exitCode = failed ? 1 : 0;
