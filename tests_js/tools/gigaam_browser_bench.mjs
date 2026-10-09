/* GigaAM in the browser (ADR D64): the site's real worker (site/core/gigaam-worker.js, the
   single-threaded sherpa-onnx WASM build) in headless Chromium on a page that is NOT
   cross-origin isolated, decoding FLEURS utterances. Compares each transcript with the native
   sherpa-onnx decode of the same file (data/asr_capture/fleurs_decodes_<channel>.jsonl, the
   spike's numbers) and prints one JSON summary: load time, RTF, text agreement, memory.

     node tests_js/tools/gigaam_browser_bench.mjs [perLanguage=10] [channel=clean]

   Needs the gitignored assets: site/vendor/sherpa-onnx/<v>/ and site/models/gigaam/<id>/
   (deploy_bootstrap), data/asr_capture/_wav/fleurs/ and the decodes (the spike). */
import { execFileSync } from "node:child_process";
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { pathToFileURL } from "node:url";
import { REPO, serveSite } from "./device_runtime.mjs";

const perLanguage = Number(process.argv[2] || 10);
const channel = process.argv[3] || "clean";
const config = JSON.parse(readFileSync(join(REPO, "site/core/qorgan-config.json"), "utf8"));
const endpoint = config.asr.endpoint;

/** 16-bit PCM mono WAV -> {samples: number[] in [-1, 1], sampleRate}. */
export function readWav(path) {
  const buf = readFileSync(path);
  let offset = 12;
  let fmt = null;
  while (offset + 8 <= buf.length) {
    const id = buf.toString("ascii", offset, offset + 4);
    const size = buf.readUInt32LE(offset + 4);
    if (id === "fmt ") fmt = { channels: buf.readUInt16LE(offset + 10), sampleRate: buf.readUInt32LE(offset + 12), bits: buf.readUInt16LE(offset + 22) };
    if (id === "data") {
      if (!fmt || fmt.bits !== 16 || fmt.channels !== 1) throw new Error(`${path}: expected 16-bit mono PCM`);
      const n = size / 2;
      const samples = new Array(n);
      for (let i = 0; i < n; i++) samples[i] = buf.readInt16LE(offset + 8 + 2 * i) / 32768;
      return { samples, sampleRate: fmt.sampleRate };
    }
    offset += 8 + size + (size % 2);
  }
  throw new Error(`${path}: no data chunk`);
}

/** Resident memory (MB) of the browser and all its child processes. */
function treeRssMb(rootPid) {
  const rows = execFileSync("ps", ["-A", "-o", "pid=,ppid=,rss="], { encoding: "utf8" }).trim().split("\n").map((l) => l.trim().split(/\s+/).map(Number));
  const children = new Map();
  for (const [pid, ppid] of rows) children.set(ppid, [...(children.get(ppid) || []), pid]);
  const rss = new Map(rows.map(([pid, , kb]) => [pid, kb]));
  let total = 0;
  const stack = [rootPid];
  while (stack.length) {
    const pid = stack.pop();
    total += rss.get(pid) || 0;
    stack.push(...(children.get(pid) || []));
  }
  return total / 1024;
}

const median = (xs) => { const s = [...xs].sort((a, b) => a - b); return s.length ? s[Math.floor(s.length / 2)] : null; };

async function main() {
  const decodes = new Map(readFileSync(join(REPO, `data/asr_capture/fleurs_decodes_${channel}.jsonl`), "utf8").trim().split("\n").map((l) => JSON.parse(l)).map((r) => [r.id, r]));
  const set = readFileSync(join(REPO, "data/asr_capture/fleurs_set.jsonl"), "utf8").trim().split("\n").map((l) => JSON.parse(l)).filter((r) => r.channel === channel);
  const rows = ["kk", "ru", "en"].flatMap((lang) => set.filter((r) => r.lang === lang && decodes.get(r.id)?.gigaam220).slice(0, perLanguage));

  const { chromium } = await import("playwright");
  const server = serveSite();
  await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
  const browserServer = await chromium.launchServer();
  const rootPid = browserServer.process().pid;
  const browser = await chromium.connect(browserServer.wsEndpoint());
  const page = await browser.newPage();
  page.on("console", (m) => { if (m.type() === "error") console.error(`[browser] ${m.text()}`); });
  await page.goto(`http://127.0.0.1:${server.address().port}/index.html`);
  const isolation = await page.evaluate(() => ({ crossOriginIsolated: globalThis.crossOriginIsolated, sab: typeof SharedArrayBuffer !== "undefined" }));
  const rssBefore = rootPid ? treeRssMb(rootPid) : null;

  const ready = await page.evaluate(async (ep) => {
    window.__gw = new Worker("/core/gigaam-worker.js");
    window.__pending = new Map();
    const readyP = new Promise((resolve, reject) => {
      window.__gw.onmessage = (ev) => {
        const m = ev.data;
        if (m.type === "ready") resolve(m);
        else if (m.type === "error" && m.stage === "init") reject(new Error(m.message));
        else if (m.type === "result" || m.type === "error") { window.__pending.get(m.id)?.(m); window.__pending.delete(m.id); }
      };
    });
    window.__gw.postMessage({ type: "init", runtimeBase: new URL(ep.runtime, location.href).href, modelUrl: new URL(`models/${ep.model.url}`, location.href).href, tokensUrl: new URL(`models/${ep.model.tokens}`, location.href).href, cacheName: ep.cache });
    return readyP;
  }, endpoint);
  const rssLoaded = rootPid ? treeRssMb(rootPid) : null;

  const results = [];
  for (const row of rows) {
    const { samples, sampleRate } = readWav(join(REPO, row.wav));
    const out = await page.evaluate(({ id, samples, sampleRate }) => new Promise((resolve) => {
      window.__pending.set(id, resolve);
      const f32 = new Float32Array(samples);
      window.__gw.postMessage({ type: "decode", id, samples: f32, sampleRate }, [f32.buffer]);
    }), { id: row.id, samples, sampleRate });
    const native = decodes.get(row.id).gigaam220;
    results.push({ id: row.id, lang: row.lang, audioS: out.audioS, decodeMs: out.decodeMs, rtf: out.decodeMs / 1000 / out.audioS, same: out.text === native.text, browser: out.text, native: native.text, error: out.type === "error" ? out.message : null });
  }
  const rssPeak = rootPid ? treeRssMb(rootPid) : null;

  // Second load from the Cache API (a fresh worker): the "cached after the first time" path.
  const reload = await page.evaluate(async (ep) => {
    const w = new Worker("/core/gigaam-worker.js");
    const m = await new Promise((resolve) => {
      w.onmessage = (ev) => { if (ev.data.type === "ready" || ev.data.type === "error") resolve(ev.data); };
      w.postMessage({ type: "init", runtimeBase: new URL(ep.runtime, location.href).href, modelUrl: new URL(`models/${ep.model.url}`, location.href).href, tokensUrl: new URL(`models/${ep.model.tokens}`, location.href).href, cacheName: ep.cache });
    });
    w.terminate();
    return m;
  }, endpoint);

  const byLang = Object.fromEntries(["kk", "ru", "en"].map((lang) => {
    const r = results.filter((x) => x.lang === lang);
    return [lang, { n: r.length, identical_to_native: r.filter((x) => x.same).length, median_rtf: median(r.map((x) => x.rtf)) }];
  }));
  const summary = {
    channel, isolation, ready, cached_reload: reload,
    n: results.length, errors: results.filter((x) => x.error).length,
    identical_to_native: results.filter((x) => x.same).length,
    rtf_median: median(results.map((x) => x.rtf)), rtf_total: results.reduce((a, x) => a + x.decodeMs / 1000, 0) / results.reduce((a, x) => a + x.audioS, 0),
    by_language: byLang,
    rss_mb: { before_model: rssBefore, after_load: rssLoaded, after_decodes: rssPeak },
    differing: results.filter((x) => !x.same).slice(0, 5).map(({ id, browser, native }) => ({ id, browser, native })),
    results: results.map(({ id, lang, audioS, decodeMs, browser, native }) => ({ id, lang, audioS, decodeMs, browser, native })),
    user_agent: await page.evaluate(() => navigator.userAgent),
  };
  console.log(JSON.stringify(summary, null, 1));
  await browser.close();
  await browserServer.close();
  server.close();
}

if (import.meta.url === pathToFileURL(process.argv[1]).href) main().catch((err) => { console.error(err); process.exit(1); });
