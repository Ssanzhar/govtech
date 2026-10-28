/* Shared helpers of the voice-mode browser runs (ADR D64, D67): joining a scene's WAVs into one
   fake-microphone file, and reading the browser's resident memory by process type. */
import { execFileSync } from "node:child_process";
import { writeFileSync } from "node:fs";
import { readWav } from "./gigaam_browser_bench.mjs";

const SPEECH_DB = -45; // a sample above this level (dBFS) counts as speech when timing a TTS line

/** Where speech starts and ends inside one clip (seconds), from its sample levels. */
export function speechBounds(samples, rate) {
  const threshold = 10 ** (SPEECH_DB / 20);
  let first = -1;
  let last = -1;
  for (let i = 0; i < samples.length; i++) {
    if (Math.abs(samples[i]) >= threshold) { if (first < 0) first = i; last = i; }
  }
  return first < 0 ? { start: 0, end: 0 } : { start: first / rate, end: (last + 1) / rate };
}

/** Join 16-bit mono WAVs (same rate) with `gapS` of silence and `leadS` before the first;
    returns {path, seconds, offsets: [{file, start, seconds, speechStart, speechEnd}]} where the
    speech times are absolute positions in the joined file. */
export function joinWavs(files, path, { leadS = 5, gapS = 1.2 } = {}) {
  const parts = files.map((f) => ({ f, ...readWav(f) }));
  const rate = parts[0].sampleRate;
  if (parts.some((p) => p.sampleRate !== rate)) throw new Error("all WAVs of a scene need one sample rate");
  const silence = (s) => new Array(Math.round(s * rate)).fill(0);
  let samples = silence(leadS);
  const offsets = [];
  for (const p of parts) {
    const start = samples.length / rate;
    const bounds = speechBounds(p.samples, rate);
    offsets.push({ file: p.f, start, seconds: p.samples.length / rate, speechStart: start + bounds.start, speechEnd: start + bounds.end });
    samples = samples.concat(Array.from(p.samples), silence(gapS));
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

/** Resident memory (MB) of a browser started with `--user-data-dir=<userDataDir>` and all its
    descendants, by process type (`--type=` / `--utility-sub-type=` of each command line; the
    root is "browser"). Works without a CDP browser session (persistent contexts). */
export function processTreeRssMb(userDataDir) {
  const found = processTree(userDataDir);
  if (!found) return null;
  const { rows, root, tree } = found;
  const byType = {};
  let total = 0;
  for (const r of rows) {
    if (!tree.has(r.pid)) continue;
    const type = r.pid === root.pid ? "browser" : (r.command.match(/--utility-sub-type=([\w.]+)/)?.[1] || r.command.match(/--type=([\w-]+)/)?.[1] || "other");
    byType[type] = Math.round((byType[type] || 0) + r.kb / 1024);
    total += r.kb / 1024;
  }
  return { total: Math.round(total), ...byType };
}

/** The browser started with `--user-data-dir=<userDataDir>` and all its descendants:
    {rows (every process), root, tree (a Set of pids)}, or null when it is not running. */
function processTree(userDataDir) {
  const rows = execFileSync("ps", ["-A", "-ww", "-o", "pid=,ppid=,rss=,command="], { encoding: "utf8" }).trim().split("\n").map((line) => {
    const m = line.trim().match(/^(\d+)\s+(\d+)\s+(\d+)\s+(.*)$/);
    return m ? { pid: Number(m[1]), ppid: Number(m[2]), kb: Number(m[3]), command: m[4] } : null;
  }).filter(Boolean);
  const root = rows.find((r) => r.command.includes(`--user-data-dir=${userDataDir}`) && !r.command.includes("--type="));
  if (!root) return null;
  const tree = new Set([root.pid]);
  for (let grew = true; grew; ) {
    grew = false;
    for (const r of rows) if (!tree.has(r.pid) && tree.has(r.ppid)) { tree.add(r.pid); grew = true; }
  }
  return { rows, root, tree };
}

/** macOS only: puts this browser's renderer processes (the page and its workers: GigaAM, e5)
    in the background QoS band (`taskpolicy -b -p`), which keeps their threads on the efficiency
    cores at a low clock -- a rough stand-in for a phone's CPU. The audio and browser processes
    are left alone, as a phone keeps audio capture at high priority (moving them too starves the
    fake microphone and nothing is heard). Returns how many processes were newly moved; call it
    again for processes started later (pids already moved are skipped via `done`). */
export function backgroundProcessTree(userDataDir, done) {
  const found = processTree(userDataDir);
  if (!found) return 0;
  let moved = 0;
  for (const r of found.rows) {
    const pid = r.pid;
    if (!found.tree.has(pid) || done.has(pid) || !r.command.includes("--type=renderer")) continue;
    try {
      execFileSync("taskpolicy", ["-b", "-p", String(pid)]);
      done.add(pid);
      moved += 1;
    } catch (err) {
      console.error(`taskpolicy -b -p ${pid} failed: ${err.message}`);
    }
  }
  return moved;
}

/** Resident memory (MB) of every process of this browser, by type, as Chromium lists them. */
export async function browserRssMb(cdp, rootPid) {
  const { processInfo } = await cdp.send("SystemInfo.getProcessInfo");
  const pids = new Map(processInfo.map((p) => [p.id, p.type]));
  pids.set(rootPid, "browser");
  const rows = execFileSync("ps", ["-A", "-o", "pid=,rss="], { encoding: "utf8" }).trim().split("\n").map((l) => l.trim().split(/\s+/).map(Number));
  const byType = {};
  let total = 0;
  for (const [pid, kb] of rows) {
    if (!pids.has(pid)) continue;
    const type = pids.get(pid);
    byType[type] = Math.round((byType[type] || 0) + kb / 1024);
    total += kb / 1024;
  }
  return { total: Math.round(total), ...byType };
}
