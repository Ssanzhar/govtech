/* The PWA promise in real Chrome (PLAN B4, ADR D86): after one visit the citizen page checks a call
   with the server gone and the browser restarted. Runs its own static server and browser profile:

     QORGAN_E2E_CHANNEL=chrome node tests_js/tools/e2e_offline.mjs

   The service worker does not register on localhost, so the site is served as qorgan.test, mapped to
   the loopback and treated as a secure origin. Needs the embedder under site/models/ (bootstrap). */

import assert from "node:assert/strict";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { chromium } from "playwright";
import { startStaticSite } from "./static_site.mjs";

const PORT = 8732;
const ORIGIN = `http://qorgan.test:${PORT}`;
const profile = mkdtempSync(join(tmpdir(), "qorgan-offline-"));
const SCRIPT = "Здравствуйте, это служба безопасности банка.\nНазовите код из смс.\nНикому не говорите об этом звонке.";

const open = () => chromium.launchPersistentContext(profile, {
  ...(process.env.QORGAN_E2E_CHANNEL ? { channel: process.env.QORGAN_E2E_CHANNEL } : {}),
  args: [`--host-resolver-rules=MAP qorgan.test 127.0.0.1`, `--unsafely-treat-insecure-origin-as-secure=${ORIGIN}`],
});

const replay = async (page, label) => {
  await page.fill("#lvScript", SCRIPT);
  await page.click("#lvStart");
  await page.waitForFunction(() => document.getElementById("lvReportOpen") || document.querySelector("#lvSetupNote.is-error"), null, { timeout: 180_000 });
  const failure = await page.evaluate(() => document.querySelector("#lvSetupNote.is-error")?.textContent || null);
  assert.equal(failure, null, `${label}: the replay failed`);
  const score = (await page.textContent(".lv-summary-score")).trim();
  console.log(`${label}: ${score}`);
  return score;
};

const server = await startStaticSite({ port: PORT, isolation: "on" });
let context = await open();
try {
  // 1. One visit, online.
  let page = await context.newPage();
  await page.goto(`${ORIGIN}/live.html`);
  await page.waitForFunction(() => navigator.serviceWorker.controller !== null, null, { timeout: 60_000 });
  assert.equal(await page.evaluate(() => crossOriginIsolated), true);
  const online = await replay(page, "online");
  // The copy is kept on the side, after the page already has its own.
  await page.waitForFunction(async () => {
    for (const name of await caches.keys()) {
      if (!name.startsWith("qorgan-binaries-")) continue;
      const kept = await (await caches.open(name)).keys();
      if (kept.some((request) => request.url.endsWith(".wasm"))) return true;
    }
    return false;
  }, null, { timeout: 120_000, polling: 500 });

  // 2. The server is gone; the same browser reloads.
  await new Promise((resolve) => server.close(resolve));
  server.closeAllConnections?.();
  await page.reload();
  await page.waitForFunction(() => document.querySelectorAll("#lvScenario option").length >= 1);
  assert.equal(await page.evaluate(() => crossOriginIsolated), true, "still isolated, from cached headers");
  assert.equal(await replay(page, "server gone, reload"), online, "the same answer");

  // 3. The browser restarts too: a new service worker process with an empty memory.
  await context.close();
  context = await open();
  page = await context.newPage();
  await page.goto(`${ORIGIN}/live.html`);
  await page.waitForFunction(() => document.querySelectorAll("#lvScenario option").length >= 1);
  assert.equal(await replay(page, "server gone, browser restarted"), online, "the same answer");
  console.log("offline e2e: OK");
} finally {
  await context.close().catch(() => {});
  rmSync(profile, { recursive: true, force: true });
}
process.exit(0);
