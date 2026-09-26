/* End-to-end check of analyst authentication on the Level-2 console in a real browser.

   Run a server first (QORGAN_ANALYST_KEYS + QORGAN_AUDIT_CHAIN_KEY set, e.g. from .env) —
   ideally against a scratch copy of data/ so the real audit log stays untouched:
     QORGAN_DATA_DIR=/tmp/scratch/data QORGAN_API_PORT=8011 python -m qorgan.api
   then (QORGAN_E2E_CHANNEL=chrome drives the installed Chrome):
     node tests_js/tools/e2e_admin_auth.mjs --base http://127.0.0.1:8011 --shots <dir>
   With --closed the server is expected to run WITHOUT analyst keys (checks the 503 path).

   Keys are read from the repo's .env (first `analyst` and first `investigator` entry of
   QORGAN_ANALYST_KEYS), never from the command line. Asserts: unauthenticated and
   caller-named requests get 401; nothing is fetched from /api/admin before sign-in; the key
   lives in sessionStorage only and rides on every console request; the analyst role renders
   the overview but the open-case control is locked with the reason; the investigator opens
   a case only after choosing a purpose; a 401 mid-session returns to the sign-in panel. */

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import { chromium } from "playwright";

const arg = (name, fallback) => {
  const i = process.argv.indexOf(name);
  return i > -1 ? process.argv[i + 1] : fallback;
};
const BASE = arg("--base", "http://127.0.0.1:8011");
const SHOTS = arg("--shots", null);
const CLOSED = process.argv.includes("--closed");
const shot = async (page, name) => {
  if (SHOTS) await page.screenshot({ path: join(SHOTS, name), fullPage: true });
};

const envFile = readFileSync(join(dirname(fileURLToPath(import.meta.url)), "..", "..", ".env"), "utf8");
const entries = (envFile.match(/^QORGAN_ANALYST_KEYS=(.*)$/m)?.[1] || "")
  .split(",").map((e) => e.trim().split(":")).filter((f) => f.length === 3);
const byRole = (role) => entries.find((f) => f[2] === role);
const [analystId, analystKey] = byRole("analyst") || [];
const [investigatorId, investigatorKey] = byRole("investigator") || [];
assert.ok(analystKey && investigatorKey, ".env must hold one analyst and one investigator key");

const status = async (path, headers = {}, init = {}) => (await fetch(BASE + path, { headers, ...init })).status;
const results = {};

const browser = await chromium.launch(process.env.QORGAN_E2E_CHANNEL ? { channel: process.env.QORGAN_E2E_CHANNEL } : {});
try {
  const context = await browser.newContext({ viewport: { width: 1280, height: 900 } });
  const page = await context.newPage();
  const adminRequests = [];
  page.on("request", (req) => {
    if (req.url().includes("/api/admin/")) adminRequests.push({ url: req.url(), headers: req.headers() });
  });
  page.on("pageerror", (err) => console.error("pageerror:", err.message));
  page.on("console", (msg) => { if (msg.type() === "error") console.error("console:", msg.text()); });

  if (CLOSED) {
    assert.equal(await status("/api/admin/overview", { "X-Analyst-Key": analystKey }), 503);
    await page.goto(`${BASE}/admin.html`);
    await page.fill("#admKey", analystKey);
    await page.click("#admSigninBtn");
    await page.waitForFunction(() => /closed/.test(document.getElementById("admSigninNote").textContent));
    results.closed_message = (await page.textContent("#admSigninNote")).trim();
    assert.equal(await page.isVisible("#admConsole"), false);
    await shot(page, "07-console-closed-503.png");
    console.log(JSON.stringify(results, null, 2));
    process.exit(0);
  }

  // 1. The API refuses anonymous and self-named callers.
  results.api_no_key = await status("/api/admin/overview");
  results.api_claimed_header = await status("/api/admin/overview", { "X-Analyst-Id": investigatorId });
  results.api_claimed_query = await status(`/api/admin/overview?analyst=${investigatorId}`);
  results.api_wrong_key = await status("/api/admin/overview", { "X-Analyst-Key": "not-a-real-key-0123456789abcdef" });
  results.api_analyst_open = await status("/api/admin/incidents/inc_0/open", {
    "X-Analyst-Key": analystKey, "Content-Type": "application/json",
  }, { method: "POST", body: JSON.stringify({ purpose: "pattern_review" }) });
  assert.deepEqual(
    [results.api_no_key, results.api_claimed_header, results.api_claimed_query, results.api_wrong_key, results.api_analyst_open],
    [401, 401, 401, 401, 403],
  );

  // 2. Signed out: only the sign-in panel; the page asks the API for nothing.
  await page.goto(`${BASE}/admin.html`);
  await page.waitForSelector("#admSignin:not([hidden])");
  assert.equal(await page.isVisible("#admConsole"), false, "console hidden before sign-in");
  assert.equal(adminRequests.length, 0, "no /api/admin request before sign-in");
  await shot(page, "01-signin.png");

  await page.fill("#admKey", "definitely-not-a-valid-key-000000");
  await page.click("#admSigninBtn");
  await page.waitForFunction(() => /not accepted/.test(document.getElementById("admSigninNote").textContent));
  results.wrong_key_message = (await page.textContent("#admSigninNote")).trim();
  await shot(page, "02-wrong-key.png");

  // 3. Analyst: overview renders; the open-case control is locked, with the reason.
  await page.fill("#admKey", analystKey);
  await page.click("#admSigninBtn");
  await page.waitForSelector("#admConsole:not([hidden]) .kpi-card");
  await page.waitForSelector("tr[data-org]");
  results.analyst_who = (await page.textContent("#admWho")).replace(/\s+/g, " ").trim();
  assert.ok(results.analyst_who.includes(analystId) && results.analyst_who.includes("analyst"));
  results.analyst_queue_rows = await page.locator("tr[data-org]").count();
  const storage = await page.evaluate(() => ({
    session: sessionStorage.getItem("qorgan.analystKey") !== null,
    local: Object.keys(localStorage).some((k) => /analyst/i.test(k)),
  }));
  assert.deepEqual(storage, { session: true, local: false }, "key in sessionStorage only");
  await shot(page, "03-analyst-overview.png");

  await page.locator("tr[data-org]").first().click();
  await page.waitForSelector('#ddCalls tr[data-incident][data-has-transcript="1"]');
  await page.locator('#ddCalls tr[data-incident][data-has-transcript="1"]').first().click();
  await page.waitForSelector(".dd-open", { timeout: 120_000 });
  results.analyst_open_locked = {
    purpose_disabled: await page.isDisabled(".dd-open-purpose"),
    button_disabled: await page.isDisabled(".dd-open .dd-open-btn"),
    reason: (await page.textContent(".dd-open-fine")).replace(/\s+/g, " ").trim(),
  };
  assert.ok(results.analyst_open_locked.purpose_disabled && results.analyst_open_locked.button_disabled);
  assert.ok(/investigator/.test(results.analyst_open_locked.reason));
  await page.locator(".dd-open").scrollIntoViewIfNeeded();
  await shot(page, "04-analyst-open-locked.png");

  // Every console request carried the key and nothing else claimed an identity.
  const unkeyed = adminRequests.filter((r) => !r.headers["x-analyst-key"] || r.headers["x-analyst-id"]);
  assert.equal(unkeyed.length, 0, `requests without the key or with a claimed id: ${JSON.stringify(unkeyed)}`);
  results.console_requests_with_key = adminRequests.length;

  await page.click("#admModalClose");
  await page.click("#admSignOut");
  await page.waitForSelector("#admSignin:not([hidden])");
  assert.equal(await page.evaluate(() => sessionStorage.getItem("qorgan.analystKey")), null, "sign-out forgets the key");
  results.signout_message = (await page.textContent("#admSigninNote")).trim();

  // 4. Investigator: the button waits for a purpose; opening shows the full transcript.
  await page.fill("#admKey", investigatorKey);
  await page.click("#admSigninBtn");
  await page.waitForSelector("tr[data-org]");
  results.investigator_who = (await page.textContent("#admWho")).replace(/\s+/g, " ").trim();
  await page.locator("tr[data-org]").first().click();
  await page.waitForSelector('#ddCalls tr[data-incident][data-has-transcript="1"]');
  await page.locator('#ddCalls tr[data-incident][data-has-transcript="1"]').first().click();
  await page.waitForSelector(".dd-open", { timeout: 120_000 });
  assert.equal(await page.isDisabled(".dd-open .dd-open-btn"), true, "no purpose, no open");
  await page.selectOption(".dd-open-purpose", "citizen_request");
  await page.fill(".dd-open-note", "hotline ticket follow-up");
  assert.equal(await page.isDisabled(".dd-open .dd-open-btn"), false);
  const openResponse = page.waitForResponse((r) => r.url().includes("/open") && r.request().method() === "POST");
  await page.click(".dd-open .dd-open-btn");
  const opened = await openResponse;
  results.open_status = opened.status();
  results.open_request_body = JSON.parse(opened.request().postData());
  assert.equal(results.open_status, 200);
  await page.waitForFunction(() => /opened by/.test(document.querySelector(".dd-analysis")?.textContent || ""));
  results.opened_label = (await page.locator(".dd-analysis .dd-section-label").first().textContent()).replace(/\s+/g, " ").trim();
  await shot(page, "05-investigator-opened.png");
  await page.click("#admModalClose");

  // 5. A 401 mid-session (key revoked) returns to sign-in with a message.
  await page.route("**/api/admin/overview*", (route) =>
    route.fulfill({ status: 401, contentType: "application/json", body: '{"detail":"missing or invalid analyst key"}' }));
  await page.click("#admRefresh");
  await page.waitForSelector("#admSignin:not([hidden])");
  results.revoked_message = (await page.textContent("#admSigninNote")).trim();
  assert.equal(await page.isVisible("#admConsole"), false);
  await shot(page, "06-revoked-back-to-signin.png");

  console.log(JSON.stringify(results, null, 2));
} finally {
  await browser.close();
}
