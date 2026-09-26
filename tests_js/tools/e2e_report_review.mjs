/* End-to-end check of the report review (ADR D44) in real Chromium against a running
   server: replay the RU scam scene on-device, open the review, edit, consent, send.

   Run: python -m qorgan.api  (QORGAN_NUMBER_HMAC_KEY set), then (QORGAN_E2E_CHANNEL=chrome
   uses the installed Chrome instead of Playwright's Chromium)
        node tests_js/tools/e2e_report_review.mjs [--base http://127.0.0.1:8000] [--shot out.png]

   Asserts: no request carries call content before Send; Send is disabled until consent;
   the POSTed body is exactly the edited, consented draft; the receipt can delete it. */

import assert from "node:assert/strict";
import { chromium } from "playwright";

const arg = (name, fallback) => {
  const i = process.argv.indexOf(name);
  return i > -1 ? process.argv[i + 1] : fallback;
};
const BASE = arg("--base", "http://127.0.0.1:8000");
const SHOT = arg("--shot", null);

const browser = await chromium.launch(process.env.QORGAN_E2E_CHANNEL ? { channel: process.env.QORGAN_E2E_CHANNEL } : {});
const page = await browser.newPage({ viewport: { width: 1280, height: 900 } });
const contentRequests = [];
page.on("request", (req) => {
  if (req.method() !== "GET") contentRequests.push({ method: req.method(), url: req.url(), body: req.postData() });
});
page.on("pageerror", (err) => console.error("pageerror:", err.message));

try {
  await page.goto(`${BASE}/live.html`);
  await page.waitForFunction(() => document.querySelectorAll("#lvScenario option").length > 1);
  await page.selectOption("#lvScenario", "live_scam_bank_ru");
  await page.click("#lvStart");
  await page.waitForSelector("#lvReportOpen", { timeout: 180_000 }); // model load + 6 turns
  const score = await page.textContent(".lv-summary-score");
  console.log(`scene: live_scam_bank_ru -> ${score.trim()}`);
  assert.equal(contentRequests.length, 0, "no content-carrying request before the citizen acts");

  await page.click("#lvReportOpen");
  await page.waitForSelector("#lvReportText");
  const original = await page.inputValue("#lvReportText");
  const lines = original.split("\n");
  assert.ok(lines.length > 2, "the draft carries the transcript for review");
  const edited = [...lines.slice(0, -1), "Мой номер +7 777 123 45 67"].join("\n");
  await page.fill("#lvReportText", edited);
  const preview = await page.textContent("#lvReportPreview");
  assert.ok(preview.includes("[PHONE]") && !preview.includes("777 123"), "preview shows the redaction");

  const tactics = await page.$$eval('input[name="lvReportTactic"]', (els) => els.map((el) => el.value));
  assert.ok(tactics.length >= 2, `expected >= 2 detected tactics, got ${tactics}`);
  const dropped = tactics[tactics.length - 1];
  await page.uncheck(`input[name="lvReportTactic"][value="${dropped}"]`);
  await page.fill("#lvReportPhone", "+7 700 101 20 30");

  assert.equal(await page.isDisabled("#lvReportSend"), true, "Send is disabled until consent");
  assert.equal(await page.isVisible("#lvReportOpen"), false, "the Review button is gone once the review is open");
  const dimmed = await page.$eval("#lvReportSend", (el) => Number(getComputedStyle(el).opacity));
  assert.ok(dimmed < 0.5, `a disabled Send must look disabled (opacity ${dimmed})`);
  assert.equal(contentRequests.length, 0, "editing sends nothing");
  if (SHOT) await page.screenshot({ path: SHOT.replace(/\.png$/, "-review.png"), fullPage: true });
  await page.check("#lvReportConsent");
  assert.equal(await page.isDisabled("#lvReportSend"), false);

  await page.click("#lvReportSend");
  await page.waitForSelector("#lvReportDelete", { timeout: 30_000 });
  assert.equal(contentRequests.length, 1, "exactly one request, on Send");
  const sent = JSON.parse(contentRequests[0].body);
  assert.equal(contentRequests[0].url, `${BASE}/api/reports`);
  assert.equal(sent.consent, true);
  assert.match(sent.consent_version, /^[a-z][a-z0-9-]*-v\d+$/, "the ticked consent wording is identified");
  assert.equal(sent.transcript, edited.trim());
  assert.ok(!sent.tactic_ids.includes(dropped), "the unticked tactic is not sent");
  assert.deepEqual(sent.tactic_ids, tactics.filter((t) => t !== dropped));
  assert.ok(sent.flagged_phrases.every((p) => edited.includes(p)), "flagged phrases stay grounded");
  const stored = await page.textContent(".lv-report-note.is-success");
  assert.ok(stored.includes("+7 700 ***") && !stored.includes("101 20 30"), "number stored as prefix only");
  if (SHOT) await page.screenshot({ path: SHOT, fullPage: true });

  await page.click("#lvReportDelete");
  await page.waitForSelector('#lvReportNote[data-state="deleted"]'); // language-independent hook (live page is kk/ru/en)
  console.log(`report review e2e: OK (dropped ${dropped}; ${sent.flagged_phrases.length} grounded phrases sent)`);
} finally {
  await browser.close();
}
