/* End-to-end check of the citizen page's smaller promises in real Chrome against a running server:
   stopping a replay, editing an example, copying the summary and the receipt, deleting a report
   later by its receipt, refusals in the page's words, and a refused microphone costing no download.

   Run: python -m qorgan.api (QORGAN_NUMBER_HMAC_KEY set; ideally a scratch QORGAN_DATA_DIR), then
        QORGAN_E2E_CHANNEL=chrome node tests_js/tools/e2e_citizen_flows.mjs [--base http://127.0.0.1:8000] */

import assert from "node:assert/strict";
import { chromium } from "playwright";
import { STRINGS, t } from "../../site/i18n.js";

const arg = (name, fallback) => {
  const i = process.argv.indexOf(name);
  return i > -1 ? process.argv[i + 1] : fallback;
};
const BASE = arg("--base", "http://127.0.0.1:8000");
const ru = STRINGS.ru;

const browser = await chromium.launch({
  ...(process.env.QORGAN_E2E_CHANNEL ? { channel: process.env.QORGAN_E2E_CHANNEL } : {}),
  args: ["--use-fake-device-for-media-stream"],
});
const open = async (permissions = []) => {
  const context = await browser.newContext({ locale: "ru-RU", viewport: { width: 1280, height: 900 } });
  if (permissions.length) await context.grantPermissions(permissions, { origin: BASE });
  const page = await context.newPage();
  page.on("pageerror", (err) => { throw new Error(`pageerror: ${err.message}`); });
  await page.goto(`${BASE}/live.html`);
  await page.waitForFunction(() => document.querySelectorAll("#lvScenario option").length > 1);
  return { context, page };
};
const text = async (page, selector) => (await page.textContent(selector)).trim();

try {
  // 1. An edited example is "your own text"; a chosen one is the example again.
  {
    const { context, page } = await open();
    await page.selectOption("#lvScenario", "live_scam_bank_ru");
    await page.fill("#lvScript", "Своя фраза");
    assert.equal(await page.inputValue("#lvScenario"), "", "an edited example reads as your own text");
    await page.selectOption("#lvScenario", "live_scam_bank_ru");
    assert.ok((await page.inputValue("#lvScript")).startsWith("Алло"));
    console.log("1. editing an example: OK");
    await context.close();
  }

  // 2. Stop mid-replay: the call so far is summed up; the mode chips were locked meanwhile.
  {
    const { context, page } = await open();
    await page.selectOption("#lvScenario", "live_scam_bank_ru");
    assert.equal(await page.isDisabled('input[name="lv-mode"][value="mic"]'), false);
    await page.click("#lvStart");
    await page.waitForFunction(() => document.querySelectorAll("#lvTranscript .lv-line").length >= 2, null, { timeout: 180_000 });
    assert.equal(await page.isDisabled('input[name="lv-mode"][value="mic"]'), true, "no second check while one runs");
    await page.click("#lvStop");
    await page.waitForSelector("#lvReportOpen");
    const lines = await page.locator("#lvTranscript .lv-line").count();
    assert.ok(lines >= 2 && lines < 6, `stopped early, ${lines} of 6 phrases`);
    assert.equal(await page.isVisible("#lvStop"), false);
    assert.equal(await page.isDisabled("#lvStart"), false);
    assert.equal(await page.isDisabled('input[name="lv-mode"][value="mic"]'), false);
    console.log(`2. stop mid-replay after ${lines} phrases: OK`);
    await context.close();
  }

  // 3. The summary copies as text with no words of the call; the report round trip; a refusal in
  //    the page's words; the receipt copies and deletes the report from a fresh page.
  {
    const { context, page } = await open(["clipboard-read", "clipboard-write"]);
    const clipboard = () => page.evaluate(() => navigator.clipboard.readText());
    await page.selectOption("#lvScenario", "live_scam_bank_ru");
    const scene = await page.inputValue("#lvScript");
    await page.click("#lvStart");
    await page.waitForSelector("#lvReportOpen", { timeout: 180_000 });

    await page.click("#lvSummaryCopy");
    await page.waitForFunction((label) => document.querySelector("#lvSummaryCopy").textContent === label, ru["summary.copied"]);
    const summary = await clipboard();
    assert.ok(summary.includes(t("ru", "summary.copy_score", { score: 98, band: { $: "band.critical" } })), summary);
    assert.ok(summary.includes(ru["summary.copy_note"]));
    for (const line of scene.split("\n")) assert.ok(!summary.includes(line), `the call's words stay out of the summary: ${line}`);

    await page.click("#lvReportOpen");
    await page.waitForSelector("#lvReportText");
    await page.fill("#lvReportPhone", "не номер");
    await page.check("#lvReportConsent");
    await page.click("#lvReportSend");
    await page.waitForSelector('#lvReportNote[data-state="error"]');
    const refusal = await text(page, "#lvReportNote");
    assert.equal(refusal, t("ru", "report.send_failed", { error: { $: "report.err_number" } }));
    assert.ok(!refusal.includes("[object"), refusal);

    await page.fill("#lvReportPhone", "+7 700 101 20 30");
    await page.click("#lvReportSend");
    await page.waitForSelector("#lvReportCopy");
    await page.click("#lvReportCopy");
    await page.waitForFunction((label) => document.querySelector("#lvReportCopy").textContent === label, ru["summary.copied"]);
    const receipt = await clipboard();
    assert.match(receipt, /^[0-9a-f]{24}$/);

    await page.goto(`${BASE}/live.html`);
    await page.click("#lvForget summary");
    await page.fill("#lvForgetReceipt", "не квитанция");
    await page.click("#lvForgetSend");
    await page.waitForSelector('#lvForgetNote[data-state="error"]');
    assert.equal(await text(page, "#lvForgetNote"), ru["forget.invalid"]);

    await page.fill("#lvForgetReceipt", "0".repeat(24));
    await page.click("#lvForgetSend");
    await page.waitForFunction((want) => document.querySelector("#lvForgetNote").textContent.trim() === want, t("ru", "report.delete_failed", { error: { $: "forget.not_found" } }));

    await page.fill("#lvForgetReceipt", `  ${receipt.toUpperCase()} `);
    await page.click("#lvForgetSend");
    await page.waitForSelector('#lvForgetNote[data-state="deleted"]');
    assert.equal(await text(page, "#lvForgetNote"), ru["report.deleted"]);
    assert.equal(await page.inputValue("#lvForgetReceipt"), "");

    await page.fill("#lvForgetReceipt", receipt);
    await page.click("#lvForgetSend");
    await page.waitForFunction((want) => document.querySelector("#lvForgetNote").textContent.trim() === want, t("ru", "report.delete_failed", { error: { $: "forget.not_found" } }));
    console.log("3. copy, refusal, send, delete by receipt from a new page: OK");
    await context.close();
  }

  // 4. A refused microphone is said in the page's words, before any model is downloaded.
  {
    const { context, page } = await open();
    const heavy = [];
    page.on("request", (req) => {
      if (/\/models\/.+\.(onnx|gz)$/.test(new URL(req.url()).pathname)) heavy.push(req.url());
    });
    await page.locator('label:has(input[name="lv-mode"][value="mic"])').click();
    await page.click("#lvMicStart");
    await page.waitForFunction(() => document.querySelector("#lvMicNote").classList.contains("is-error"));
    assert.equal(await text(page, "#lvMicNote"), t("ru", "mic.start_failed", { error: { $: "mic.err_denied" } }));
    assert.deepEqual(heavy, [], "a refusal costs no download");
    assert.equal(await page.isDisabled("#lvMicStart"), false, "the citizen can try again");
    console.log("4. refused microphone: OK");
    await context.close();
  }
  console.log("citizen flows e2e: OK");
} finally {
  await browser.close();
}
