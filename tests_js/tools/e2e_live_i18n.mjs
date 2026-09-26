/* End-to-end check of the live page's language control in real Chrome against a running
   server: every visible string follows the chosen language, a switch mid-call re-renders
   what is already on screen without restarting the call or losing the citizen's edits, and
   nothing carrying call content leaves the page before Send.

   Run: python -m qorgan.api, then (QORGAN_E2E_CHANNEL=chrome uses the installed Chrome)
        node tests_js/tools/e2e_live_i18n.mjs [--base http://127.0.0.1:8000] [--shots DIR]

   Scenes: (A) ru desktop -> RU scam replay -> kk mid-call -> summary -> review edits survive
   kk/ru/en switches; (B) fresh kk-KZ phone at 360 px -> KK scam replay -> review, no
   horizontal overflow; (C) a stored preference beats the browser language. */

import assert from "node:assert/strict";
import { mkdirSync, readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { chromium } from "playwright";
import { STRINGS, t } from "../../site/i18n.js";

const REPO = join(dirname(fileURLToPath(import.meta.url)), "..", "..");
const config = JSON.parse(readFileSync(join(REPO, "site", "core", "qorgan-config.json"), "utf8"));
const { scenarios } = JSON.parse(readFileSync(join(REPO, "site", "core", "scenarios.json"), "utf8"));

const arg = (name, fallback) => {
  const i = process.argv.indexOf(name);
  return i > -1 ? process.argv[i + 1] : fallback;
};
const BASE = arg("--base", "http://127.0.0.1:8000");
const SHOTS = arg("--shots", null);
if (SHOTS) mkdirSync(SHOTS, { recursive: true });
const shot = (name) => (SHOTS ? join(SHOTS, name) : null);
const MODEL_MB = 300;

const adviceOf = (locale) => config.advice[locale].tactic_advice;
const tacticNames = (locale) => new Set(config.taxonomy.tactics.map((x) => x.names[locale]));
const bandOf = async (page, selector) => (await page.getAttribute(selector, "class")).match(/(?:^|\s)band-(low|medium|high|critical)(?:\s|$)/)[1];
const pill = (locale, band) => t(locale, "band.pill", { band: { $: `band.${band}` }, desc: { $: `band.${band}_desc` } });

// Latin words that legitimately appear in a ru/kk page: the brand and terms that page's own
// reviewed sources use (SMS, CVV, AnyDesk, Wi-Fi, …). Anything else is leaked English.
const latinTokens = (text) => text.match(/[A-Za-z]{3,}/g) || [];
const allowedLatin = (locale) => new Set([
  "Qor", "ğan", "qor", ...Object.values(STRINGS[locale]).flatMap(latinTokens),
  ...latinTokens(JSON.stringify([config.advice[locale], config.templates[locale], [...tacticNames(locale)]])),
]);

/** Visible text outside call content (transcript, textareas, stored text) and the language switch. */
const visibleChrome = (page) =>
  page.evaluate(() => {
    const skip = ".lv-transcript, .lv-partial, textarea, .lv-stored, .lang-switch, .brand, script, style";
    const out = [];
    const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
    for (let node = walker.nextNode(); node; node = walker.nextNode()) {
      const el = node.parentElement;
      const text = node.textContent.trim();
      if (!text || !el || el.closest(skip)) continue;
      if (!el.checkVisibility({ visibilityProperty: true, opacityProperty: true })) continue;
      out.push(text);
    }
    for (const el of document.querySelectorAll("input[placeholder], textarea[placeholder]")) {
      if (el.checkVisibility() && !el.value) out.push(el.placeholder);
    }
    return out;
  });

const assertNoEnglish = async (page, where) => {
  const locale = await page.getAttribute("html", "lang");
  const allowed = allowedLatin(locale);
  const leaked = (await visibleChrome(page)).flatMap((text) => latinTokens(text).filter((w) => !allowed.has(w)).map((w) => `${w} (in "${text.slice(0, 60)}")`));
  assert.deepEqual(leaked, [], `${where}: English left on a ${locale} page`);
};

/** Elements poking out to the right of the viewport (the page itself hides overflow-x). */
const overflowing = (page) =>
  page.evaluate(() => {
    const width = document.documentElement.clientWidth;
    return [...document.querySelectorAll("body *")]
      .filter((el) => el.checkVisibility() && !el.closest(".rails"))
      .map((el) => ({ el, r: el.getBoundingClientRect() }))
      .filter(({ r }) => r.width > 0 && r.right > width + 1)
      .map(({ el, r }) => `${el.tagName.toLowerCase()}${el.id ? `#${el.id}` : ""}.${[...el.classList].join(".")} right=${Math.round(r.right)}`);
  });

const browser = await chromium.launch(process.env.QORGAN_E2E_CHANNEL ? { channel: process.env.QORGAN_E2E_CHANNEL } : {});
const errors = [];

/** A fresh context that records every request, so "nothing left before Send" is checkable. */
async function open(options) {
  const context = await browser.newContext(options);
  const requests = [];
  context.on("request", (req) => requests.push({ method: req.method(), url: req.url(), body: req.postData() }));
  const page = await context.newPage();
  page.on("pageerror", (err) => errors.push(err.message));
  await page.goto(`${BASE}/live.html`);
  await page.waitForFunction(() => document.querySelectorAll("#lvScenario option").length > 1);
  return { context, page, requests };
}

/** No request carries the call: nothing but GETs, and no URL contains a fragment of it. */
const assertNoContentLeft = (requests, lines, where) => {
  const fragments = lines.map((line) => line.slice(0, 14));
  const leaks = requests.filter(
    (r) => r.method !== "GET" || fragments.some((f) => r.url.includes(f) || r.url.includes(encodeURIComponent(f)))
  );
  assert.deepEqual(leaks.map((r) => `${r.method} ${r.url}`), [], `${where}: a request carried call content`);
};

const lang = (page) => page.getAttribute("html", "lang");
const setLang = (page, value) => page.check(`input[name="lv-lang"][value="${value}"]`);
const scene = (id) => scenarios.find((s) => s.id === id);

try {
  // ── A: Russian desktop, switch to Kazakh mid-call ───────────────────────────
  {
    const { context, page, requests } = await open({ locale: "ru-RU", viewport: { width: 1280, height: 900 } });
    const lines = scene("live_scam_bank_ru").lines;
    assert.equal(await lang(page), "ru");
    assert.equal((await page.textContent("#lvStart")).trim(), STRINGS.ru["replay.start"]);
    assert.equal((await page.textContent("#lvDlReplay")).trim(), t("ru", "download.model_first", { mb: MODEL_MB }));
    assert.equal(requests.filter((r) => r.url.includes("/models/")).length, 0, "nothing is downloaded before the citizen starts");
    if (SHOTS) await page.locator("#lvSetupReplay").screenshot({ path: shot("disclosure-ru.png") });
    await assertNoEnglish(page, "A/load-ru");

    await page.selectOption("#lvScenario", "live_scam_bank_ru");
    await page.click("#lvStart");
    await page.waitForSelector("#lvAdvice:not([hidden])", { timeout: 180_000 });
    assert.ok(requests.some((r) => r.url.includes("/models/")), "the model download starts only after Start");
    const ruAdvice = (await page.textContent("#lvAdviceText")).trim();
    const tacticId = Object.entries(adviceOf("ru")).find(([, text]) => text === ruAdvice)?.[0];
    assert.ok(tacticId, `advice is a reviewed ru template: ${ruAdvice}`);
    const turnsBefore = await page.locator("#lvTranscript .lv-line").count();
    await assertNoEnglish(page, "A/mid-call-ru");
    if (SHOTS) await page.screenshot({ path: shot("ru-desktop.png"), fullPage: true });

    await setLang(page, "kk");
    assert.equal(await page.isHidden("#lvSummaryWrap"), true, "the switch happened mid-call");
    assert.equal(await lang(page), "kk");
    assert.equal((await page.textContent("#lvAdviceText")).trim(), adviceOf("kk")[tacticId], "advice on screen re-rendered in Kazakh");
    assert.equal((await page.textContent("#lvBandPill")).trim(), pill("kk", await bandOf(page, "#lvBandPill")));
    const tags = await page.$$eval("#lvTags .tw-tag", (els) => els.map((el) => el.textContent.trim()));
    assert.ok(tags.length > 0 && tags.every((name) => tacticNames("kk").has(name)), `tactic names in Kazakh: ${tags}`);
    assert.equal((await page.textContent("#lvStart")).trim(), STRINGS.kk["replay.running"]);
    assert.equal(await page.getAttribute("#lvMeter", "aria-valuetext").then((v) => v.startsWith("100-ден")), true);
    console.log(`A: switched ru -> kk at turn ${turnsBefore}/${lines.length}; advice [${tacticId}] now Kazakh`);

    await page.waitForSelector("#lvReportOpen", { timeout: 60_000 });
    assert.equal(await page.locator("#lvTranscript .lv-line").count(), lines.length, "the call ran to the end, not restarted");
    assert.equal((await page.textContent(".lv-summary-eyebrow")).trim(), STRINGS.kk["summary.eyebrow"]);
    assert.ok((await page.textContent(".lv-summary-note")).includes(config.templates.kk.human_note));
    assert.ok((await page.textContent(".lv-summary-reason")).startsWith(config.templates.kk.reason_template.split("{tags}")[0]));
    const actions = await page.$$eval(".lv-summary-actions li", (els) => els.map((el) => el.textContent.trim()));
    assert.ok(actions.length > 0 && actions.every((a) => Object.values(adviceOf("kk")).includes(a)), "summary advice is Kazakh");
    assert.equal((await page.textContent("#lvReportOpen")).trim(), STRINGS.kk["report.open"]);
    await assertNoEnglish(page, "A/summary-kk");
    if (SHOTS) await page.screenshot({ path: shot("kk-desktop.png"), fullPage: true });

    // The review: edits and consent survive language switches.
    await page.click("#lvReportOpen");
    await page.waitForSelector("#lvReportText");
    assert.equal((await page.textContent('label[for="lvReportText"]')).trim(), STRINGS.kk["report.text_label"]);
    const edited = `${await page.inputValue("#lvReportText")}\nМен трубканы қойдым.`;
    await page.fill("#lvReportText", edited);
    await page.fill("#lvReportPhone", "+7 700 101 20 30");
    const chips = await page.$$eval('input[name="lvReportTactic"]', (els) => els.map((el) => el.value));
    await page.uncheck(`input[name="lvReportTactic"][value="${chips[0]}"]`);
    await page.check("#lvReportConsent");
    await assertNoEnglish(page, "A/review-kk");

    await setLang(page, "ru");
    assert.equal((await page.textContent('label[for="lvReportText"]')).trim(), STRINGS.ru["report.text_label"]);
    assert.equal((await page.textContent("#lvReportSend")).trim(), STRINGS.ru["report.send"]);
    const ruChips = await page.$$eval(".lv-report-tactic span", (els) => els.map((el) => el.textContent.trim()));
    assert.ok(ruChips.every((name) => tacticNames("ru").has(name)), `review chips in Russian: ${ruChips}`);
    assert.equal(await page.inputValue("#lvReportText"), edited, "the edited transcript survives the switch");
    assert.equal(await page.inputValue("#lvReportPhone"), "+7 700 101 20 30");
    assert.equal(await page.isChecked(`input[name="lvReportTactic"][value="${chips[0]}"]`), false);
    assert.equal(await page.isChecked("#lvReportConsent"), true);
    assert.equal(await page.isDisabled("#lvReportSend"), false);

    await setLang(page, "en"); // English chrome, reviewed Russian content, and it says so
    assert.equal(await lang(page), "en");
    assert.equal((await page.textContent(".lv-summary-eyebrow")).trim(), STRINGS.en["summary.eyebrow"]);
    assert.ok((await page.textContent(".lv-summary-note")).includes(config.templates.ru.human_note));
    assert.equal((await page.textContent("#lvSummaryContent .lv-fallback")).trim(), STRINGS.en["call.content_fallback"]);
    assert.equal(await page.inputValue("#lvReportText"), edited);

    await setLang(page, "kk");
    assertNoContentLeft(requests, lines, "A");
    await page.reload();
    await page.waitForFunction(() => document.querySelectorAll("#lvScenario option").length > 1);
    assert.equal(await lang(page), "kk", "the choice is remembered");
    assert.equal(await page.inputValue("#lvScenario"), "live_scam_bank_kk", "a Kazakh visitor starts on the Kazakh scene");
    await context.close();
    console.log("A: ru -> kk mid-call, summary + review re-rendered, edits kept, en fallback, choice persisted: OK");
  }

  // ── B: fresh Kazakh phone, 360 px ──────────────────────────────────────────
  {
    const { context, page, requests } = await open({
      locale: "kk-KZ", viewport: { width: 360, height: 780 }, deviceScaleFactor: 2, isMobile: true, hasTouch: true,
      userAgent: "Mozilla/5.0 (Linux; Android 14; Pixel 7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/140.0 Mobile Safari/537.36",
    });
    const lines = scene("live_scam_bank_kk").lines;
    assert.equal(await lang(page), "kk", "kk-KZ browser opens in Kazakh");
    assert.equal(await page.title(), STRINGS.kk["meta.title"]);
    assert.equal(await page.inputValue("#lvScenario"), "live_scam_bank_kk");
    assert.equal((await page.textContent("#lvDlReplay")).trim(), t("kk", "download.model_first", { mb: MODEL_MB }));
    assert.equal((await page.textContent("#lvModeNote")).trim(), t("kk", "mic.unavailable", { reasons: [{ $: "mic.reason_phone" }] }));
    await assertNoEnglish(page, "B/load");
    assert.deepEqual(await overflowing(page), [], "B/load: horizontal overflow at 360 px");
    if (SHOTS) await page.locator("#lvSetupReplay").screenshot({ path: shot("disclosure-kk-360.png") });

    await page.click("#lvStart");
    await page.waitForSelector("#lvReportOpen", { timeout: 180_000 });
    await page.click("#lvReportOpen");
    await page.waitForSelector("#lvReportText");
    await assertNoEnglish(page, "B/review");
    assert.deepEqual(await overflowing(page), [], "B/review: horizontal overflow at 360 px");
    assertNoContentLeft(requests, lines, "B");
    if (SHOTS) await page.screenshot({ path: shot("kk-360.png"), fullPage: true });
    await context.close();
    console.log("B: fresh kk-KZ phone at 360 px: all Kazakh, no overflow, nothing sent: OK");
  }

  // ── C: a stored choice beats the browser language ──────────────────────────
  {
    const context = await browser.newContext({ locale: "kk-KZ" });
    await context.addInitScript(() => localStorage.setItem("qorgan.locale", "en"));
    const page = await context.newPage();
    await page.goto(`${BASE}/live.html`);
    assert.equal(await lang(page), "en");
    assert.equal((await page.textContent("#lvStart")).trim(), STRINGS.en["replay.start"]);
    await context.close();
    console.log("C: stored preference wins over navigator.language: OK");
  }

  assert.deepEqual(errors, [], "no page errors");
  console.log("live i18n e2e: OK");
} finally {
  await browser.close();
}
