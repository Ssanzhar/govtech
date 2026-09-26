import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { REPO } from "./helpers.mjs";
import {
  CONTENT_LOCALES, DEFAULT_LOCALE, LOCALES, STRINGS, contentLocale, pickLocale, placeholdersOf, t, tHtml,
} from "../site/i18n.js";

const keys = (locale) => Object.keys(STRINGS[locale]).sort();
const read = (...parts) => readFileSync(join(REPO, ...parts), "utf8");

test("every locale has exactly the keys of the default locale", () => {
  assert.deepEqual([...LOCALES].sort(), ["en", "kk", "ru"]);
  for (const locale of LOCALES) {
    const missing = keys(DEFAULT_LOCALE).filter((k) => !(k in STRINGS[locale]));
    const extra = keys(locale).filter((k) => !(k in STRINGS[DEFAULT_LOCALE]));
    assert.deepEqual({ missing, extra }, { missing: [], extra: [] }, locale);
  }
});

test("no string is empty and every placeholder set matches across locales", () => {
  for (const locale of LOCALES) {
    for (const [key, value] of Object.entries(STRINGS[locale])) {
      assert.equal(typeof value, "string", `${locale}:${key}`);
      assert.ok(value.trim().length > 0, `${locale}:${key} is empty`);
      assert.equal(value, value.trim(), `${locale}:${key} has stray whitespace`);
      assert.deepEqual(placeholdersOf(value), placeholdersOf(STRINGS[DEFAULT_LOCALE][key]), `${locale}:${key} placeholders`);
    }
  }
});

test("markup is confined to _html keys and to <em>, <b> and the analyst link", () => {
  const allowed = /^<\/?(em|b)>$|^<a href="admin\.html">$|^<\/a>$/;
  for (const locale of LOCALES) {
    for (const [key, value] of Object.entries(STRINGS[locale])) {
      const tags = value.match(/<[^>]*>/g) || [];
      if (!key.endsWith("_html")) {
        assert.deepEqual(tags, [], `${locale}:${key} has markup but is not an _html key`);
        continue;
      }
      for (const tag of tags) assert.match(tag, allowed, `${locale}:${key}: ${tag}`);
      const opened = tags.filter((tag) => !tag.startsWith("</")).length;
      assert.equal(opened * 2, tags.length, `${locale}:${key} has unbalanced tags`);
    }
  }
});

// A Kazakh value identical to the Russian one is almost always an untranslated copy.
const SAME_IN_KK_AND_RU = new Set(["mode.mic", "band.pill"]);
test("Kazakh is translated, not copied from Russian", () => {
  const copied = keys("kk").filter((k) => STRINGS.kk[k] === STRINGS.ru[k] && !SAME_IN_KK_AND_RU.has(k));
  assert.deepEqual(copied, []);
});

test("every key the live page uses exists", () => {
  const html = read("site", "live.html");
  const js = read("site", "live.js");
  const used = new Set();
  for (const m of html.matchAll(/data-i18n(?:-html)?="([^"]+)"/g)) used.add(m[1]);
  for (const m of html.matchAll(/data-i18n-attr="([^"]+)"/g)) for (const pair of m[1].split(";")) used.add(pair.split(":")[1]);
  // tr("key") / trHtml("key") / bind(el, "key") / { $: "key" } / KEY constants in live.js
  for (const m of js.matchAll(/(?:\btr|\btrHtml|\bbind|\bsetNote)\((?:[\w.]+,\s*)?"([a-z_]+\.[a-z0-9_]+)"/g)) used.add(m[1]);
  for (const m of js.matchAll(/\$: "([a-z_]+\.[a-z0-9_]+)"/g)) used.add(m[1]);
  for (const m of js.matchAll(/"((?:mic|report|replay|download|model|call|band|summary)\.[a-z0-9_]+)"/g)) used.add(m[1]);
  assert.ok(used.size > 60, `expected the page to use most keys, found ${used.size}`);
  const missing = [...used].filter((k) => !(k in STRINGS[DEFAULT_LOCALE]));
  assert.deepEqual(missing, []);
});

test("every demo scenario has a localised label", () => {
  const { scenarios } = JSON.parse(read("site", "core", "scenarios.json"));
  assert.ok(scenarios.length > 0);
  const missing = scenarios.map((s) => `scenario.${s.id}`).filter((k) => !(k in STRINGS[DEFAULT_LOCALE]));
  assert.deepEqual(missing, [], "add a label per locale in site/i18n.js");
});

test("content (advice, tactic names, templates) exists in exactly the content locales", () => {
  const config = JSON.parse(read("site", "core", "qorgan-config.json"));
  assert.deepEqual([...config.locales].sort(), [...CONTENT_LOCALES].sort());
  for (const locale of CONTENT_LOCALES) {
    assert.ok(config.advice[locale] && config.templates[locale], locale);
    for (const tactic of config.taxonomy.tactics) assert.ok(tactic.names[locale], `${tactic.id} ${locale}`);
  }
  assert.equal(contentLocale("kk"), "kk");
  assert.equal(contentLocale("ru"), "ru");
  assert.equal(contentLocale("en"), "ru", "English chrome shows the reviewed Russian content");
});

test("initial locale: stored choice, then kk/ru browser languages, else Russian", () => {
  assert.equal(pickLocale({ stored: "en", languages: ["kk-KZ"] }), "en");
  assert.equal(pickLocale({ stored: "xx", languages: ["kk-KZ", "ru"] }), "kk");
  assert.equal(pickLocale({ stored: null, languages: ["en-US", "ru-RU"] }), "ru");
  assert.equal(pickLocale({ languages: ["kk"] }), "kk");
  assert.equal(pickLocale({ languages: ["en-US", "de"] }), "ru");
  assert.equal(pickLocale({ languages: ["en-US"] }), "ru", "English browsers get Russian unless they choose");
  assert.equal(pickLocale({}), "ru");
  assert.equal(pickLocale({ languages: [undefined, "KK_kz"] }), "kk");
});

test("t() fills params, resolves key params, escapes only in tHtml, and falls back", () => {
  assert.equal(t("ru", "call.head", { n: 3 }), "Звонок · фраз: 3");
  assert.equal(t("kk", "mic.loading_model", { language: { $: "lang_name.kk" } }), "Сөз моделі жүктелуде (қазақ тілі) — тек алғаш рет…");
  assert.equal(
    t("ru", "mic.unavailable", { reasons: [{ $: "mic.reason_capture" }, { $: "mic.reason_phone" }] }),
    `Микрофон здесь недоступен: ${STRINGS.ru["mic.reason_capture"]}; ${STRINGS.ru["mic.reason_phone"]}.`
  );
  assert.equal(tHtml("en", "report.sent_html", { receipt: "<x>", prefix: "+7 700 ***" }).includes("<b>&lt;x&gt;</b>"), true);
  assert.equal(t("en", "report.err_status", { status: "<b>" }), "the server answered with error <b>");
  assert.equal(t("kk", "no.such_key"), "no.such_key");
  assert.equal(t("ru", "call.head"), "Звонок · фраз: {n}", "a missing param stays visible, never 'undefined'");
});
