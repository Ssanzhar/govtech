import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { REPO } from "./helpers.mjs";
import { ADMIN_STRINGS } from "../site/i18n-admin.js";
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
const SAME_IN_KK_AND_RU = new Set(["mode.mic", "band.pill", "call.line_meta_src"]); // pure templates
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

test("every key the landing page uses exists, and it uses the landing keys", () => {
  const html = read("site", "index.html");
  const js = read("site", "try.js") + read("site", "landing.js");
  const used = new Set();
  for (const m of html.matchAll(/data-i18n(?:-html)?="([^"]+)"/g)) used.add(m[1]);
  for (const m of html.matchAll(/data-i18n-attr="([^"]+)"/g)) for (const pair of m[1].split(";")) used.add(pair.split(":")[1]);
  for (const m of js.matchAll(/"((?:landing|lang_name)\.[a-z0-9_]+)"/g)) used.add(m[1]);
  const missing = [...used].filter((k) => !(k in STRINGS[DEFAULT_LOCALE]));
  assert.deepEqual(missing, []);
  const unused = keys(DEFAULT_LOCALE).filter((k) => k.startsWith("landing.") && !used.has(k));
  assert.deepEqual(unused, [], "landing keys nothing renders");
});

test("the landing and the citizen page share one language control contract", () => {
  for (const page of ["index.html", "live.html"]) {
    const html = read("site", page);
    const values = [...html.matchAll(/<input type="radio" name="(?:lp|lv)-lang" value="([a-z]+)"/g)].map((m) => m[1]);
    assert.deepEqual(values, [...LOCALES], `${page}: one radio per locale, in display order`);
  }
  assert.match(read("site", "i18n-dom.js"), /STORAGE_KEY/, "the stored choice is the one key both pages read");
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
  assert.equal(contentLocale("en"), "en", "English content is reviewed YAML now, not a RU alias");
  assert.equal(contentLocale("de"), "ru", "a locale without reviewed content still falls back");
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
    t("ru", "mic.unavailable", { reasons: [{ $: "mic.reason_capture" }, { $: "mic.reason_browser" }] }),
    `Микрофон здесь недоступен: ${STRINGS.ru["mic.reason_capture"]}; ${STRINGS.ru["mic.reason_browser"]}.`
  );
  assert.equal(tHtml("en", "report.sent_html", { receipt: "<x>", prefix: "+7 700 ***" }).includes("<b>&lt;x&gt;</b>"), true);
  assert.equal(t("en", "report.err_status", { status: "<b>" }), "the server answered with error <b>");
  assert.equal(t("kk", "no.such_key"), "no.such_key");
  assert.equal(t("ru", "call.head"), "Звонок · фраз: {n}", "a missing param stays visible, never 'undefined'");
});

// ADR D62: every voice-mode language is named in every locale -- the transcript tags each line
// with `lang_short.<language>` and the loading note says `lang_name.<language>`; a missing key
// rendered as the raw key on the first English line.
test("every voice-mode language has a full and a short name in every locale", () => {
  const config = JSON.parse(readFileSync(join(REPO, "site", "core", "qorgan-config.json"), "utf8"));
  for (const locale of LOCALES) {
    for (const language of config.asr.order) {
      for (const key of [`lang_name.${language}`, `lang_short.${language}`]) {
        assert.ok(STRINGS[locale][key], `${locale}: missing ${key}`);
      }
    }
  }
});

// ── the analyst console's own dictionary (site/i18n-admin.js) ────────────────────────────────

test("the console's strings follow the same conventions", () => {
  const ruKeys = Object.keys(ADMIN_STRINGS.ru).sort();
  assert.ok(ruKeys.every((key) => key.startsWith("admin.")));
  const allowed = /^<\/?(em|b|code)>$/;
  for (const locale of LOCALES) {
    assert.deepEqual(Object.keys(ADMIN_STRINGS[locale]).sort(), ruKeys, locale);
    for (const [key, value] of Object.entries(ADMIN_STRINGS[locale])) {
      assert.ok(value.trim().length > 0 && value === value.trim(), `${locale}:${key}`);
      assert.deepEqual(placeholdersOf(value), placeholdersOf(ADMIN_STRINGS.ru[key]), `${locale}:${key} placeholders`);
      const tags = value.match(/<[^>]*>/g) || [];
      if (!key.endsWith("_html")) assert.deepEqual(tags, [], `${locale}:${key} has markup`);
      for (const tag of tags) assert.match(tag, allowed, `${locale}:${key}: ${tag}`);
    }
  }
  const copied = ruKeys.filter((key) => ADMIN_STRINGS.kk[key] === ADMIN_STRINGS.ru[key] && key !== "admin.backend"); // a technical term
  assert.deepEqual(copied, [], "Kazakh is translated, not copied from Russian");
});

test("every key the console uses exists, and none is unused", () => {
  const html = read("site", "admin.html");
  const js = read("site", "admin.js");
  const used = new Set();
  for (const m of html.matchAll(/data-i18n(?:-html)?="([^"]+)"/g)) used.add(m[1]);
  for (const m of html.matchAll(/data-i18n-attr="([^"]+)"/g)) for (const pair of m[1].split(";")) used.add(pair.split(":")[1]);
  for (const m of js.matchAll(/"(admin\.[a-z0-9_]+)"/g)) used.add(m[1]);
  const known = { ...STRINGS.ru, ...ADMIN_STRINGS.ru };
  assert.deepEqual([...used].filter((key) => !(key in known)), []);
  // Built at run time: the purpose, role and feedback-state codes the server returns.
  const dynamic = /^admin\.(purpose|role|fb_state)_/;
  assert.deepEqual(Object.keys(ADMIN_STRINGS.ru).filter((key) => !used.has(key) && !dynamic.test(key)), [], "strings nothing renders");
});

test("every code the server can return has a console label", () => {
  const purposes = read("src", "qorgan", "audit.py").match(/AccessPurpose = Literal\[([^\]]+)\]/)?.[1].match(/[a-z_]+/g) || [];
  assert.ok(purposes.length >= 3, "the server's purposes are readable");
  for (const purpose of purposes) assert.ok(`admin.purpose_${purpose}` in ADMIN_STRINGS.ru, purpose);
  for (const role of ["analyst", "investigator"]) assert.ok(`admin.role_${role}` in ADMIN_STRINGS.ru, role);
  for (const state of ["confirmed", "dismissed", "merged"]) assert.ok(`admin.fb_state_${state}` in ADMIN_STRINGS.ru, state);
});
