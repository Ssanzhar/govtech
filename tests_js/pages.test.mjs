import test from "node:test";
import assert from "node:assert/strict";
import { existsSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { REPO } from "./helpers.mjs";

// Static guarantees of the three pages (site/): what the design/QA pass fixed must stay fixed.
const read = (page) => readFileSync(join(REPO, "site", page), "utf8");
const PAGES = ["index.html", "live.html", "admin.html"];

test("the analyst console keeps its script lockdown", () => {
  const html = read("admin.html");
  const csp = html.match(/http-equiv="Content-Security-Policy" content="([^"]+)"/)?.[1] || "";
  assert.match(csp, /script-src 'self'(;|$)/, "script-src stays 'self' only");
  assert.match(csp, /object-src 'none'/);
  const inline = [...html.matchAll(/<script(?![^>]*\bsrc=)[^>]*>/g)];
  assert.deepEqual(inline.map((m) => m[0]), [], "no inline script on the page that holds a key");
  assert.doesNotMatch(html, /\son[a-z]+="/, "no inline event handlers");
});

// A page that promises the call stays on the device tells no third party it was opened, and a
// slow CDN cannot hold it up (a font CDN stalled the live page for 26 s on bad Wi-Fi, ADR D83).
test("no page, stylesheet or script fetches anything from another origin", () => {
  for (const page of PAGES) {
    const html = read(page);
    const fetched = [...html.matchAll(/<(?:link|script|img|iframe|source)\b[^>]*\b(?:href|src)="([^"]+)"/g)].map((m) => m[1]);
    assert.deepEqual(fetched.filter((url) => /^(https?:)?\/\//.test(url)), [], page);
  }
  for (const sheet of ["styles.css", "landing.css", "live.css", "admin.css"]) {
    assert.doesNotMatch(read(sheet), /url\(["']?(https?:)?\/\//, sheet);
    assert.doesNotMatch(read(sheet), /@import/, sheet);
  }
});

test("every self-hosted font a stylesheet names is in the site", () => {
  const css = read("styles.css");
  const files = [...css.matchAll(/url\("(fonts\/[^"]+)"\)/g)].map((m) => m[1]);
  assert.ok(files.length >= 16, "4 Inter + 12 Plex Mono subsets");
  for (const file of files) assert.ok(existsSync(join(REPO, "site", file)), file);
  for (const licence of ["LICENSE-Inter.txt", "LICENSE-IBM-Plex-Mono.txt"]) assert.ok(existsSync(join(REPO, "site", "fonts", licence)), licence);
});

test("every page declares the one deliberate theme", () => {
  for (const page of PAGES) {
    assert.match(read(page), /<meta name="color-scheme" content="only dark">/, page);
    assert.match(read(page), /<link rel="stylesheet" href="styles.css">/, `${page} loads the shared base`);
  }
  assert.match(read("styles.css"), /color-scheme:only dark/);
});

test("in-page and cross-page anchors point at elements that exist", () => {
  const ids = (page) => new Set([...read(page).matchAll(/\sid="([^"]+)"/g)].map((m) => m[1]));
  for (const page of PAGES) {
    for (const [, target, hash] of read(page).matchAll(/href="([\w-]*\.html)?#([\w-]+)"/g)) {
      const where = target || page;
      assert.ok(ids(where).has(hash), `${page}: href to ${where}#${hash} has no element`);
    }
  }
});

// The Russian in the markup is what a visitor sees before the scripts run. It had drifted: the live
// page still promised three speech models, the landing English calls.
test("the markup's fallback text is the Russian string it stands for", async () => {
  const { STRINGS } = await import("../site/i18n.js");
  const { ADMIN_STRINGS } = await import("../site/i18n-admin.js");
  const known = { ...STRINGS.ru, ...ADMIN_STRINGS.ru };
  const stale = [];
  for (const page of PAGES) {
    for (const [, , , key, inner] of read(page).matchAll(/<([a-z0-9]+)\b[^>]*?\bdata-i18n(-html)?="([^"]+)"[^>]*>([\s\S]*?)<\/\1>/g)) {
      const want = known[key];
      if (want !== undefined && !want.includes("{") && inner !== want) stale.push(`${page}: ${key}`);
    }
  }
  assert.deepEqual(stale, []);
});

test("the landing sends citizens to the citizen page", () => {
  const html = read("index.html");
  assert.ok((html.match(/href="live\.html" data-i18n="landing\.cta_check"/g) || []).length >= 2, "header + hero call to action");
});

// The language is applied on load too. Before sign-in that must not load the console: a keyless
// request is a 401, which the console reads as a revoked key ("Your key is no longer accepted") on
// a first visit.
test("a language change reloads the console only for a signed-in analyst", () => {
  const js = read("admin.js");
  const apply = js.match(/const applyLanguage = \(next[\s\S]*?\n {2}\};\n/)?.[0] || "";
  assert.ok(apply, "the language wiring exists");
  assert.match(apply, /if \(me\) \{\s*showConsole\(\);\s*load\(\);/);
});

// Loads overlap (sign-in completing while the stored language is applied): only the newest
// may render, or a slower answer in the previous language overwrites the console.
test("an overtaken console load never renders", () => {
  const js = read("admin.js");
  const load = js.match(/const load = async \(\) => \{[\s\S]*?\n {2}\};\n/)?.[0] || "";
  assert.match(load, /const seq = \+\+loadSeq;/);
  assert.match(load, /if \(stale\(\)\) return;\n\n {6}if \(!body\.available\)/, "checked before the first render");
  assert.match(load, /loadStats\(stale\)/);
});

// Open demo access (ADR D59): without a stored key the console asks the server first and
// skips the sign-in only when the server itself says it is open.
test("a keyless visitor gets the console only when the server reports open access", () => {
  const js = read("admin.js");
  assert.match(js, /else tryOpenAccess\(\);/);
  const probe = js.match(/const tryOpenAccess = async \(\) => \{[\s\S]*?\n {2}\};\n/)?.[0] || "";
  assert.match(probe, /if \(!body\?\.open_access\) \{\s*showSignin\(\);/);
  assert.doesNotMatch(probe, /X-Analyst-Key/, "the probe carries no key");
});

// ADR D62: qorgan-config.json is written with sorted keys, so `asr.models` arrives as en, kk, ru.
// The page must build the recognisers in `asr.order` (kk, ru, en): the first language is the
// initial preference for ties and the first partial shown.
test("the live page builds its recognisers in the configured vote order", () => {
  const js = read("live.js");
  const specs = js.match(/const modelSpecs = \(cfg\) => \{[\s\S]*?\n\};\n/)?.[0] || "";
  assert.match(specs, /cfg\.asr\?\.order/);
  const config = JSON.parse(read("core/qorgan-config.json"));
  assert.deepEqual(config.asr.order, ["kk", "ru"]); // kk/ru only (ADR D70)
});
