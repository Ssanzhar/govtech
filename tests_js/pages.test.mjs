import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
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

test("every page declares the one deliberate theme", () => {
  for (const page of PAGES) {
    assert.match(read(page), /<meta name="color-scheme" content="only light">/, page);
  }
  assert.match(read("styles.css"), /color-scheme:only light/);
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

test("the landing sends citizens to the citizen page", () => {
  const html = read("index.html");
  assert.ok((html.match(/href="live\.html" data-i18n="landing\.cta_check"/g) || []).length >= 2, "header + hero call to action");
});

// admin-lang.js applies the stored/browser language on load by firing "change" on the content
// radios. Before sign-in that must not load the console: a keyless request is a 401, which the
// console reads as a revoked key ("Your key is no longer accepted") on a first visit.
test("a content-language change reloads the console only for a signed-in analyst", () => {
  const js = read("admin.js");
  const wiring = js.match(/querySelectorAll\('input\[name="adm-loc"\]'\)[\s\S]*?\);\n/)?.[0] || "";
  assert.ok(wiring, "the adm-loc wiring exists");
  assert.doesNotMatch(wiring, /addEventListener\("change", load\)/, "guard the reload with the session");
  assert.match(wiring, /if \(me\) load\(\)/);
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
