import test from "node:test";
import assert from "node:assert/strict";
import { readdirSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { REPO } from "./helpers.mjs";

// The PWA promise is "offline after first load" (PLAN B4): every on-device core module the
// pages can import must be in the service worker's precached shell.
test("every site/core module is precached by the service worker", () => {
  const sw = readFileSync(join(REPO, "site", "sw.js"), "utf8");
  const shell = new Set([...sw.matchAll(/"(\/[^"]+)"/g)].map((m) => m[1]));
  const core = readdirSync(join(REPO, "site", "core"))
    .filter((name) => /\.(js|json)$/.test(name))
    .map((name) => `/core/${name}`);
  const missing = core.filter((path) => !shell.has(path));
  assert.deepEqual(missing, [], `add to SHELL in site/sw.js: ${missing.join(", ")}`);
});

// A module a precached page script imports statically must be precached too, or the page
// breaks offline (live.js imports i18n.js and core modules at load time).
test("every module a precached page script imports is precached", () => {
  const sw = readFileSync(join(REPO, "site", "sw.js"), "utf8");
  const shell = new Set([...sw.matchAll(/"(\/[^"]+)"/g)].map((m) => m[1]));
  const pages = [...shell].filter((path) => /^\/[\w-]+\.js$/.test(path) && path !== "/sw.js");
  assert.ok(pages.includes("/live.js"));
  const missing = [];
  for (const page of pages) {
    const source = readFileSync(join(REPO, "site", page.slice(1)), "utf8");
    for (const m of source.matchAll(/(?:^|\n)\s*import\s[^;]*?from\s+"\.\/([^"]+)"/g)) {
      if (!shell.has(`/${m[1]}`)) missing.push(`${page} -> /${m[1]}`);
    }
  }
  assert.deepEqual(missing, [], `add to SHELL in site/sw.js: ${missing.join(", ")}`);
});
