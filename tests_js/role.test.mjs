/* Layer 2 role slot (ADR D63): the JS port must make exactly Python's choice.
   Fixture: `python scripts/export_role_fixture.py`. */
import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";

import { detectRoles, roleClaim } from "../site/core/role.js";
import { explain } from "../site/core/explain.js";
import { config } from "./helpers.mjs";

const fixture = JSON.parse(readFileSync(new URL("./fixtures/roles.json", import.meta.url), "utf8"));

test("fixture is non-trivial", () => {
  assert.ok(fixture.cases.length >= 15);
  assert.ok(fixture.cases.filter((c) => c.claims.length).length >= 10);
  assert.ok(fixture.cases.some((c) => !c.claims.length));
});

test("every claim (id + span offsets) matches Python", () => {
  for (const c of fixture.cases) {
    assert.deepEqual(detectRoles(c.transcript, config.roles, config.role_intro), c.claims, c.transcript.slice(0, 60));
  }
});

test("the rendered role line matches Python in every locale", () => {
  for (const c of fixture.cases) {
    for (const locale of ["ru", "kk", "en"]) {
      assert.equal(roleClaim(c.transcript, locale, config).role_line, c.lines[locale], `${locale}: ${c.transcript.slice(0, 50)}`);
    }
  }
});

test("explain() carries the role slot without touching the reason", () => {
  const transcript = "Здравствуйте, вас беспокоит служба безопасности банка.";
  const result = { risk: 0.1, tags: [], attributions: [], backend: "linear", raw_confidence: 0.9 };
  const e = explain(result, transcript, "ru", config);
  assert.equal(e.role_id, "bank_security");
  assert.equal(transcript.slice(e.role_span.start, e.role_span.end), e.role_span.text);
  assert.equal(e.reason, config.templates.ru.no_signal_reason);
  assert.equal(explain(result, "Привет, как дела?", "ru", config).role_line, null);
});

test("self-introduction by organisation is exported and matched (D63 phase 2)", () => {
  assert.ok(config.role_intro && config.role_intro.markers.length > 10 && config.role_intro.window > 0);
  const intro = fixture.cases.filter((c) => c.transcript.includes("Асель из Kaspi"));
  assert.equal(intro.length, 1);
  assert.equal(intro[0].claims[0].role_id, "bank_security");
});
