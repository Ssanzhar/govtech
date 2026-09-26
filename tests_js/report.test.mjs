import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { finalizeReport, ReportReviewError, reviewDraft, scrubText } from "../site/core/report.js";
import { REPO } from "./helpers.mjs";

const scrubFixture = JSON.parse(readFileSync(join(REPO, "tests_js", "fixtures", "scrub.json"), "utf8"));

test("scrubText reproduces Python's scrub_text on every fixture case (1:1)", () => {
  for (const { input, expected } of scrubFixture.cases) {
    assert.equal(scrubText(input), expected, JSON.stringify(input));
  }
});

test("scrubText is idempotent", () => {
  for (const { input } of scrubFixture.cases) {
    assert.equal(scrubText(scrubText(input)), scrubText(input));
  }
});

const draft = Object.freeze({
  phone_number: null,
  transcript: "Это служба безопасности банка.\nНазовите код из SMS.\nМой номер +7 777 123 45 67.",
  flagged_phrases: ["служба безопасности банка", "Назовите код из SMS"],
  tactic_ids: ["impersonation_bank", "otp_request"],
  timestamp: "2026-09-25T10:00:00.000Z",
  risk_score: 88,
});

const approve = (overrides = {}) =>
  finalizeReport(draft, {
    transcript: draft.transcript,
    phoneNumber: "",
    includedTacticIds: draft.tactic_ids,
    consent: true,
    ...overrides,
  });

test("reviewDraft exposes every field for editing, all tactics included", () => {
  const view = reviewDraft(draft);
  assert.equal(view.transcript, draft.transcript);
  assert.equal(view.phoneNumber, "");
  assert.deepEqual(view.tactics, [
    { id: "impersonation_bank", included: true },
    { id: "otp_request", included: true },
  ]);
});

test("no consent, no payload", () => {
  assert.throws(() => approve({ consent: false }), ReportReviewError);
  assert.throws(() => approve({ consent: "yes" }), ReportReviewError);
});

test("an empty edited transcript is refused", () => {
  assert.throws(() => approve({ transcript: "   \n " }), ReportReviewError);
});

test("the unedited draft goes out as-is, with explicit consent", () => {
  const payload = approve();
  assert.equal(payload.consent, true);
  assert.equal(payload.transcript, draft.transcript);
  assert.deepEqual(payload.flagged_phrases, draft.flagged_phrases);
  assert.deepEqual(payload.tactic_ids, draft.tactic_ids);
  assert.equal(payload.phone_number, null);
  assert.equal(payload.risk_score, 88);
});

test("unticked tactics are dropped; tactics cannot be added", () => {
  const payload = approve({ includedTacticIds: ["otp_request", "safe_account"] });
  assert.deepEqual(payload.tactic_ids, ["otp_request"]);
});

test("a flagged phrase the citizen deleted from the transcript is not sent", () => {
  const payload = approve({ transcript: "Это служба безопасности банка." });
  assert.deepEqual(payload.flagged_phrases, ["служба безопасности банка"]);
});

test("the caller number is trimmed and optional", () => {
  assert.equal(approve({ phoneNumber: "  +7 700 101 20 30 " }).phone_number, "+7 700 101 20 30");
  assert.equal(approve({ phoneNumber: "  " }).phone_number, null);
});
