import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { finalizeReport, normalizeReceipt, rejectionKey, REPORT_LIMITS, ReportReviewError, reviewDraft, scrubText } from "../site/core/report.js";
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

test("a flagged phrase that carried PII is sent in its scrubbed form (ADR D66)", () => {
  const transcript =
    "Меня зовут Асет, служба безопасности банка.\nНазовите код из СМС: один два три четыре пять шесть семь восемь.";
  const flagged = [
    "Меня зовут Асет, служба безопасности банка.",
    "Назовите код из СМС: один два три четыре пять шесть семь восемь.",
    "один два три четыре пять шесть семь восемь",
  ];
  const payload = finalizeReport(
    { ...draft, transcript, flagged_phrases: flagged },
    { transcript, phoneNumber: "", includedTacticIds: draft.tactic_ids, consent: true },
  );
  assert.deepEqual(payload.flagged_phrases, [
    "Меня зовут [NAME], служба безопасности банка.",
    "Назовите код из СМС: [NUMBER].",
  ]);
});

test("a phrase whose name needs the words before it is sent masked (ADR D66)", () => {
  const transcript = "Hello, this is Major Ivanov from the Financial Crimes Unit. Do not tell anyone.";
  const payload = finalizeReport(
    { ...draft, transcript, flagged_phrases: ["Major Ivanov from the Financial Crimes Unit", "Do not tell anyone"] },
    { transcript, phoneNumber: "", includedTacticIds: draft.tactic_ids, consent: true },
  );
  assert.deepEqual(payload.flagged_phrases, ["Major [NAME] from the Financial Crimes Unit", "Do not tell anyone"]);
});

test("the caller number is trimmed and optional", () => {
  assert.equal(approve({ phoneNumber: "  +7 700 101 20 30 " }).phone_number, "+7 700 101 20 30");
  assert.equal(approve({ phoneNumber: "  " }).phone_number, null);
});

// A refused report used to read "сервер отклонил сообщение: [object Object]": FastAPI's request
// validation answers `detail` as a list, and the route's own refusals are English.
test("rejectionKey names a refusal in the page's words, whatever shape the server used", () => {
  assert.equal(rejectionKey({ detail: "caller number not understood: expected a phone number with at least 7 digits" }), "report.err_number");
  assert.equal(rejectionKey({ detail: "timestamp is in the future or older than the retention period" }), "report.err_invalid");
  assert.equal(rejectionKey({ detail: [{ type: "string_too_long", loc: ["body", "transcript"], msg: "too long" }] }), "report.err_too_long");
  assert.equal(rejectionKey({ detail: [{ type: "string_too_long", loc: ["body", "phone_number"], msg: "too long" }] }), "report.err_number");
  assert.equal(rejectionKey({ detail: [{ type: "value_error", loc: ["body", "tactic_ids"], msg: "unknown" }] }), "report.err_invalid");
  for (const junk of [null, undefined, {}, { detail: 5 }, { detail: [null] }]) assert.equal(rejectionKey(junk), "report.err_invalid");
});

test("normalizeReceipt accepts what the server issued and tolerates a careless paste", () => {
  assert.equal(normalizeReceipt("0123456789abcdef01234567"), "0123456789abcdef01234567");
  assert.equal(normalizeReceipt("  0123456789ABCDEF01234567\n"), "0123456789abcdef01234567");
  for (const bad of ["", "abc", "0123456789abcdef0123456", "0123456789abcdef012345678", "g123456789abcdef01234567", "../../etc/passwd", null, undefined]) {
    assert.equal(normalizeReceipt(bad), null, String(bad));
  }
});

// A long call flags whole sentences; one over the route's limit made the whole report a 422 that
// no edit of the visible text could fix.
test("finalizeReport never sends more flagged evidence than the server accepts", () => {
  const long = "Назовите код ".repeat(40).trim();
  assert.ok(long.length > REPORT_LIMITS.phraseChars);
  const many = Array.from({ length: REPORT_LIMITS.phrases + 10 }, (_, i) => `фраза номер ${"п".repeat(i + 1)}`);
  const payload = finalizeReport(
    { ...draft, transcript: `${long}\n${many.join("\n")}`, flagged_phrases: [long, ...many] },
    { transcript: `${long}\n${many.join("\n")}`, phoneNumber: "", includedTacticIds: [], consent: true },
  );
  assert.ok(!payload.flagged_phrases.includes(long), "an over-long phrase is dropped, not cut: evidence stays verbatim");
  assert.equal(payload.flagged_phrases.length, REPORT_LIMITS.phrases);
  assert.ok(payload.flagged_phrases.every((phrase) => phrase.length <= REPORT_LIMITS.phraseChars));
});

test("the client's report limits are the route's limits", () => {
  const route = readFileSync(join(REPO, "src", "qorgan", "api_reports.py"), "utf8");
  const limit = (name) => Number(route.match(new RegExp(`^${name} = ([\\d_]+)`, "m"))?.[1].replaceAll("_", ""));
  assert.equal(REPORT_LIMITS.phrases, limit("_MAX_PHRASES"));
  assert.equal(REPORT_LIMITS.phraseChars, limit("_MAX_PHRASE_CHARS"));
});
