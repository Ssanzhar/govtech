/* Report review -- the citizen sees, edits and explicitly approves exactly what is sent
   (task.md §8: never automatic, review before submission, editable contents; ADR D44).

   `scrubText` is a 1:1 port of `qorgan/data/scrub.py::scrub_text` so the review shows
   what the server will store; the server re-scrubs regardless (it never trusts the
   client). Python's `re` is Unicode-aware (`\w`, `\d`, `\b` cover Cyrillic/Kazakh), JS's
   is not even with the `u` flag, so those classes are spelled out below. Parity is pinned
   by `tests_js/fixtures/scrub.json`, generated from Python. The context-aware rules of ADR D66
   (spoken numbers, split digit groups, names) live in scrub-numbers.js / scrub-names.js. */

import { maskNames } from "./scrub-names.js";
import { maskNumberRuns } from "./scrub-numbers.js";
import { PLACEHOLDERS } from "./scrub-tokens.js";

export { PLACEHOLDERS };

const W = "[\\p{L}\\p{N}_]"; // Python's Unicode \w
const B = `(?:(?<!${W})(?=${W})|(?<=${W})(?!${W}))`; // Python's Unicode \b
const D = "\\p{Nd}"; // Python's Unicode \d
// Card / IIN edges (see scrub.py): only an ASCII letter/underscore or a digit blocks a match,
// so PII glued to a Cyrillic/Kazakh word is still redacted.
const EB = "(?<![A-Za-z_\\p{Nd}])";
const EA = "(?![A-Za-z_\\p{Nd}])";

// Same order as Python: most specific first, so a longer digit run is never half-consumed.
const RULES = [
  [new RegExp(`${B}[\\p{L}\\p{N}_.+\\-]+@[\\p{L}\\p{N}_\\-]+\\.[\\p{L}\\p{N}_.\\-]+${B}`, "gu"), PLACEHOLDERS.email],
  [/(?<![A-Za-z0-9_])[Kk][Zz][0-9]{2}(?: ?[A-Za-z0-9]){16}(?![A-Za-z0-9_])/gu, PLACEHOLDERS.iban],
  [new RegExp(`${EB}(?:${D}{16}|${D}{4}[ \\-]${D}{4}[ \\-]${D}{4}[ \\-]${D}{4})${EA}`, "gu"), PLACEHOLDERS.card],
  [new RegExp(`${EB}${D}{12}${EA}`, "gu"), PLACEHOLDERS.iin],
  [new RegExp(`(?<!${D})(?:\\+7|8|7)[\\s\\-]?\\(?${D}{3}\\)?[\\s\\-]?${D}{3}[\\s\\-]?${D}{2}[\\s\\-]?${D}{2}(?!${D})`, "gu"), PLACEHOLDERS.phone],
  [new RegExp(`(?<![\\p{L}\\p{N}_+])\\+${D}(?:[ \\u00a0().\\-]{0,2}${D}){7,14}(?!${D})`, "gu"), PLACEHOLDERS.phone],
  [new RegExp(`(?<!${D})(?:\\(7${D}{2}\\)[ \\u00a0.\\-]?${D}{3}[ \\u00a0.\\-]?${D}{2}[ \\u00a0.\\-]?${D}{2}|7${D}{2}\\.${D}{3}\\.${D}{2}\\.${D}{2})(?!${D})`, "gu"), PLACEHOLDERS.phone],
];

export function scrubText(text) {
  if (typeof text !== "string") throw new TypeError(`scrubText expects a string, got ${typeof text}`);
  if (!text.trim()) return text;
  const digitRules = RULES.reduce((acc, [pattern, placeholder]) => acc.replace(pattern, placeholder), text);
  return maskNames(maskNumberRuns(digitRules));
}

const PLACEHOLDER_TEXTS = Object.values(PLACEHOLDERS);
const LETTER = /\p{L}/u;

function hasEvidence(phrase) {
  const rest = PLACEHOLDER_TEXTS.reduce((acc, placeholder) => acc.replaceAll(placeholder, ""), phrase);
  return LETTER.test(rest);
}

const CONTEXT_CHARS = 40;

function scrubbedInContext(phrase, original) {
  const at = original.indexOf(phrase);
  if (at < 0) return null;
  const context = Array.from(original.slice(0, at)).slice(-CONTEXT_CHARS).join("");
  const withPhrase = scrubText(context + phrase);
  const alone = scrubText(context);
  return withPhrase.startsWith(alone) ? withPhrase.slice(alone.length) : null;
}

function groundedForm(phrase, redacted, original) {
  if (redacted.includes(phrase)) return phrase;
  for (const candidate of [scrubText(phrase), scrubbedInContext(phrase, original)]) {
    if (candidate && redacted.includes(candidate) && hasEvidence(candidate)) return candidate;
  }
  return null;
}

/** Port of `data/span_grounding.py::ground_flagged_phrases` (ADR D66): a flagged phrase is kept
 *  verbatim, or with its PII masked (alone, then in its original context), while that form is
 *  verbatim in the redacted transcript and says something besides placeholders. */
export function groundFlaggedPhrases(flaggedPhrases, redacted, original = "") {
  const kept = [];
  for (const phrase of flaggedPhrases) {
    const candidate = phrase?.trim() ? groundedForm(phrase, redacted, original) : null;
    if (candidate !== null && !kept.includes(candidate)) kept.push(candidate);
  }
  return kept;
}

export class ReportReviewError extends Error {}

// What `POST /api/reports` accepts as flagged evidence (src/qorgan/api_reports.py, pinned by
// tests_js/report.test.mjs). A phrase over the limit is dropped, never cut: evidence stays verbatim.
export const REPORT_LIMITS = Object.freeze({ phrases: 50, phraseChars: 300 });

/** The page string (site/i18n.js) for a refused report, from the HTTP 422 body. FastAPI answers a
    list for request validation and the route a string, in English; neither reaches a citizen. */
export function rejectionKey(body) {
  const detail = body?.detail;
  if (typeof detail === "string") return detail.startsWith("caller number") ? "report.err_number" : "report.err_invalid";
  const fields = Array.isArray(detail) ? detail.map((item) => item?.loc?.at?.(-1)) : [];
  if (fields.includes("transcript")) return "report.err_too_long";
  if (fields.includes("phone_number")) return "report.err_number";
  return "report.err_invalid";
}

/** The canonical form of a receipt a citizen pasted (`RECEIPT_ID_PATTERN`), or null. */
export function normalizeReceipt(value) {
  const receipt = String(value ?? "").trim().toLowerCase();
  return /^[0-9a-f]{24}$/.test(receipt) ? receipt : null;
}

/** The editable view of a post-call draft (`summary.buildReport`). */
export function reviewDraft(draft) {
  return {
    transcript: draft.transcript,
    phoneNumber: draft.phone_number ?? "",
    tactics: draft.tactic_ids.map((id) => ({ id, included: true })),
    flaggedPhrases: [...draft.flagged_phrases],
  };
}

/**
 * The exact payload for `POST /api/reports`, from the original draft + the citizen's edits.
 * Edits can only remove: tactics are a subset of the detected ones, flagged phrases survive
 * only while still verbatim in the (edited, redacted) transcript, as-is or with their own PII
 * masked -- grounded evidence, never text the model did not flag. Throws unless consent was given explicitly.
 */
export function finalizeReport(draft, { transcript, phoneNumber, includedTacticIds, consent }) {
  if (consent !== true) throw new ReportReviewError("consent is required before a report is sent");
  const text = String(transcript ?? "").trim();
  if (!text) throw new ReportReviewError("the transcript is empty");
  const redacted = scrubText(text);
  const included = new Set(includedTacticIds ?? []);
  const phone = String(phoneNumber ?? "").trim();
  return {
    transcript: text,
    phone_number: phone || null,
    flagged_phrases: groundFlaggedPhrases(draft.flagged_phrases, redacted, text)
      .filter((phrase) => phrase.length <= REPORT_LIMITS.phraseChars)
      .slice(0, REPORT_LIMITS.phrases),
    tactic_ids: draft.tactic_ids.filter((id) => included.has(id)),
    timestamp: draft.timestamp,
    risk_score: draft.risk_score,
    consent: true,
  };
}
