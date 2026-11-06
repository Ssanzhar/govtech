/* Marking evidence in a transcript. Spans are verbatim; their offsets count code points when the server
   made them and UTF-16 units when the page did, which differ after an emoji. */
import test from "node:test";
import assert from "node:assert/strict";
import { markSpans } from "../site/core/highlight.js";

const plain = (s) => s;
const span = (text, start) => ({ text, start, end: start + Array.from(text).length });

test("spans are marked where they are, in order", () => {
  const text = "Назовите код из смс. Никому не говорите.";
  assert.equal(
    markSpans(text, [span("Никому", 21), span("код", 9)], plain),
    "Назовите <mark>код</mark> из смс. <mark>Никому</mark> не говорите.",
  );
});

test("a server's code-point offsets still land after an emoji", () => {
  const text = "😀😀 назовите код";
  const start = Array.from(text).indexOf("к");
  assert.equal(markSpans(text, [{ text: "код", start, end: start + 3 }], plain), "😀😀 назовите <mark>код</mark>");
});

test("the page's own UTF-16 offsets land too", () => {
  const text = "😀😀 назовите код";
  const start = text.indexOf("код");
  assert.equal(markSpans(text, [{ text: "код", start, end: start + 3 }], plain), "😀😀 назовите <mark>код</mark>");
});

// A cue span can sit inside an utterance span already marked. Searching for its text from there on
// would mark a later, innocent repeat of the phrase as evidence.
test("a span inside one already marked is skipped, never moved to a later repeat", () => {
  const text = "назовите код сейчас. Код у вас есть, назовите код позже.";
  const whole = span("назовите код сейчас.", 0);
  const inner = span("код", 9);
  assert.equal(markSpans(text, [whole, inner], plain), "<mark>назовите код сейчас.</mark> Код у вас есть, назовите код позже.");
});

test("a span that is not where it says is dropped: nothing unverifiable is marked", () => {
  assert.equal(markSpans("назовите код", [{ text: "пароль", start: 0, end: 6 }], plain), "назовите код");
});

test("everything is escaped by the caller's function", () => {
  const esc = (s) => s.replaceAll("<", "&lt;");
  assert.equal(markSpans("a<b код", [span("код", 4)], esc), "a&lt;b <mark>код</mark>");
});
