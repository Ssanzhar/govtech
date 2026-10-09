/* Tokens for the context-aware scrub rules -- 1:1 port of `qorgan/data/scrub_tokens.py`
   (ADR D66). A token is a maximal run of Unicode letters (\p{L}) or of ASCII digits. Offsets
   are code-point indices into `Array.from(text)`, like Python's string indices, so the two
   runtimes agree even on emoji. Parity is pinned by `tests_js/fixtures/scrub.json`. */

export const PLACEHOLDERS = Object.freeze({
  phone: "[PHONE]",
  card: "[CARD]",
  iin: "[IIN]",
  email: "[EMAIL]",
  iban: "[IBAN]",
  number: "[NUMBER]",
  name: "[NAME]",
});

const LETTER = /\p{L}/u;

export const fold = (word) => word.toLowerCase().replaceAll("ё", "е");

function charKind(character) {
  if (character >= "0" && character <= "9") return "digits";
  if (LETTER.test(character)) return "letters";
  return null;
}

/** `chars` is `Array.from(text)`; returns frozen `{text, start, end, kind, folded}` tokens. */
export function tokenize(chars) {
  const tokens = [];
  let start = 0;
  let kind = null;
  const push = (end) => {
    const text = chars.slice(start, end).join("");
    tokens.push(Object.freeze({ text, start, end, kind, folded: fold(text) }));
  };
  chars.forEach((character, index) => {
    const current = charKind(character);
    if (current !== kind) {
      if (kind !== null) push(index);
      start = index;
      kind = current;
    }
  });
  if (kind !== null) push(chars.length);
  return tokens;
}

export function isCapitalised(word) {
  const first = Array.from(word)[0] ?? "";
  return first !== first.toLowerCase() && first === first.toUpperCase();
}

export const slice = (chars, start, end) => chars.slice(start, end).join("");

/** `chars` joined with each non-overlapping `[start, end, placeholder]` substituted. */
export function applyReplacements(chars, replacements) {
  if (!replacements.length) return chars.join("");
  const sorted = [...replacements].sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  const pieces = [];
  let cursor = 0;
  for (const [start, end, placeholder] of sorted) {
    pieces.push(slice(chars, cursor, start), placeholder);
    cursor = end;
  }
  pieces.push(slice(chars, cursor, chars.length));
  return pieces.join("");
}
