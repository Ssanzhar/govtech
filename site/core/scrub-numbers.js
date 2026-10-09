/* Numbers dictated as words, and digit groups split by spaces -- 1:1 port of
   `qorgan/data/scrub_numbers.py` (ADR D66, gap M7); read the rules there. */

import { PLACEHOLDERS, applyReplacements, slice, tokenize } from "./scrub-tokens.js";

export const SPOKEN_NUMBER_MIN_DIGITS = 7;
const MAX_GAP_CHARS = 3;
const GAP_CHARS = new Set([" ", "\t", "\r", "\n", "\u00a0", ",", "-"]);
const DIGIT_GROUP_GAPS = new Set([" ", "-"]);
const ASCII_WORD = /^[A-Za-z0-9_]$/;
const CURRENCY_WORDS = new Set([
  "тенге", "теңге", "тг", "tenge", "kzt", "рублей", "рубля", "руб", "долларов", "доллара",
  "dollars", "usd", "евро", "euro", "сом",
]);
const CURRENCY_SIGNS = new Set(["₸", "$", "€", "₽"]);

const words = (table) => new Map(Object.entries(table).flatMap(([value, list]) => list.map((w) => [w, Number(value)])));

const ZERO = new Set(["ноль", "нуль", "нөл", "нол", "zero"]);
const UNITS = words({
  1: ["один", "одна", "одно", "бір", "бир", "one"],
  2: ["два", "две", "екі", "еки", "two"],
  3: ["три", "үш", "уш", "three"],
  4: ["четыре", "төрт", "торт", "four"],
  5: ["пять", "бес", "five"],
  6: ["шесть", "алты", "six"],
  7: ["семь", "жеті", "жети", "seven"],
  8: ["восемь", "сегіз", "сегиз", "eight"],
  9: ["девять", "тоғыз", "тогыз", "nine"],
});
const TEENS = words({
  10: ["десять", "ten"],
  11: ["одиннадцать", "одинадцать", "eleven"],
  12: ["двенадцать", "twelve"],
  13: ["тринадцать", "thirteen"],
  14: ["четырнадцать", "fourteen"],
  15: ["пятнадцать", "fifteen"],
  16: ["шестнадцать", "sixteen"],
  17: ["семнадцать", "seventeen"],
  18: ["восемнадцать", "eighteen"],
  19: ["девятнадцать", "nineteen"],
});
const TENS = words({
  1: ["он"],
  2: ["двадцать", "жиырма", "twenty"],
  3: ["тридцать", "отыз", "thirty"],
  4: ["сорок", "қырық", "кырык", "forty"],
  5: ["пятьдесят", "пятдесят", "елу", "fifty"],
  6: ["шестьдесят", "шестдесят", "алпыс", "sixty"],
  7: ["семьдесят", "семдесят", "жетпіс", "жетпис", "seventy"],
  8: ["восемьдесят", "восемдесят", "сексен", "eighty"],
  9: ["девяносто", "тоқсан", "токсан", "ninety"],
});
const HUNDREDS = words({
  1: ["сто"],
  2: ["двести"],
  3: ["триста"],
  4: ["четыреста"],
  5: ["пятьсот", "пятсот"],
  6: ["шестьсот", "шестсот"],
  7: ["семьсот", "семсот"],
  8: ["восемьсот", "восемсот"],
  9: ["девятьсот", "девятсот"],
});
const HUNDRED_MULTIPLIERS = new Set(["жүз", "жуз", "hundred"]);
const TABLES = [["unit", UNITS], ["teen", TEENS], ["tens", TENS], ["hundreds", HUNDREDS]];

function numberKind(token) {
  if (token.kind === "digits") return ["digits", 0];
  const word = token.folded;
  if (ZERO.has(word)) return ["zero", 0];
  for (const [kind, table] of TABLES) if (table.has(word)) return [kind, table.get(word)];
  if (HUNDRED_MULTIPLIERS.has(word)) return ["multiplier", 1];
  return null;
}

// A pending dictation group is [form, hundreds, tens, units, teen]; form is "H", "T" or "U".
function flush(group) {
  if (group === null) return "";
  const [form, hundreds, tens, units, teen] = group;
  if (form === "U") return String(units);
  if (form === "T") return `${tens}${units ?? 0}`;
  if (teen !== null) return `${hundreds}${teen}`;
  return `${hundreds}${tens ?? 0}${units ?? 0}`;
}

function step(group, kind, value, text) {
  const form = group ? group[0] : null;
  const openHundreds = form === "H" && group[2] === null && group[3] === null && group[4] === null;
  if (kind === "digits") return [flush(group) + text, null];
  if (kind === "zero") {
    if (openHundreds) return ["", ["H", group[1], 0, null, null]];
    if (form === "H" && group[2] === 0 && group[3] === null) return [flush(["H", group[1], 0, 0, null]), null];
    return [flush(group) + "0", null];
  }
  if (kind === "unit") {
    if (form === "H" && group[3] === null && group[4] === null) return [flush(["H", group[1], group[2], value, null]), null];
    if (form === "T" && group[3] === null) return [flush(["T", null, group[2], value, null]), null];
    return [flush(group), ["U", null, null, value, null]];
  }
  if (kind === "teen") {
    if (openHundreds) return [flush(["H", group[1], null, null, value]), null];
    return [flush(group) + String(value), null];
  }
  if (kind === "tens") {
    if (openHundreds) return ["", ["H", group[1], value, null, null]];
    return [flush(group), ["T", null, value, null, null]];
  }
  if (kind === "hundreds") return [flush(group), ["H", value, null, null, null]];
  if (form === "U") return ["", ["H", group[3], null, null, null]];
  return [flush(group), ["H", 1, null, null, null]];
}

export function composeDigits(tokens) {
  const digits = [];
  let group = null;
  for (const token of tokens) {
    const [kind, value] = numberKind(token) ?? ["", 0];
    const [emitted, next] = step(group, kind, value, token.text);
    digits.push(emitted);
    group = next;
  }
  digits.push(flush(group));
  return digits.join("");
}

export function spokenDigits(text) {
  return composeDigits(tokenize(Array.from(text)).filter((t) => numberKind(t) !== null));
}

const joinable = (gap) => gap.length <= MAX_GAP_CHARS && gap.every((c) => GAP_CHARS.has(c));

function runs(chars, tokens) {
  const found = [];
  let current = [];
  tokens.forEach((token, index) => {
    if (numberKind(token) === null) {
      if (current.length) found.push(current);
      current = [];
      return;
    }
    if (current.length && !joinable(chars.slice(tokens[current.at(-1)].end, token.start))) {
      found.push(current);
      current = [];
    }
    current.push(index);
  });
  if (current.length) found.push(current);
  return found;
}

function placeholderFor(digits) {
  if (digits.length === 16) return PLACEHOLDERS.card;
  if (digits.length === 12) return PLACEHOLDERS.iin;
  if ((digits.length === 11 && "78".includes(digits[0])) || (digits.length === 10 && digits[0] === "7")) {
    return PLACEHOLDERS.phone;
  }
  return PLACEHOLDERS.number;
}

const SPACES = new Set([" ", "\u00a0"]);

function followedByCurrency(chars, end, nextToken) {
  let cursor = end;
  while (cursor < chars.length && SPACES.has(chars[cursor])) cursor += 1;
  if (cursor < chars.length && CURRENCY_SIGNS.has(chars[cursor])) return true;
  if (nextToken === null) return false;
  if (!chars.slice(end, nextToken.start).every((c) => SPACES.has(c))) return false;
  return CURRENCY_WORDS.has(nextToken.folded);
}

function digitGroupsQualify(chars, run, digits) {
  if (run.length < 2 || placeholderFor(digits) === PLACEHOLDERS.number) return false;
  for (let i = 1; i < run.length; i += 1) {
    if (!DIGIT_GROUP_GAPS.has(slice(chars, run[i - 1].end, run[i].start))) return false;
  }
  const before = run[0].start > 0 ? chars[run[0].start - 1] : "";
  const after = run.at(-1).end < chars.length ? chars[run.at(-1).end] : "";
  return !ASCII_WORD.test(before) && !ASCII_WORD.test(after);
}

function shouldMask(chars, run, digits) {
  if (run.some((token) => token.kind === "letters")) return digits.length >= SPOKEN_NUMBER_MIN_DIGITS;
  return digitGroupsQualify(chars, run, digits);
}

function lineSpans(chars, run) {
  const spans = [];
  let [start, end] = [run[0].start, run[0].end];
  for (let i = 1; i < run.length; i += 1) {
    if (chars.slice(run[i - 1].end, run[i].start).includes("\n")) {
      spans.push([start, end]);
      start = run[i].start;
    }
    end = run[i].end;
  }
  spans.push([start, end]);
  return spans;
}

export function maskNumberRuns(text) {
  const chars = Array.from(text);
  const tokens = tokenize(chars);
  const replacements = [];
  for (const indices of runs(chars, tokens)) {
    const run = indices.map((i) => tokens[i]);
    const last = indices.at(-1);
    const nextToken = last + 1 < tokens.length ? tokens[last + 1] : null;
    const digits = composeDigits(run);
    if (!shouldMask(chars, run, digits) || followedByCurrency(chars, run.at(-1).end, nextToken)) continue;
    const placeholder = placeholderFor(digits);
    for (const [start, end] of lineSpans(chars, run)) replacements.push([start, end, placeholder]);
  }
  return applyReplacements(chars, replacements);
}
