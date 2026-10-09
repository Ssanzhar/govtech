/* Self-introduced personal names -- 1:1 port of `qorgan/data/scrub_names.py` (ADR D66);
   read the rules there. Only the name is replaced; the introduction stays. */

import { PLACEHOLDERS, applyReplacements, isCapitalised, slice, tokenize } from "./scrub-tokens.js";

const STRONG_TRIGGERS = [
  ["меня", "зовут"], ["мое", "имя"], ["моя", "фамилия"], ["атым"], ["есімім"], ["есимим"], ["my", "name", "is"],
];
const WEAK_TRIGGERS = [["это"], ["this", "is"], ["i", "m"], ["i", "am"]];
const WEAK_FOLLOWERS = new Set(["из", "с", "со", "от", "from", "calling", "with"]);
const STOP_WORDS = new Set([
  // ru
  "не", "я", "мы", "вы", "он", "она", "они", "это", "из", "с", "со", "от", "и", "а", "но",
  "ваш", "ваша", "вам", "вас", "так", "как", "что", "сотрудник", "сотрудница", "служба",
  "службы", "банк", "банка", "отдел", "полиция", "оператор", "специалист", "менеджер",
  "нацбанк", "казпочта", "налоговая", "компания",
  // kk
  "ба", "бе", "ма", "ме", "па", "пе", "мен", "сіз", "біз", "банк", "банктен", "қызметкер",
  "қызметкері", "қызметі",
  // en
  "the", "a", "an", "from", "calling", "with", "and", "i", "not", "your", "bank", "service",
  "security", "officer", "agent", "here", "just", "is", "post", "kazakhstan", "казахстан",
  // brands callers impersonate: evidence, never a personal name
  "kaspi", "каспи", "halyk", "халык", "халық", "homebank", "kazpost", "казпост", "kcell",
  "кселл", "beeline", "билайн", "tele", "теле", "egov", "егов", "jusan", "жусан", "forte",
  "форте", "bereke", "береке", "freedom", "фридом", "kazakhtelecom", "казахтелеком",
]);
// Kept in front of a weak-trigger name: «это майор [NAME] из …», "this is Major [NAME] from …".
const TITLES = new Set([
  "майор", "капитан", "лейтенант", "старший", "полковник", "подполковник", "сержант",
  "следователь", "инспектор", "господин", "госпожа", "major", "captain", "lieutenant",
  "colonel", "sergeant", "inspector", "detective", "mr", "mrs", "ms", "dr",
]);
const SURNAME_ENDINGS = [
  "ов", "ова", "ев", "ева", "ина", "ын", "ына", "ский", "ская", "цкий", "цкая", "енко", "ұлы",
  "қызы", "улы", "кызы", "ov", "ova", "ev", "eva", "ina", "sky", "skaya", "enko", "uly", "kyzy",
];
const MAX_EXTRA_NAME_WORDS = 2;
const MIN_NAME_CHARS = 2;
const MIN_SURNAME_CHARS = 4;
const TRIGGER_GAP_CHARS = new Set([" ", "\t", "\u00a0", "'", "\u2019"]);
const INTRO_GAP_CHARS = new Set([" ", "\t", "\u00a0", ":", "\u2014", "\u2013", "-"]);
const MAX_INTRO_GAP = 4;
const NAME_GAP_CHARS = new Set([" ", "\t", "\u00a0"]);
const MAX_NAME_GAP = 2;

const charCount = (text) => Array.from(text).length;
const gapOk = (gap, allowed, limit) => {
  const chars = Array.from(gap);
  return chars.length > 0 && chars.length <= limit && chars.every((c) => allowed.has(c));
};

function matchesTrigger(chars, tokens, index, trigger) {
  if (index + trigger.length > tokens.length) return false;
  const window = tokens.slice(index, index + trigger.length);
  if (window.some((token, i) => token.folded !== trigger[i])) return false;
  for (let i = 1; i < window.length; i += 1) {
    if (!gapOk(slice(chars, window[i - 1].end, window[i].start), TRIGGER_GAP_CHARS, MAX_INTRO_GAP)) return false;
  }
  return true;
}

const nameWord = (token) =>
  token.kind === "letters" && charCount(token.text) >= MIN_NAME_CHARS && !STOP_WORDS.has(token.folded);
const looksLikeSurname = (token) =>
  charCount(token.text) >= MIN_SURNAME_CHARS && SURNAME_ENDINGS.some((e) => token.folded.endsWith(e));

function strongName(chars, tokens, start) {
  if (start >= tokens.length || !nameWord(tokens[start])) return 0;
  let count = 1;
  while (count <= MAX_EXTRA_NAME_WORDS && start + count < tokens.length) {
    const token = tokens[start + count];
    if (!gapOk(slice(chars, tokens[start + count - 1].end, token.start), NAME_GAP_CHARS, MAX_NAME_GAP)) break;
    if (!(nameWord(token) && (isCapitalised(token.text) || looksLikeSurname(token)))) break;
    count += 1;
  }
  return count;
}

function weakName(chars, tokens, start) {
  for (const count of [2, 1]) {
    const end = start + count;
    if (end >= tokens.length) continue;
    const names = tokens.slice(start, end);
    if (!names.every((t) => nameWord(t) && isCapitalised(t.text))) continue;
    if (count === 2 && !looksLikeSurname(names[1])) continue;
    let gapsOk = true;
    for (let i = start; i <= end; i += 1) {
      if (!gapOk(slice(chars, tokens[i - 1].end, tokens[i].start), NAME_GAP_CHARS, MAX_NAME_GAP)) gapsOk = false;
    }
    if (gapsOk && WEAK_FOLLOWERS.has(tokens[end].folded)) return count;
  }
  return 0;
}

function nameAt(chars, tokens, index) {
  for (const trigger of STRONG_TRIGGERS) {
    if (!matchesTrigger(chars, tokens, index, trigger)) continue;
    const start = index + trigger.length;
    if (start < tokens.length && gapOk(slice(chars, tokens[start - 1].end, tokens[start].start), INTRO_GAP_CHARS, MAX_INTRO_GAP)) {
      const count = strongName(chars, tokens, start);
      if (count) return [start, count];
    }
  }
  for (const trigger of WEAK_TRIGGERS) {
    if (matchesTrigger(chars, tokens, index, trigger)) {
      let start = index + trigger.length;
      if (start + 1 < tokens.length && TITLES.has(tokens[start].folded)) start += 1; // the title stays
      const count = start < tokens.length ? weakName(chars, tokens, start) : 0;
      if (count) return [start, count];
    }
  }
  return [0, 0];
}

export function maskNames(text) {
  const chars = Array.from(text);
  const tokens = tokenize(chars);
  const replacements = [];
  let index = 0;
  while (index < tokens.length) {
    const [start, count] = nameAt(chars, tokens, index);
    if (count) {
      replacements.push([tokens[start].start, tokens[start + count - 1].end, PLACEHOLDERS.name]);
      index = start + count;
    } else {
      index += 1;
    }
  }
  return applyReplacements(chars, replacements);
}
