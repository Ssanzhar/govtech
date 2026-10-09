/* Layer 2 role slot -- port of `qorgan/explain/roles.py` (ADR D63). Display-only: who the
   caller claimed to be, as a verbatim span + a reviewed template. Never a model input.
   Rule (both runtimes): per role, the first cue `findCue` matches gives its span; a span
   overlapped by a strictly longer span of another role is dropped; earliest start wins
   (ties: lexicon order). Pinned by tests_js/fixtures/roles.json.
   Self-introduction by organisation (phase 2): per line, tokens = runs of letters/digits folded
   per character (lower, ё -> е); an organisation (`orgs`, `*` = prefix token, else exact or
   exact + ablative affix) is a claim when it carries an affix and opens the line, or an intro
   marker sits within `intro.window` tokens (nearest wins, ties: before). Longest organisation
   wins at a position; per role the earlier of cue span and intro span (ties: longer). */

import { findCue } from "./cue-match.js";

const BOUNDARY = "\n";
const PREFIX = "*";
const TOKEN_CHAR = /^[0-9a-zа-яәғқңөұүһі]$/;

function firstMatch(transcript, cues) {
  for (const cue of cues) {
    const found = findCue(transcript, cue);
    if (found) {
      const [start, end] = found;
      return { text: transcript.slice(start, end), start, end };
    }
  }
  return null;
}

const overlaps = (a, b) => a.start < b.end && b.start < a.end;

/** `[token, start, end]` runs of letters/digits, folded per character (lower, ё -> е). */
export function tokenize(line) {
  const out = [];
  let buf = "";
  let start = -1;
  for (let i = 0; i < line.length; i += 1) {
    let folded = line[i].toLowerCase();
    if (folded === "ё") folded = "е";
    if (folded.length === 1 && TOKEN_CHAR.test(folded)) {
      if (start < 0) start = i;
      buf += folded;
    } else if (start >= 0) {
      out.push([buf, start, i]);
      buf = ""; start = -1;
    }
  }
  if (start >= 0) out.push([buf, start, line.length]);
  return out;
}

function tokenMatch(token, pattern, affixes) {
  if (pattern.endsWith(PREFIX)) {
    const stem = pattern.slice(0, -1);
    return token.startsWith(stem) ? token.slice(stem.length) : null;
  }
  if (token === pattern) return "";
  if (token.startsWith(pattern) && affixes.includes(token.slice(pattern.length))) return token.slice(pattern.length);
  return null;
}

function seqMatch(tokens, at, pattern, affixes) {
  if (at + pattern.length > tokens.length) return null;
  let remainder = "";
  for (let k = 0; k < pattern.length; k += 1) {
    remainder = tokenMatch(tokens[at + k][0], pattern[k], affixes);
    if (remainder === null) return null;
  }
  return remainder;
}

function orgAt(tokens, at, roles, affixes) {
  let best = null;
  roles.forEach((role, order) => {
    for (const org of role.orgs || []) {
      const remainder = seqMatch(tokens, at, org, affixes);
      if (remainder !== null && (best === null || org.length > best.length)) best = { order, length: org.length, remainder };
    }
  });
  return best;
}

function markerRanges(tokens, intro) {
  const ranges = [];
  for (let at = 0; at < tokens.length; at += 1) {
    for (const marker of intro.markers) {
      if (seqMatch(tokens, at, marker, []) !== null) ranges.push([at, at + marker.length - 1]);
    }
  }
  return ranges;
}

function introSpan(tokens, first, last, remainder, markers, intro) {
  const opensLine = first <= intro.window;
  if (opensLine && remainder && intro.affixes.some((a) => remainder.endsWith(a))) return [first, last];
  if (opensLine && !remainder && last + 1 < tokens.length && intro.affixes.includes(tokens[last + 1][0])) return [first, last + 1];
  let best = null;
  for (const [mStart, mEnd] of markers) {
    let gap; let after;
    if (mEnd < first) { gap = first - mEnd - 1; after = 0; } else if (mStart > last) { gap = mStart - last - 1; after = 1; } else continue;
    if (gap <= intro.window && (best === null || gap < best.gap || (gap === best.gap && after < best.after))) {
      best = { gap, after, start: Math.min(first, mStart), end: Math.max(last, mEnd) };
    }
  }
  return best === null ? null : [best.start, best.end];
}

/** {role order: earliest self-introduction span}. */
export function introClaims(transcript, roles, intro) {
  const found = new Map();
  if (!intro || (!intro.markers.length && !intro.affixes.length)) return found;
  let offset = 0;
  for (const line of transcript.split(BOUNDARY)) {
    const tokens = tokenize(line);
    const markers = markerRanges(tokens, intro);
    let at = 0;
    while (at < tokens.length) {
      const org = orgAt(tokens, at, roles, intro.affixes);
      if (org === null) { at += 1; continue; }
      const span = introSpan(tokens, at, at + org.length - 1, org.remainder, markers, intro);
      if (span !== null && !found.has(org.order)) {
        const start = tokens[span[0]][1] + offset;
        const end = tokens[span[1]][2] + offset;
        found.set(org.order, { text: transcript.slice(start, end), start, end });
      }
      at += org.length;
    }
    offset += line.length + BOUNDARY.length;
  }
  return found;
}

function earlier(a, b) {
  if (!a || !b) return a || b || null;
  if (a.start !== b.start) return a.start < b.start ? a : b;
  return a.text.length >= b.text.length ? a : b;
}

/** Every non-dominated claim, ordered by start then lexicon order. */
export function detectRoles(transcript, roles, intro = null) {
  if (!transcript || !transcript.trim()) return [];
  const intros = introClaims(transcript, roles, intro);
  const candidates = [];
  roles.forEach((role, order) => {
    const span = earlier(firstMatch(transcript, role.cues), intros.get(order));
    if (span) candidates.push({ order, claim: { role_id: role.id, span } });
  });
  const kept = candidates.filter(({ claim }) => !candidates.some(
    ({ claim: other }) => overlaps(claim.span, other.span) && other.span.text.length > claim.span.text.length,
  ));
  kept.sort((a, b) => (a.claim.span.start - b.claim.span.start) || (a.order - b.order));
  return kept.map(({ claim }) => claim);
}

export function detectRole(transcript, roles, intro = null) {
  const claims = detectRoles(transcript, roles, intro);
  return claims.length ? claims[0] : null;
}

export function renderRoleLine(template, role, span) {
  return template.replaceAll("{role}", role).replaceAll("{span}", span);
}

/** `{role_id, role_span, role_line}` for an explanation, or nulls when nothing was claimed. */
export function roleClaim(transcript, locale, config) {
  const roles = config.roles || [];
  const claim = detectRole(transcript, roles, config.role_intro || null);
  if (!claim) return { role_id: null, role_span: null, role_line: null };
  const role = roles.find((r) => r.id === claim.role_id);
  const line = renderRoleLine(config.templates[locale].role_template, role.names[locale], claim.span.text);
  return { role_id: claim.role_id, role_span: claim.span, role_line: line };
}
