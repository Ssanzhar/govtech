/* Layer 2 role slot -- port of `qorgan/explain/roles.py` (ADR D63). Display-only: who the
   caller claimed to be, as a verbatim span + a reviewed template. Never a model input.
   Rule (both runtimes): per role, the first cue `findCue` matches gives its span; a span
   overlapped by a strictly longer span of another role is dropped; earliest start wins
   (ties: lexicon order). Pinned by tests_js/fixtures/roles.json. */

import { findCue } from "./cue-match.js";

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

/** Every non-dominated claim, ordered by start then lexicon order. */
export function detectRoles(transcript, roles) {
  if (!transcript || !transcript.trim()) return [];
  const candidates = [];
  roles.forEach((role, order) => {
    const span = firstMatch(transcript, role.cues);
    if (span) candidates.push({ order, claim: { role_id: role.id, span } });
  });
  const kept = candidates.filter(({ claim }) => !candidates.some(
    ({ claim: other }) => overlaps(claim.span, other.span) && other.span.text.length > claim.span.text.length,
  ));
  kept.sort((a, b) => (a.claim.span.start - b.claim.span.start) || (a.order - b.order));
  return kept.map(({ claim }) => claim);
}

export function detectRole(transcript, roles) {
  const claims = detectRoles(transcript, roles);
  return claims.length ? claims[0] : null;
}

export function renderRoleLine(template, role, span) {
  return template.replaceAll("{role}", role).replaceAll("{span}", span);
}

/** `{role_id, role_span, role_line}` for an explanation, or nulls when nothing was claimed. */
export function roleClaim(transcript, locale, config) {
  const roles = config.roles || [];
  const claim = detectRole(transcript, roles);
  if (!claim) return { role_id: null, role_span: null, role_line: null };
  const role = roles.find((r) => r.id === claim.role_id);
  const line = renderRoleLine(config.templates[locale].role_template, role.names[locale], claim.span.text);
  return { role_id: claim.role_id, role_span: claim.span, role_line: line };
}
