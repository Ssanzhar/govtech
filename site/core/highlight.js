/* Marks evidence spans in a transcript. Spans are verbatim, so an offset is only trusted if the text
   is really there: the server counts code points and the page UTF-16 units, which differ after an emoji. */

/** `[start, end]` of `span` in `text` as UTF-16 offsets, or null if it is not where it says. */
function locate(text, span) {
  if (text.slice(span.start, span.end) === span.text) return [span.start, span.end];
  const start = Array.from(text).slice(0, span.start).join("").length;
  return text.startsWith(span.text, start) ? [start, start + span.text.length] : null;
}

/** `text` with each span wrapped in <mark>, escaped by `esc`. A span overlapping one already marked is
    skipped: it is a cue inside a marked utterance, and looking for its words further on could mark an
    innocent repeat of them as evidence. */
export function markSpans(text, spans, esc) {
  const sorted = [...spans].sort((a, b) => a.start - b.start);
  let html = "";
  let cursor = 0;
  for (const span of sorted) {
    const found = locate(text, span);
    if (!found || found[0] < cursor) continue;
    const [start, end] = found;
    html += `${esc(text.slice(cursor, start))}<mark>${esc(text.slice(start, end))}</mark>`;
    cursor = end;
  }
  return html + esc(text.slice(cursor));
}
