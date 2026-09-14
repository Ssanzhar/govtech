/* Grounded, localized explanation -- port of `qorgan/explain/explainer.py` + `templates.py`.
   Every highlight is re-validated as a verbatim slice of the transcript. */

export class ExplainerError extends Error {}

export function explain(result, transcript, locale, config) {
  if (!config.locales.includes(locale)) throw new ExplainerError(`Unsupported locale ${locale}`);
  if (!transcript || !transcript.trim()) throw new ExplainerError("transcript must not be empty");
  for (const span of result.attributions) {
    if (transcript.slice(span.start, span.end) !== span.text) {
      throw new ExplainerError(`span is not verbatim: ${JSON.stringify(span.text)}`);
    }
  }
  const t = config.templates[locale];
  const highlights = [...result.attributions].sort((a, b) => a.start - b.start);
  const names = result.tags.map((tag) => displayName(tag.id, locale, config)).filter((n) => n);
  const reason = renderReason(t, names, highlights.map((s) => s.text));
  const calibrated = config.scoring.calibrated_backends.includes(result.backend);
  return {
    reason,
    highlights,
    tags: result.tags,
    confidence: result.raw_confidence,
    confidence_label: confidenceBand(t, result.raw_confidence, config.scoring.confidence_bands),
    caveat: renderCaveat(t, result.backend, calibrated),
    human_note: t.human_note,
  };
}

export function displayName(tacticId, locale, config) {
  const tactic = config.taxonomy.tactics.find((t) => t.id === tacticId);
  return tactic ? tactic.names[locale] : null;
}

export function renderReason(t, tacticNames, phrases) {
  if (!tacticNames.length) return t.no_signal_reason;
  const spans = phrases.length ? phrases.map((p) => `«${p}»`).join(", ") : t.spans_placeholder;
  return t.reason_template.replaceAll("{tags}", tacticNames.join(", ")).replaceAll("{spans}", spans);
}

export function renderCaveat(t, backend, isCalibrated) {
  return isCalibrated ? t.caveat : t.caveat + t.uncalibrated_suffix.replaceAll("{backend}", backend);
}

export function confidenceBand(t, confidence, bands) {
  if (confidence === null || confidence === undefined) return t.confidence.unknown;
  if (confidence >= bands.high) return t.confidence.high;
  if (confidence >= bands.medium) return t.confidence.medium;
  return t.confidence.low;
}
