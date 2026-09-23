/* `scoreTranscript` -- port of `qorgan/classifier/predict.py::_linear_score` for the
   hybrid bundle. `embed(texts) -> Promise<number[][]>` is injected (an ONNX worker in the
   browser, recorded vectors in tests) so this module stays pure and testable. */

import { selectTopUtteranceSpans } from "./attribution.js";
import { decodeTactics, hybridRow, riskProba, tacticProba } from "./head.js";
import { MATCHER_VERSION } from "./cue-match.js";
import { compileReassurance, hardSignalFeatures, matchCues, reassuranceFeature } from "./lexicon.js";

export const BACKEND = "linear";

/** Bind weights + config once; returns an async `score(transcript)`. */
export function createScorer({ weights, config, embed }) {
  if (weights.format_version !== 1) throw new Error(`unsupported weights format ${weights.format_version}`);
  if (!weights.lexicon) throw new Error("web scorer needs a hybrid bundle (lexicon block)");
  // The cue block is a model input: weights trained under another matcher would be scored
  // with features they were never fitted on (review, 2026-09-23).
  if (weights.cue_matcher_version !== MATCHER_VERSION) {
    throw new Error(
      `weights were trained with cue matcher v${weights.cue_matcher_version} but this build uses v${MATCHER_VERSION}`,
    );
  }
  const hardIds = config.taxonomy.hard_signal_ids;
  const matcher = compileReassurance(weights.lexicon.reassurance);
  const join = config.window.join;
  const { attribution_top_k: topK, attribution_min_score: minScore } = config.scoring;

  const featureRows = (texts, embeddings) =>
    texts.map((text, i) =>
      hybridRow(embeddings[i], hardSignalFeatures(text, weights.lexicon.cues, hardIds), reassuranceFeature(text, matcher)),
    );

  return async function score(transcript) {
    const utterances = transcript.split(join);
    const [transcriptEmbedding, ...utteranceEmbeddings] = await embed([transcript, ...utterances]);
    const [row] = featureRows([transcript], [transcriptEmbedding]);
    const risk = riskProba(row, weights.risk_head);
    const probs = tacticProba(transcriptEmbedding, weights.tactic_head, weights.label_space);
    const utteranceScores = featureRows(utterances, utteranceEmbeddings).map((r) => riskProba(r, weights.risk_head));

    let tags = decodeTactics(probs, weights.label_space, weights.tactic_head.threshold, weights.tactic_head.thresholds || {}).map(([id, p]) => ({
      id,
      weight: Math.min(1, Math.max(0, p)),
    }));
    let spans = selectTopUtteranceSpans(utterances, utteranceScores, { topK, minScore, join });
    ({ tags, spans } = mergeCueEvidence(tags, spans, matchCues(transcript, weights.lexicon.cues, hardIds)));

    return { risk, tags, attributions: spans, backend: BACKEND, raw_confidence: Math.max(risk, 1 - risk) };
  };
}

/** Port of `predict._merge_cue_evidence`: a verbatim cue hit tags its tactic at weight 1.0
    (added or upgraded) and contributes its span; spans dedup by (start,end), sorted by start. */
export function mergeCueEvidence(tags, spans, cueMatches) {
  const tagIds = new Set(tags.map((t) => t.id));
  const mergedTags = [...tags];
  const spanKeys = new Set(spans.map((s) => `${s.start}:${s.end}`));
  const mergedSpans = [...spans];
  for (const m of cueMatches) {
    if (!tagIds.has(m.tacticId)) {
      mergedTags.push({ id: m.tacticId, weight: 1 });
      tagIds.add(m.tacticId);
    }
    const key = `${m.span.start}:${m.span.end}`;
    if (!spanKeys.has(key)) {
      mergedSpans.push(m.span);
      spanKeys.add(key);
    }
  }
  const cueIds = new Set(cueMatches.map((m) => m.tacticId));
  return {
    tags: mergedTags.map((t) => (cueIds.has(t.id) ? { id: t.id, weight: 1 } : t)),
    spans: mergedSpans.sort((a, b) => a.start - b.start),
  };
}
