package kz.qorgan.core.session

import kz.qorgan.core.cue.CueMatch
import kz.qorgan.core.cue.CueMatcher
import kz.qorgan.core.cue.Lexicon
import kz.qorgan.core.model.ConfigError
import kz.qorgan.core.model.QorganConfig
import kz.qorgan.core.model.Weights

/** Text -> embedding rows (768-d, L2-normalised, `query: ` prefix applied by the implementation). */
fun interface Embedder {
    fun embed(texts: List<String>): List<FloatArray>
}

/**
 * Port of `site/core/score.js::createScorer` (itself `predict._linear_score`). Pure apart from the
 * injected embedder: nothing is stored, nothing leaves the process.
 */
class Scorer(private val weights: Weights, private val config: QorganConfig, private val embedder: Embedder) {
    init {
        // The cue block is a model input: weights trained under another matcher are refused.
        if (weights.cueMatcherVersion != CueMatcher.MATCHER_VERSION) {
            throw ConfigError(
                "weights were trained with cue matcher v${weights.cueMatcherVersion} but this build uses v${CueMatcher.MATCHER_VERSION}",
            )
        }
    }

    private val hardIds = config.hardSignalIds
    private val reassurance = Lexicon.compileReassurance(weights.reassurance)
    private val join = config.window.join

    private fun featureRow(text: String, embedding: DoubleArray): DoubleArray =
        Heads.hybridRow(
            embedding,
            Lexicon.hardSignalFeatures(text, weights.cues, hardIds),
            Lexicon.reassuranceFeature(text, reassurance),
        )

    fun score(transcript: String): ScoreResult {
        val utterances = transcript.split(join)
        val rows = embedder.embed(listOf(transcript) + utterances)
        if (rows.size != utterances.size + 1) error("embedder returned ${rows.size} rows for ${utterances.size + 1} texts")
        val vectors = rows.map { r -> DoubleArray(r.size) { r[it].toDouble() } }
        val transcriptEmbedding = vectors[0]
        val risk = Heads.riskProba(featureRow(transcript, transcriptEmbedding), weights.riskMembers)
        val probs = Heads.tacticProba(transcriptEmbedding, weights.tacticModels, weights.labelSpace)
        val utteranceScores = DoubleArray(utterances.size) { i ->
            Heads.riskProba(featureRow(utterances[i], vectors[i + 1]), weights.riskMembers)
        }
        val tags = Heads.decodeTactics(probs, weights.labelSpace, weights.tacticThreshold, weights.tacticThresholds)
            .map { (id, p) -> Tag(id, p.coerceIn(0.0, 1.0)) }
        val spans = Heads.selectTopUtteranceSpans(
            utterances, utteranceScores, config.scoring.attributionTopK, config.scoring.attributionMinScore, join,
        )
        val (mergedTags, mergedSpans) = mergeCueEvidence(tags, spans, Lexicon.matchCues(transcript, weights.cues, hardIds))
        return ScoreResult(risk, mergedTags, mergedSpans, BACKEND, maxOf(risk, 1 - risk))
    }

    companion object {
        const val BACKEND = "linear"

        /** A verbatim cue hit tags its tactic at weight 1.0 and contributes its span (dedup by offsets). */
        fun mergeCueEvidence(tags: List<Tag>, spans: List<Span>, cueMatches: List<CueMatch>): Pair<List<Tag>, List<Span>> {
            val tagIds = tags.map { it.id }.toMutableSet()
            val mergedTags = tags.toMutableList()
            val spanKeys = spans.map { it.start to it.end }.toMutableSet()
            val mergedSpans = spans.toMutableList()
            for (m in cueMatches) {
                if (tagIds.add(m.tacticId)) mergedTags.add(Tag(m.tacticId, 1.0))
                if (spanKeys.add(m.span.start to m.span.end)) mergedSpans.add(m.span)
            }
            val cueIds = cueMatches.map { it.tacticId }.toSet()
            return mergedTags.map { if (it.id in cueIds) Tag(it.id, 1.0) else it } to mergedSpans.sortedBy { it.start }
        }
    }
}
