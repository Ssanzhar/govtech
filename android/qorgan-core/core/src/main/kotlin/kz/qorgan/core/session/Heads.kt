package kz.qorgan.core.session

import kz.qorgan.core.model.LinearModel
import kz.qorgan.core.model.RiskMember
import kotlin.math.exp

/** The exported linear heads -- port of `site/core/head.js`, double precision like sklearn. */
object Heads {
    private fun sigmoid(x: Double): Double = 1.0 / (1.0 + exp(-x))

    private fun dot(coef: DoubleArray, row: DoubleArray): Double {
        require(coef.size == row.size) { "feature width ${row.size} != coefficient width ${coef.size}" }
        var acc = 0.0
        for (i in coef.indices) acc += coef[i] * row[i]
        return acc
    }

    /** Mean over CV members of sigmoid(-(a·d + b)), d = coef·x + b0. */
    fun riskProba(row: DoubleArray, members: List<RiskMember>): Double {
        var total = 0.0
        for (m in members) {
            val decision = dot(m.coef, row) + m.intercept
            total += sigmoid(-(m.calibA * decision + m.calibB))
        }
        return total / members.size
    }

    /** Per-tactic probabilities in label-space order; a tactic without a model scores 0. */
    fun tacticProba(embedding: DoubleArray, models: Map<String, LinearModel>, labelSpace: List<String>): DoubleArray =
        DoubleArray(labelSpace.size) { i ->
            models[labelSpace[i]]?.let { sigmoid(dot(it.coef, embedding) + it.intercept) } ?: 0.0
        }

    /** `[embedding | cue block | reassurance]`. */
    fun hybridRow(embedding: DoubleArray, cueBlock: DoubleArray, reassurance: Double): DoubleArray =
        embedding + cueBlock + doubleArrayOf(reassurance)

    /** (id, prob) at/above threshold, sorted by prob desc then id asc. */
    fun decodeTactics(
        probs: DoubleArray,
        labelSpace: List<String>,
        threshold: Double,
        perTactic: Map<String, Double>,
    ): List<Pair<String, Double>> =
        labelSpace.indices
            .filter { probs[it] >= (perTactic[labelSpace[it]] ?: threshold) }
            .map { labelSpace[it] to probs[it] }
            .sortedWith(compareByDescending<Pair<String, Double>> { it.second }.thenBy { it.first })

    /** Top-k utterances by score as verbatim spans of `utterances.joinToString(join)`, sorted by start. */
    fun selectTopUtteranceSpans(
        utterances: List<String>,
        scores: DoubleArray,
        topK: Int,
        minScore: Double,
        join: String,
    ): List<Span> {
        require(utterances.size == scores.size) { "utterances (${utterances.size}) and scores (${scores.size}) differ" }
        require(topK > 0) { "topK must be > 0, got $topK" }
        data class Candidate(val score: Double, val index: Int, val start: Int, val end: Int)
        val candidates = ArrayList<Candidate>()
        var cursor = 0
        utterances.forEachIndexed { index, u ->
            val start = cursor
            val end = cursor + u.length
            cursor = end + join.length
            if (scores[index] >= minScore && u.isNotBlank()) candidates.add(Candidate(scores[index], index, start, end))
        }
        return candidates
            .sortedWith(compareByDescending<Candidate> { it.score }.thenBy { it.index })
            .take(topK)
            .map { Span(utterances[it.index], it.start, it.end) }
            .sortedBy { it.start }
    }
}
