package kz.qorgan.core.gate

import kotlinx.serialization.json.Json
import kotlinx.serialization.json.boolean
import kotlinx.serialization.json.double
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import kz.qorgan.core.model.QorganConfig
import kz.qorgan.core.model.Weights
import kz.qorgan.core.session.Embedder
import kz.qorgan.core.session.Scorer
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.abs

/**
 * The runtime gate (ADR D32/D33) as a library function, so the same measurement runs in a JVM unit
 * test on the Mac and in an instrumented test on a phone. Input: `tests_js/fixtures/runtime_gate.json`
 * (Python's verdicts on 200 transcripts) and, optionally, `runtime_gate_browser.f32` (the browser's
 * WASM embeddings of the same transcripts -- the runtime the shipped heads were trained on).
 * Diagnostics only: nothing is stored.
 */
object RuntimeGate {
    const val MAX_FLIP_RATE = 0.03
    const val MAX_RISK_DELTA_P95 = 0.15
    const val MIN_TAG_JACCARD = 0.9

    data class Comparison(val flips: List<String>, val p95: Double, val max: Double, val jaccard: Double?)

    data class Result(
        val cases: Int,
        val msPerTranscript: Double,
        val vsPython: Comparison,
        val browserCosineMean: Double?,
        val browserCosineMin: Double?,
        val vsBrowser: Comparison?,
    ) {
        val passes: Boolean
            get() = vsPython.flips.size <= (MAX_FLIP_RATE * cases).toInt() &&
                vsPython.p95 < MAX_RISK_DELTA_P95 && (vsPython.jaccard ?: 1.0) >= MIN_TAG_JACCARD

        fun summary(): String = buildString {
            append("flips vs Python=${vsPython.flips.size}/$cases, |Δrisk| p95=${f(vsPython.p95)} max=${f(vsPython.max)}, ")
            append("tag Jaccard=${f(vsPython.jaccard ?: 1.0)}, ${"%.0f".format(msPerTranscript)} ms/transcript")
            if (vsBrowser != null) {
                append("; vs browser WASM: cosine mean=${"%.4f".format(browserCosineMean)} min=${"%.4f".format(browserCosineMin)}, ")
                append("decision flips=${vsBrowser.flips.size}/$cases, |Δrisk| p95=${f(vsBrowser.p95)} max=${f(vsBrowser.max)}")
            }
        }

        private fun f(x: Double) = "%.3f".format(x)
    }

    private data class Case(val id: String, val transcript: String, val risk: Double, val flagged: Boolean, val tags: Set<String>)

    fun run(
        gateJson: String,
        browserF32: ByteArray?,
        weights: Weights,
        config: QorganConfig,
        embedder: Embedder,
    ): Result {
        val cases = Json.parseToJsonElement(gateJson).jsonObject.getValue("cases").jsonArray.map {
            val o = it.jsonObject
            Case(
                o.getValue("id").jsonPrimitive.content,
                o.getValue("transcript").jsonPrimitive.content,
                o.getValue("risk").jsonPrimitive.double,
                o.getValue("flagged").jsonPrimitive.boolean,
                o.getValue("tags").jsonArray.map { t -> t.jsonObject.getValue("id").jsonPrimitive.content }.toSet(),
            )
        }
        val texts = cases.map { it.transcript }
        val t0 = System.nanoTime()
        val mine = embedder.embed(texts)
        val msPer = (System.nanoTime() - t0) / 1e6 / texts.size
        val thr = config.scoring.riskThreshold

        val myScores = score(texts, mine, weights, config)
        val vsPython = compare(cases.map { it.id }, myScores.map { it.first }, cases.map { it.risk }, thr,
            myScores.map { it.second }, cases.map { it.tags })

        if (browserF32 == null) return Result(cases.size, msPer, vsPython, null, null, null)
        val dim = weights.embeddingDim
        val buf = ByteBuffer.wrap(browserF32).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer()
        require(buf.remaining() == texts.size * dim) { "browser capture size does not match the gate: recapture" }
        val browser = texts.indices.map { i -> FloatArray(dim) { buf.get(i * dim + it) } }
        val cos = texts.indices.map { i -> var s = 0.0; for (d in 0 until dim) s += browser[i][d].toDouble() * mine[i][d]; s }
        val browserScores = score(texts, browser, weights, config)
        val vsBrowser = compare(cases.map { it.id }, myScores.map { it.first }, browserScores.map { it.first }, thr, null, null)
        return Result(cases.size, msPer, vsPython, cos.average(), cos.min(), vsBrowser)
    }

    /** Risk + tag ids per transcript; utterances reuse the transcript row (only spans depend on them). */
    private fun score(texts: List<String>, vectors: List<FloatArray>, weights: Weights, config: QorganConfig): List<Pair<Double, Set<String>>> {
        val table = texts.zip(vectors).toMap()
        val scorer = Scorer(weights, config, Embedder { batch -> List(batch.size) { table.getValue(batch[0]) } })
        return texts.map { t -> scorer.score(t).let { r -> r.risk to r.tags.map { it.id }.toSet() } }
    }

    private fun compare(
        ids: List<String>,
        mine: List<Double>,
        ref: List<Double>,
        thr: Double,
        myTags: List<Set<String>>?,
        refTags: List<Set<String>>?,
    ): Comparison {
        val flips = ids.indices.filter { (mine[it] >= thr) != (ref[it] >= thr) }
            .map { "${ids[it]} ref=${"%.3f".format(ref[it])} this=${"%.3f".format(mine[it])}" }
        val deltas = ids.indices.map { abs(mine[it] - ref[it]) }.sorted()
        val jaccard = if (myTags == null || refTags == null) null else ids.indices.map { i ->
            val union = (myTags[i] + refTags[i]).size
            if (union == 0) 1.0 else (myTags[i] intersect refTags[i]).size.toDouble() / union
        }.average()
        return Comparison(flips, deltas[(0.95 * (deltas.size - 1)).toInt()], deltas.last(), jaccard)
    }
}
