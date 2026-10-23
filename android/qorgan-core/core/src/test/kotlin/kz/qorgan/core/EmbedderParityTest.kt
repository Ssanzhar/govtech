package kz.qorgan.core

import kotlinx.serialization.json.double
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import kz.qorgan.core.session.Scorer
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Assume.assumeTrue
import org.junit.Test
import kotlin.math.abs

/**
 * Kotlin twin of `tests_js/integration/embedding.test.mjs`: tokenizer (DJL / HF tokenizers) + int8
 * ONNX on the JVM vs the embeddings Python recorded in parity.json. The fixture's `embed_backend`
 * says what to expect: `onnx` (server-native ORT 1.21, the same version pinned here) means
 * bit-for-bit; `device` (browser WASM) means only the documented proxy level.
 * Skipped when the self-hosted model is absent.
 */
class EmbedderParityTest {
    @Test
    fun jvmOnnxReproducesThePythonFixtureEmbeddings() {
        assumeTrue("self-hosted model not downloaded (scripts/deploy_bootstrap.py)", EmbedderSupport.available)
        val parity = Fixtures.parity
        val backend = parity.getValue("embed_backend").jsonPrimitive.content
        val (minCosine, maxRiskDelta) = if (backend == "device") 0.94 to 0.35 else 0.9999 to 1e-3
        val cases = parity.getValue("cases").jsonArray.map { it.jsonObject }
        val texts = cases.map { it.getValue("transcript").jsonPrimitive.content }

        val recorded = Fixtures.recordedEmbedder.embed(texts)
        val t0 = System.nanoTime()
        val mine = EmbedderSupport.embedder.embed(texts)
        val msPer = (System.nanoTime() - t0) / 1e6 / texts.size
        val cosines = mine.indices.map { EmbedderSupport.cosine(mine[it], recorded[it]) }
        var maxAbs = 0.0
        mine.indices.forEach { i -> mine[i].indices.forEach { d -> maxAbs = maxOf(maxAbs, abs(mine[i][d] - recorded[i][d]).toDouble()) } }
        println(
            "EmbedderParity ($backend fixtures): ${texts.size} transcripts, cosine min=${"%.6f".format(cosines.min())} " +
                "mean=${"%.6f".format(cosines.average())}, max |Δcomponent|=${"%.2e".format(maxAbs)}, ${"%.0f".format(msPer)} ms/transcript (Mac JVM)",
        )
        assertTrue("min cosine ${cosines.min()}", cosines.min() >= minCosine)

        // Full scorer on real JVM embeddings (utterances too) vs Python's verdicts.
        val scorer = Scorer(Fixtures.weights, Fixtures.config, EmbedderSupport.embedder)
        var worst = 0.0
        for (c in cases) {
            val r = scorer.score(c.getValue("transcript").jsonPrimitive.content)
            worst = maxOf(worst, abs(r.risk - c.getValue("score").jsonObject.getValue("risk").jsonPrimitive.double))
        }
        println("EmbedderParity: risk |Δ| max=${"%.2e".format(worst)} over ${cases.size} golden cases")
        assertTrue("max risk delta $worst", worst < maxRiskDelta)
    }

    @Test
    fun tokenizerAddsSpecialTokensAndTruncatesAt512() {
        assumeTrue(EmbedderSupport.available)
        val ids = EmbedderSupport.tokenizer.encode("query: Здравствуйте, это служба безопасности банка.")
        assertEquals(0L, ids.first()) // <s>
        assertEquals(2L, ids.last()) // </s>
        val long = EmbedderSupport.tokenizer.encode("query: " + "слово ".repeat(2000))
        assertEquals(512, long.size)
        assertEquals(2L, long.last())
    }
}
