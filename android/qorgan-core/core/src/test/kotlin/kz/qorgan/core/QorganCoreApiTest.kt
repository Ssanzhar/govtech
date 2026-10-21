package kz.qorgan.core

import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import kz.qorgan.core.embed.CachingEmbedder
import kz.qorgan.core.session.Embedder
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

/** The public facade over the parity-tested parts (recorded embeddings, no model needed). */
class QorganCoreApiTest {
    private val core = QorganCore.fromJson(Fixtures.configText, Fixtures.weightsText, Fixtures.recordedEmbedder)
    private val cases = Fixtures.parity.getValue("cases").jsonArray.map { it.jsonObject }

    @Test
    fun feedReproducesTheGoldenTrajectory() {
        for (c in cases) {
            val session = core.newSession(c.getValue("locale").jsonPrimitive.content)
            val turns = c.getValue("live").jsonObject.getValue("turns").jsonArray.map { it.jsonObject }
            c.getValue("utterances").jsonArray.forEachIndexed { i, u ->
                val a = session.feed(u.jsonPrimitive.content)
                assertEquals(i + 1, a.turn)
                assertEquals(turns[i].getValue("band").jsonPrimitive.content, a.band.wire)
                assertEquals(turns[i].getValue("latched").jsonPrimitive.content.toBoolean(), a.latched)
            }
        }
    }

    @Test
    fun cachingEmbedderIsTransparentAndBounded() {
        var calls = 0
        val counting = Embedder { texts -> calls += texts.size; Fixtures.recordedEmbedder.embed(texts) }
        val cached = CachingEmbedder(counting, capacity = 2)
        val text = cases[0].getValue("transcript").jsonPrimitive.content
        val first = cached.embed(listOf(text, text))
        val again = cached.embed(listOf(text))
        assertEquals(1, calls)
        assertTrue(first[0].contentEquals(again[0]))
    }

    @Test
    fun scrubIsExposed() {
        assertEquals("Перезвоните на [PHONE] пожалуйста", core.scrub("Перезвоните на +7 777 123 45 67 пожалуйста"))
    }
}
