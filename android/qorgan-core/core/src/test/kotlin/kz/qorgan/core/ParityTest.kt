package kz.qorgan.core

import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.boolean
import kotlinx.serialization.json.double
import kotlinx.serialization.json.int
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import kotlinx.serialization.json.contentOrNull
import kz.qorgan.core.session.Explainer
import kz.qorgan.core.session.LiveSession
import kz.qorgan.core.session.Scorer
import kz.qorgan.core.session.Span
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import kotlin.math.abs

/**
 * Golden parity, the Kotlin twin of `tests_js/parity.test.mjs`: on every case of
 * `tests_js/fixtures/parity.json` (Python's outputs on recorded embeddings), the Kotlin core must
 * reproduce score(), explain() for ru + kk, and the live-session trajectory turn by turn.
 * Tolerances are the JS test's own.
 */
class ParityTest {
    private val config = Fixtures.config
    private val parity = Fixtures.parity
    private val cases = parity.getValue("cases").jsonArray.map { it.jsonObject }
    private val scorer = Scorer(Fixtures.weights, config, Fixtures.recordedEmbedder)

    private fun JsonObject.s(k: String) = getValue(k).jsonPrimitive.content
    private fun JsonObject.d(k: String) = getValue(k).jsonPrimitive.double
    private fun JsonObject.strings(k: String) = getValue(k).jsonArray.map { it.jsonPrimitive.content }

    @Test
    fun configKnobsMatchWhatPythonUsed() {
        val c = parity.getValue("config").jsonObject
        assertEquals(c.d("risk_threshold"), config.scoring.riskThreshold, 0.0)
        assertEquals(c.d("enter"), config.meter.enter, 0.0)
        assertEquals(c.d("exit"), config.meter.exit, 0.0)
        assertEquals(c.d("alpha_up"), config.meter.alphaUp, 0.0)
        assertEquals(c.d("alpha_down"), config.meter.alphaDown, 0.0)
        assertTrue(cases.size >= 20)
    }

    @Test
    fun scoreMatchesPythonOnEveryCase() {
        var maxDelta = 0.0
        for (c in cases) {
            val id = c.s("id")
            val want = c.getValue("score").jsonObject
            val r = scorer.score(c.s("transcript"))
            maxDelta = maxOf(maxDelta, abs(r.risk - want.d("risk")))
            assertEquals("$id risk", want.d("risk"), r.risk, 1e-6)
            assertEquals("$id confidence", want.d("raw_confidence"), r.rawConfidence, 1e-6)
            val wantTags = want.getValue("tags").jsonArray.map { it.jsonObject }
            assertEquals("$id tag ids", wantTags.map { it.s("id") }, r.tags.map { it.id })
            r.tags.forEachIndexed { i, t -> assertEquals("$id weight ${t.id}", wantTags[i].d("weight"), t.weight, 1e-6) }
            val wantSpans = want.getValue("attributions").jsonArray.map {
                val o = it.jsonObject
                Span(o.s("text"), o.getValue("start").jsonPrimitive.int, o.getValue("end").jsonPrimitive.int)
            }
            assertEquals("$id attributions", wantSpans, r.attributions)
        }
        println("ParityTest.score: ${cases.size} cases, max |Δrisk| = $maxDelta")
    }

    @Test
    fun explainMatchesPythonForRuAndKk() {
        var checked = 0
        for (c in cases) {
            val r = scorer.score(c.s("transcript"))
            for (locale in listOf("ru", "kk")) {
                val e = Explainer.explain(r, c.s("transcript"), locale, config)
                val want = c.getValue("explanations").jsonObject.getValue(locale).jsonObject
                val tag = "${c.s("id")}/$locale"
                assertEquals("$tag reason", want.s("reason"), e.reason)
                assertEquals("$tag confidence label", want.s("confidence_label"), e.confidenceLabel)
                assertEquals("$tag caveat", want.s("caveat"), e.caveat)
                assertEquals("$tag human note", want.s("human_note"), e.humanNote)
                assertEquals("$tag highlights", want.strings("highlights"), e.highlights.map { it.text })
                checked += 1
            }
        }
        println("ParityTest.explain: $checked explanations identical")
    }

    @Test
    fun liveSessionTrajectoryMatchesPythonTurnByTurn() {
        val tol = 1e-3 // meter scores are 100 x risk; see tests_js/parity.test.mjs
        var turns = 0
        var maxScoreDelta = 0.0
        for (c in cases) {
            val id = c.s("id")
            var state = LiveSession.initial(c.s("locale"), config)
            val utterances = c.strings("utterances")
            val wantTurns = c.getValue("live").jsonObject.getValue("turns").jsonArray.map { it.jsonObject }
            utterances.forEachIndexed { i, text ->
                val want = wantTurns[i]
                val out = LiveSession.advance(state, text, 1.0, scorer::score, config)
                state = out.state
                val u = out.update
                val at = "$id turn ${i + 1}"
                assertEquals("$at window", want.s("window_text"), u.windowText)
                maxScoreDelta = maxOf(maxScoreDelta, abs(u.meter.score - want.d("score")))
                assertEquals("$at score", want.d("score"), u.meter.score, tol)
                assertEquals("$at latched", want.getValue("latched").jsonPrimitive.boolean, u.meter.latched)
                assertEquals("$at hard signals", want.strings("hard_signal_ids"), u.meter.hardSignalIds)
                assertEquals("$at band", want.s("band"), u.band.wire)
                assertEquals("$at risk", want.d("risk"), u.result.risk, 1e-6)
                assertEquals("$at evidence", want.strings("new_evidence"), u.newEvidence.map { it.text })
                val rec = want.getValue("recommendation").jsonObject
                assertEquals("$at advices", rec.strings("advices"), u.recommendation.advices)
                assertEquals("$at questions", rec.strings("verification_questions"), u.recommendation.verificationQuestions)
                assertEquals("$at softened", rec.getValue("softened").jsonPrimitive.boolean, u.recommendation.softened)
                assertEquals("$at note", rec.getValue("note").jsonPrimitive.contentOrNull, u.recommendation.note)
                turns += 1
            }
            val wantTags = c.getValue("live").jsonObject.getValue("tags").jsonArray.map { it.jsonObject.s("id") }.sorted()
            assertEquals("$id accumulated tags", wantTags, state.tags.map { it.id }.sorted())
        }
        println("ParityTest.live: ${cases.size} sessions, $turns turns identical; max |Δmeter score| = $maxScoreDelta")
    }
}
