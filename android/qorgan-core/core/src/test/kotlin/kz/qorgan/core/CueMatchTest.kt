package kz.qorgan.core

import kotlinx.serialization.json.JsonNull
import kotlinx.serialization.json.int
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import kz.qorgan.core.cue.CueMatcher
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

/** Kotlin twin of `tests_js/cue-match.test.mjs` over `tests_js/fixtures/cue_match.json`. */
class CueMatchTest {
    private val fixture = Fixtures.json("tests_js/fixtures/cue_match.json")

    @Test
    fun matcherVersionMatchesTheFixture() {
        assertEquals(fixture.getValue("matcher_version").jsonPrimitive.int, CueMatcher.MATCHER_VERSION)
        assertEquals(Fixtures.weights.cueMatcherVersion, CueMatcher.MATCHER_VERSION)
    }

    @Test
    fun everyFixtureCaseResolvesToTheSameSpanAsPython() {
        val cases = fixture.getValue("cases").jsonArray.map { it.jsonObject }
        var hits = 0
        var mismatches = 0
        val firstFailures = ArrayList<String>()
        for (c in cases) {
            val text = c.getValue("text").jsonPrimitive.content
            val cue = c.getValue("cue").jsonPrimitive.content
            val spanEl = c.getValue("span")
            val want = if (spanEl is JsonNull) null else spanEl.jsonArray.map { it.jsonPrimitive.int }
            val got = CueMatcher.findCue(text, cue)?.let { listOf(it.first, it.last + 1) }
            if (want != null) hits += 1
            if (want != got) {
                mismatches += 1
                if (firstFailures.size < 5) firstFailures.add("cue=$cue want=$want got=$got text=${text.take(70)}")
            }
        }
        println("CueMatchTest: ${cases.size} cases, $hits with a span, $mismatches mismatches")
        assertEquals("mismatches: $firstFailures", 0, mismatches)
        assertTrue(hits > 0)
    }

    @Test
    fun normalisationDeSpacesAndFoldsYo() {
        assertEquals("продиктуйтекодизsmsпожалуйста", CueMatcher.normalize("Продиктуйте КОД из SMS, пожалуйста!").first)
        assertEquals("счет", CueMatcher.normalize("Счёт").first)
    }

    @Test
    fun budgetIsZeroForShortCuesAndMonotonic() {
        assertEquals(0, CueMatcher.budgetFor(7))
        val budgets = listOf(0, 5, 11, 12, 20, 21, 30, 31, 60).map(CueMatcher::budgetFor)
        assertEquals(budgets.sorted(), budgets)
    }
}
