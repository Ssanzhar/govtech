package kz.qorgan.core

import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import kz.qorgan.core.scrub.Scrubber
import org.junit.Assert.assertEquals
import org.junit.Test

/** Kotlin twin of the scrub tests in `tests_js/report.test.mjs` over `tests_js/fixtures/scrub.json`. */
class ScrubTest {
    private val cases = Fixtures.json("tests_js/fixtures/scrub.json").getValue("cases").jsonArray.map {
        val o = it.jsonObject
        o.getValue("input").jsonPrimitive.content to o.getValue("expected").jsonPrimitive.content
    }

    @Test
    fun reproducesPythonScrubTextOnEveryFixtureCase() {
        val failures = cases.filter { (input, expected) -> Scrubber.scrub(input) != expected }
            .map { (input, expected) -> "in=$input\n want=$expected\n  got=${Scrubber.scrub(input)}" }
        println("ScrubTest: ${cases.size} cases, ${failures.size} mismatches")
        assertEquals(failures.joinToString("\n"), 0, failures.size)
    }

    @Test
    fun isIdempotent() {
        for ((input, _) in cases) {
            val once = Scrubber.scrub(input)
            assertEquals(input, once, Scrubber.scrub(once))
        }
    }
}
