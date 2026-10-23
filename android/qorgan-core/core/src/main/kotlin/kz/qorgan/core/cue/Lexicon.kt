package kz.qorgan.core.cue

import kz.qorgan.core.model.ReassurancePatterns
import kz.qorgan.core.session.Span
import java.util.Locale
import java.util.regex.Pattern

/** One grounded hard-signal hit: the tactic and the verbatim span that fired it. */
data class CueMatch(val tacticId: String, val span: Span)

/**
 * Hard-signal cue features + reassurance matcher -- port of `site/core/lexicon.js`. Works over the
 * lexicon block inside weights.json, so the device can never drift from the weights it was
 * trained with.
 */
object Lexicon {
    fun hardSignalFeatures(text: String, cues: Map<String, List<String>>, hardSignalIds: List<String>): DoubleArray =
        DoubleArray(hardSignalIds.size) { i ->
            if (cues[hardSignalIds[i]].orEmpty().any { CueMatcher.findCue(text, it) != null }) 1.0 else 0.0
        }

    fun matchCues(text: String, cues: Map<String, List<String>>, hardSignalIds: List<String>): List<CueMatch> =
        hardSignalIds.mapNotNull { id -> firstGroundedSpan(text, cues[id].orEmpty())?.let { CueMatch(id, it) } }

    private fun firstGroundedSpan(text: String, cueList: List<String>): Span? {
        for (cue in cueList) {
            val found = CueMatcher.findCue(text, cue) ?: continue
            val start = found.first
            val end = found.last + 1
            val candidate = text.substring(start, end)
            // An exact hit must slice back to the cue; a fuzzy hit is the recogniser's own wording.
            val sameLength = candidate.length == cue.length
            if (candidate.isNotEmpty() && (!sameLength || candidate.lowercase(Locale.ROOT) == cue.lowercase(Locale.ROOT))) {
                return Span(candidate, start, end)
            }
        }
        return null
    }

    private const val SENTENCE_STOP = "[^.!?\\n]"

    /** JavaScript's `\s` (the JS port is the reference; Java's `\s` is ASCII-only). */
    internal const val JS_SPACE = "[\\t\\n\\u000B\\f\\r \\u00A0\\u1680\\u2000-\\u200A\\u2028\\u2029\\u202F\\u205F\\u3000\\uFEFF]"

    /** A term as a regex: tokens joined by optional whitespace (ASR glues «не нужно» -> «ненужно»). */
    private fun termPattern(term: String): String =
        term.split(Regex("\\s+")).filter { it.isNotEmpty() }.joinToString("$JS_SPACE*") { Pattern.quote(it) }

    /** Sensitive term within `windowChars` of a reassurance term, either order, never across a stop. */
    fun compileReassurance(p: ReassurancePatterns): Regex {
        val sensitive = "(?:" + p.sensitiveTerms.joinToString("|", transform = ::termPattern) + ")"
        val reassure = "(?:" + p.reassuranceTerms.joinToString("|", transform = ::termPattern) + ")"
        val gap = "$SENTENCE_STOP{0,${p.windowChars}}?"
        return Regex("$sensitive$gap$reassure|$reassure$gap$sensitive", setOf(RegexOption.IGNORE_CASE))
    }

    fun reassuranceFeature(text: String, matcher: Regex): Double = if (matcher.containsMatchIn(text)) 1.0 else 0.0
}
