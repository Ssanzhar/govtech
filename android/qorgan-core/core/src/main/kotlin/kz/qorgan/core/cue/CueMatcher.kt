package kz.qorgan.core.cue

import java.text.Normalizer
import java.util.Locale

/**
 * Cue matching that survives the speech recogniser -- a 1:1 port of `site/core/cue-match.js`
 * (itself a port of `qorgan/classifier/cue_match.py`, ADR D39). Offsets are UTF-16 code units,
 * exactly like the JS port. Pinned by `tests_js/fixtures/cue_match.json`.
 */
object CueMatcher {
    const val MATCHER_VERSION = 2
    const val UTTERANCE_BOUNDARY = "\n"

    // Letters and digits only (spaces dropped). Kazakh letters listed explicitly.
    private val KEEP: Set<Char> = buildSet {
        addAll('0'..'9'); addAll('a'..'z'); addAll('а'..'я')
        addAll(listOf('ё', 'ә', 'ғ', 'қ', 'ң', 'ө', 'ұ', 'ү', 'һ', 'і'))
    }
    private val BUDGET_STEPS = listOf(12 to 0, 21 to 1, 31 to 2)
    private const val MAX_BUDGET = 3

    fun budgetFor(length: Int): Int = BUDGET_STEPS.firstOrNull { length < it.first }?.second ?: MAX_BUDGET

    /** Normalised de-spaced text plus, per normalised char, its offset in the original string. */
    fun normalize(text: String): Pair<String, IntArray> {
        val out = StringBuilder()
        val map = ArrayList<Int>()
        for (index in text.indices) {
            val folded = Normalizer.normalize(text[index].toString(), Normalizer.Form.NFKC)
                .lowercase(Locale.ROOT).replace('ё', 'е')
            for (piece in folded) {
                if (piece in KEEP) {
                    out.append(piece)
                    map.add(index)
                }
            }
        }
        return out.toString() to map.toIntArray()
    }

    /** `[start, end)` of `cue` in `text`, or null. */
    fun findCue(text: String, cue: String, prefilter: Boolean = true): IntRange? {
        if (text.isEmpty() || cue.isEmpty()) return null
        val exact = text.lowercase(Locale.ROOT).indexOf(cue.lowercase(Locale.ROOT))
        if (exact != -1) return exact until exact + cue.length

        val normalizedCue = normalize(cue).first
        val budget = budgetFor(normalizedCue.length)
        if (normalizedCue.isEmpty() || budget == 0) return null

        var offset = 0
        for (segment in text.split(UTTERANCE_BOUNDARY)) {
            val found = findInSegment(segment, normalizedCue, budget, prefilter)
            if (found != null) return (offset + found.first) until (offset + found.second)
            offset += segment.length + UTTERANCE_BOUNDARY.length
        }
        return null
    }

    private fun findInSegment(segment: String, cue: String, budget: Int, prefilter: Boolean): Pair<Int, Int>? {
        val (normalizedText, indexMap) = normalize(segment)
        if (normalizedText.isEmpty()) return null
        if (prefilter && !mayMatch(cue, normalizedText, budget)) return null
        val (start, end) = bestWindow(cue, normalizedText, budget) ?: return null
        return indexMap[start] to indexMap[end - 1] + 1
    }

    /** Pigeonhole prefilter: one of `budget + 1` disjoint cue blocks must survive verbatim. */
    private fun mayMatch(cue: String, text: String, budget: Int): Boolean {
        val blocks = budget + 1
        val size = cue.length / blocks
        val remainder = cue.length % blocks
        if (size == 0) return true
        var start = 0
        for (index in 0 until blocks) {
            val end = start + size + (if (index < remainder) 1 else 0)
            if (text.contains(cue.substring(start, end))) return true
            start = end
        }
        return false
    }

    /** Levenshtein with a free start; ties: substitution -> deletion -> insertion, earliest end. */
    private fun bestWindow(cue: String, text: String, budget: Int): Pair<Int, Int>? {
        val n = text.length
        var costs = IntArray(n + 1)
        var starts = IntArray(n + 1) { it }
        for (i in 1..cue.length) {
            val prevCosts = costs
            val prevStarts = starts
            costs = IntArray(n + 1)
            starts = IntArray(n + 1)
            costs[0] = i
            for (j in 1..n) {
                val substitute = prevCosts[j - 1] + if (cue[i - 1] == text[j - 1]) 0 else 1
                val del = prevCosts[j] + 1
                val ins = costs[j - 1] + 1
                val best = minOf(substitute, del, ins)
                costs[j] = best
                starts[j] = when (best) {
                    substitute -> prevStarts[j - 1]
                    del -> prevStarts[j]
                    else -> starts[j - 1]
                }
            }
        }
        var bestCost = budget + 1
        var bestEnd = -1
        for (j in 1..n) {
            if (costs[j] < bestCost) {
                bestCost = costs[j]
                bestEnd = j
            }
        }
        if (bestEnd < 0 || bestCost > budget) return null
        val start = starts[bestEnd]
        return if (bestEnd > start) start to bestEnd else null
    }
}
