package kz.qorgan.core.scrub

import kz.qorgan.core.scrub.ScrubTokens.PLACEHOLDERS

/**
 * PII scrubber -- port of `site/core/report.js::scrubText` (itself `qorgan/data/scrub.py`). Used
 * before a citizen reviews a report draft; the server re-scrubs regardless. Pinned by
 * `tests_js/fixtures/scrub.json`.
 */
object Scrubber {
    private const val W = "[\\p{L}\\p{N}_]" // Python's Unicode \w
    private const val B = "(?:(?<!$W)(?=$W)|(?<=$W)(?!$W))" // Python's Unicode \b
    private const val D = "\\p{Nd}"
    private const val EB = "(?<![A-Za-z_\\p{Nd}])"
    private const val EA = "(?![A-Za-z_\\p{Nd}])"
    // JavaScript's `\s` (the JS port is the reference implementation here).
    private const val S = "\\t\\n\\u000B\\f\\r \\u00A0\\u1680\\u2000-\\u200A\\u2028\\u2029\\u202F\\u205F\\u3000\\uFEFF"
    private const val GAP = "[$S\\-]?" // JS `[\\s\\-]?`

    // Most specific first, so a longer digit run is never half-consumed (same order as Python).
    private val RULES: List<Pair<Regex, String>> = listOf(
        Regex("$B[\\p{L}\\p{N}_.+\\-]+@[\\p{L}\\p{N}_\\-]+\\.[\\p{L}\\p{N}_.\\-]+$B") to PLACEHOLDERS.getValue("email"),
        Regex("(?<![A-Za-z0-9_])[Kk][Zz][0-9]{2}(?: ?[A-Za-z0-9]){16}(?![A-Za-z0-9_])") to PLACEHOLDERS.getValue("iban"),
        Regex("$EB(?:$D{16}|$D{4}[ \\-]$D{4}[ \\-]$D{4}[ \\-]$D{4})$EA") to PLACEHOLDERS.getValue("card"),
        Regex("$EB$D{12}$EA") to PLACEHOLDERS.getValue("iin"),
        Regex("(?<!$D)(?:\\+7|8|7)$GAP\\(?$D{3}\\)?$GAP$D{3}$GAP$D{2}$GAP$D{2}(?!$D)") to PLACEHOLDERS.getValue("phone"),
    )

    // JavaScript's String.prototype.trim() whitespace set (WhiteSpace + LineTerminator).
    private const val JS_SPACE_CHARS = "\t\n\u000B\u000C\r \u00A0\u1680\u2028\u2029\u202F\u205F\u3000\uFEFF"
    private fun isJsSpace(c: Char) = c in JS_SPACE_CHARS || c in '\u2000'..'\u200A'

    fun scrub(text: String): String {
        if (text.all(::isJsSpace)) return text
        val digitRules = RULES.fold(text) { acc, (pattern, placeholder) -> pattern.replace(acc, Regex.escapeReplacement(placeholder)) }
        return ScrubNames.maskNames(ScrubNumbers.maskNumberRuns(digitRules))
    }
}
