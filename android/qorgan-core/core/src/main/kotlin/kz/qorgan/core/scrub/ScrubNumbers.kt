package kz.qorgan.core.scrub

import kz.qorgan.core.scrub.ScrubTokens.Kind
import kz.qorgan.core.scrub.ScrubTokens.PLACEHOLDERS
import kz.qorgan.core.scrub.ScrubTokens.Replacement
import kz.qorgan.core.scrub.ScrubTokens.Token

/** Numbers dictated as words and split digit groups -- port of `site/core/scrub-numbers.js` (ADR D66). */
internal object ScrubNumbers {
    const val SPOKEN_NUMBER_MIN_DIGITS = 7
    private const val MAX_GAP_CHARS = 3
    private val GAP_CHARS = setOf(" ", "\t", "\r", "\n", "\u00A0", ",", "-")
    private val DIGIT_GROUP_GAPS = setOf(" ", "-")
    private val ASCII_WORD = Regex("^[A-Za-z0-9_]$")
    private val CURRENCY_WORDS = setOf(
        "тенге", "теңге", "тг", "tenge", "kzt", "рублей", "рубля", "руб", "долларов", "доллара",
        "dollars", "usd", "евро", "euro", "сом",
    )
    private val CURRENCY_SIGNS = setOf("₸", "$", "€", "₽")
    private val SPACES = setOf(" ", "\u00A0")

    private fun words(table: Map<Int, List<String>>): Map<String, Int> =
        table.flatMap { (value, list) -> list.map { it to value } }.toMap()

    private val ZERO = setOf("ноль", "нуль", "нөл", "нол", "zero")
    private val UNITS = words(mapOf(
        1 to listOf("один", "одна", "одно", "бір", "бир", "one"),
        2 to listOf("два", "две", "екі", "еки", "two"),
        3 to listOf("три", "үш", "уш", "three"),
        4 to listOf("четыре", "төрт", "торт", "four"),
        5 to listOf("пять", "бес", "five"),
        6 to listOf("шесть", "алты", "six"),
        7 to listOf("семь", "жеті", "жети", "seven"),
        8 to listOf("восемь", "сегіз", "сегиз", "eight"),
        9 to listOf("девять", "тоғыз", "тогыз", "nine"),
    ))
    private val TEENS = words(mapOf(
        10 to listOf("десять", "ten"),
        11 to listOf("одиннадцать", "одинадцать", "eleven"),
        12 to listOf("двенадцать", "twelve"),
        13 to listOf("тринадцать", "thirteen"),
        14 to listOf("четырнадцать", "fourteen"),
        15 to listOf("пятнадцать", "fifteen"),
        16 to listOf("шестнадцать", "sixteen"),
        17 to listOf("семнадцать", "seventeen"),
        18 to listOf("восемнадцать", "eighteen"),
        19 to listOf("девятнадцать", "nineteen"),
    ))
    private val TENS = words(mapOf(
        1 to listOf("он"),
        2 to listOf("двадцать", "жиырма", "twenty"),
        3 to listOf("тридцать", "отыз", "thirty"),
        4 to listOf("сорок", "қырық", "кырык", "forty"),
        5 to listOf("пятьдесят", "пятдесят", "елу", "fifty"),
        6 to listOf("шестьдесят", "шестдесят", "алпыс", "sixty"),
        7 to listOf("семьдесят", "семдесят", "жетпіс", "жетпис", "seventy"),
        8 to listOf("восемьдесят", "восемдесят", "сексен", "eighty"),
        9 to listOf("девяносто", "тоқсан", "токсан", "ninety"),
    ))
    private val HUNDREDS = words(mapOf(
        1 to listOf("сто"), 2 to listOf("двести"), 3 to listOf("триста"), 4 to listOf("четыреста"),
        5 to listOf("пятьсот", "пятсот"), 6 to listOf("шестьсот", "шестсот"), 7 to listOf("семьсот", "семсот"),
        8 to listOf("восемьсот", "восемсот"), 9 to listOf("девятьсот", "девятсот"),
    ))
    private val HUNDRED_MULTIPLIERS = setOf("жүз", "жуз", "hundred")
    private val TABLES = listOf("unit" to UNITS, "teen" to TEENS, "tens" to TENS, "hundreds" to HUNDREDS)

    private fun numberKind(token: Token): Pair<String, Int>? {
        if (token.kind == Kind.DIGITS) return "digits" to 0
        val word = token.folded
        if (word in ZERO) return "zero" to 0
        for ((kind, table) in TABLES) table[word]?.let { return kind to it }
        if (word in HUNDRED_MULTIPLIERS) return "multiplier" to 1
        return null
    }

    /** A pending dictation group: form "H" / "T" / "U" plus the digits seen so far. */
    private data class Group(val form: String, val hundreds: Int?, val tens: Int?, val units: Int?, val teen: Int?)

    private fun flush(g: Group?): String {
        if (g == null) return ""
        return when {
            g.form == "U" -> "${g.units}"
            g.form == "T" -> "${g.tens}${g.units ?: 0}"
            g.teen != null -> "${g.hundreds}${g.teen}"
            else -> "${g.hundreds}${g.tens ?: 0}${g.units ?: 0}"
        }
    }

    private fun step(g: Group?, kind: String, value: Int, text: String): Pair<String, Group?> {
        val form = g?.form
        val openHundreds = form == "H" && g!!.tens == null && g.units == null && g.teen == null
        return when (kind) {
            "digits" -> flush(g) + text to null
            "zero" -> when {
                openHundreds -> "" to Group("H", g!!.hundreds, 0, null, null)
                form == "H" && g!!.tens == 0 && g.units == null -> flush(Group("H", g.hundreds, 0, 0, null)) to null
                else -> flush(g) + "0" to null
            }
            "unit" -> when {
                form == "H" && g!!.units == null && g.teen == null -> flush(Group("H", g.hundreds, g.tens, value, null)) to null
                form == "T" && g!!.units == null -> flush(Group("T", null, g.tens, value, null)) to null
                else -> flush(g) to Group("U", null, null, value, null)
            }
            "teen" -> if (openHundreds) flush(Group("H", g!!.hundreds, null, null, value)) to null else flush(g) + "$value" to null
            "tens" -> if (openHundreds) "" to Group("H", g!!.hundreds, value, null, null) else flush(g) to Group("T", null, value, null, null)
            "hundreds" -> flush(g) to Group("H", value, null, null, null)
            else -> if (form == "U") "" to Group("H", g!!.units, null, null, null) else flush(g) to Group("H", 1, null, null, null)
        }
    }

    fun composeDigits(tokens: List<Token>): String {
        val digits = StringBuilder()
        var group: Group? = null
        for (token in tokens) {
            val (kind, value) = numberKind(token) ?: ("" to 0)
            val (emitted, next) = step(group, kind, value, token.text)
            digits.append(emitted)
            group = next
        }
        digits.append(flush(group))
        return digits.toString()
    }

    private fun joinable(gap: List<String>) = gap.size <= MAX_GAP_CHARS && gap.all { it in GAP_CHARS }

    private fun runs(chars: List<String>, tokens: List<Token>): List<List<Int>> {
        val found = ArrayList<List<Int>>()
        var current = ArrayList<Int>()
        tokens.forEachIndexed { index, token ->
            if (numberKind(token) == null) {
                if (current.isNotEmpty()) found.add(current)
                current = ArrayList()
                return@forEachIndexed
            }
            if (current.isNotEmpty() && !joinable(chars.subList(tokens[current.last()].end, token.start))) {
                found.add(current)
                current = ArrayList()
            }
            current.add(index)
        }
        if (current.isNotEmpty()) found.add(current)
        return found
    }

    private fun placeholderFor(digits: String): String = when {
        digits.length == 16 -> PLACEHOLDERS.getValue("card")
        digits.length == 12 -> PLACEHOLDERS.getValue("iin")
        (digits.length == 11 && digits[0] in "78") || (digits.length == 10 && digits[0] == '7') -> PLACEHOLDERS.getValue("phone")
        else -> PLACEHOLDERS.getValue("number")
    }

    private fun followedByCurrency(chars: List<String>, end: Int, next: Token?): Boolean {
        var cursor = end
        while (cursor < chars.size && chars[cursor] in SPACES) cursor += 1
        if (cursor < chars.size && chars[cursor] in CURRENCY_SIGNS) return true
        if (next == null) return false
        if (!chars.subList(end, next.start).all { it in SPACES }) return false
        return next.folded in CURRENCY_WORDS
    }

    private fun digitGroupsQualify(chars: List<String>, run: List<Token>, digits: String): Boolean {
        if (run.size < 2 || placeholderFor(digits) == PLACEHOLDERS.getValue("number")) return false
        for (i in 1 until run.size) {
            if (ScrubTokens.slice(chars, run[i - 1].end, run[i].start) !in DIGIT_GROUP_GAPS) return false
        }
        val before = if (run[0].start > 0) chars[run[0].start - 1] else ""
        val after = if (run.last().end < chars.size) chars[run.last().end] else ""
        return !ASCII_WORD.matches(before) && !ASCII_WORD.matches(after)
    }

    private fun shouldMask(chars: List<String>, run: List<Token>, digits: String): Boolean =
        if (run.any { it.kind == Kind.LETTERS }) digits.length >= SPOKEN_NUMBER_MIN_DIGITS else digitGroupsQualify(chars, run, digits)

    private fun lineSpans(chars: List<String>, run: List<Token>): List<Pair<Int, Int>> {
        val spans = ArrayList<Pair<Int, Int>>()
        var start = run[0].start
        var end = run[0].end
        for (i in 1 until run.size) {
            if ("\n" in chars.subList(run[i - 1].end, run[i].start)) {
                spans.add(start to end)
                start = run[i].start
            }
            end = run[i].end
        }
        spans.add(start to end)
        return spans
    }

    fun maskNumberRuns(text: String): String {
        val chars = ScrubTokens.chars(text)
        val tokens = ScrubTokens.tokenize(chars)
        val replacements = ArrayList<Replacement>()
        for (indices in runs(chars, tokens)) {
            val run = indices.map { tokens[it] }
            val last = indices.last()
            val next = if (last + 1 < tokens.size) tokens[last + 1] else null
            val digits = composeDigits(run)
            if (!shouldMask(chars, run, digits) || followedByCurrency(chars, run.last().end, next)) continue
            val placeholder = placeholderFor(digits)
            lineSpans(chars, run).forEach { (s, e) -> replacements.add(Replacement(s, e, placeholder)) }
        }
        return ScrubTokens.applyReplacements(chars, replacements)
    }
}
