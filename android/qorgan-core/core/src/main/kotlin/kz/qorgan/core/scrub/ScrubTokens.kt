package kz.qorgan.core.scrub

import java.util.Locale

/**
 * Tokens for the context-aware scrub rules -- port of `site/core/scrub-tokens.js`. Offsets are
 * CODE-POINT indices (like Python and the JS `Array.from(text)`), so emoji cannot shift them.
 */
internal object ScrubTokens {
    val PLACEHOLDERS = mapOf(
        "phone" to "[PHONE]", "card" to "[CARD]", "iin" to "[IIN]", "email" to "[EMAIL]",
        "iban" to "[IBAN]", "number" to "[NUMBER]", "name" to "[NAME]",
    )

    enum class Kind { DIGITS, LETTERS }

    data class Token(val text: String, val start: Int, val end: Int, val kind: Kind, val folded: String)

    fun fold(word: String): String = word.lowercase(Locale.ROOT).replace("ё", "е")

    /** The text as code points, each a String (the JS `Array.from(text)`). */
    fun chars(text: String): List<String> {
        val out = ArrayList<String>(text.length)
        var i = 0
        while (i < text.length) {
            val cp = text.codePointAt(i)
            out.add(String(Character.toChars(cp)))
            i += Character.charCount(cp)
        }
        return out
    }

    private fun charKind(c: String): Kind? {
        val cp = c.codePointAt(0)
        if (cp in '0'.code..'9'.code) return Kind.DIGITS
        if (Character.isLetter(cp)) return Kind.LETTERS
        return null
    }

    fun tokenize(chars: List<String>): List<Token> {
        val tokens = ArrayList<Token>()
        var start = 0
        var kind: Kind? = null
        fun push(end: Int) {
            val text = chars.subList(start, end).joinToString("")
            tokens.add(Token(text, start, end, kind!!, fold(text)))
        }
        chars.forEachIndexed { index, c ->
            val current = charKind(c)
            if (current != kind) {
                if (kind != null) push(index)
                start = index
                kind = current
            }
        }
        if (kind != null) push(chars.size)
        return tokens
    }

    fun isCapitalised(word: String): Boolean {
        if (word.isEmpty()) return false
        val first = String(Character.toChars(word.codePointAt(0)))
        return first != first.lowercase(Locale.ROOT) && first == first.uppercase(Locale.ROOT)
    }

    fun slice(chars: List<String>, start: Int, end: Int): String = chars.subList(start, end).joinToString("")

    data class Replacement(val start: Int, val end: Int, val placeholder: String)

    fun applyReplacements(chars: List<String>, replacements: List<Replacement>): String {
        if (replacements.isEmpty()) return chars.joinToString("")
        val sorted = replacements.sortedWith(compareBy<Replacement> { it.start }.thenBy { it.end })
        val out = StringBuilder()
        var cursor = 0
        for (r in sorted) {
            out.append(slice(chars, cursor, r.start)).append(r.placeholder)
            cursor = r.end
        }
        out.append(slice(chars, cursor, chars.size))
        return out.toString()
    }
}
