package kz.qorgan.core.scrub

import kz.qorgan.core.scrub.ScrubTokens.Kind
import kz.qorgan.core.scrub.ScrubTokens.Replacement
import kz.qorgan.core.scrub.ScrubTokens.Token

/** Self-introduced personal names -- port of `site/core/scrub-names.js` (ADR D66). */
internal object ScrubNames {
    private val STRONG_TRIGGERS = listOf(
        listOf("меня", "зовут"), listOf("мое", "имя"), listOf("моя", "фамилия"), listOf("атым"),
        listOf("есімім"), listOf("есимим"), listOf("my", "name", "is"),
    )
    private val WEAK_TRIGGERS = listOf(listOf("это"), listOf("this", "is"), listOf("i", "m"), listOf("i", "am"))
    private val WEAK_FOLLOWERS = setOf("из", "с", "со", "от", "from", "calling", "with")
    private val STOP_WORDS = setOf(
        // ru
        "не", "я", "мы", "вы", "он", "она", "они", "это", "из", "с", "со", "от", "и", "а", "но",
        "ваш", "ваша", "вам", "вас", "так", "как", "что", "сотрудник", "сотрудница", "служба",
        "службы", "банк", "банка", "отдел", "полиция", "оператор", "специалист", "менеджер",
        "нацбанк", "казпочта", "налоговая", "компания",
        // kk
        "ба", "бе", "ма", "ме", "па", "пе", "мен", "сіз", "біз", "банк", "банктен", "қызметкер",
        "қызметкері", "қызметі",
        // en
        "the", "a", "an", "from", "calling", "with", "and", "i", "not", "your", "bank", "service",
        "security", "officer", "agent", "here", "just", "is", "post", "kazakhstan", "казахстан",
        // brands callers impersonate: evidence, never a personal name
        "kaspi", "каспи", "halyk", "халык", "халық", "homebank", "kazpost", "казпост", "kcell",
        "кселл", "beeline", "билайн", "tele", "теле", "egov", "егов", "jusan", "жусан", "forte",
        "форте", "bereke", "береке", "freedom", "фридом", "kazakhtelecom", "казахтелеком",
    )
    private val TITLES = setOf(
        "майор", "капитан", "лейтенант", "старший", "полковник", "подполковник", "сержант",
        "следователь", "инспектор", "господин", "госпожа", "major", "captain", "lieutenant",
        "colonel", "sergeant", "inspector", "detective", "mr", "mrs", "ms", "dr",
    )
    private val SURNAME_ENDINGS = listOf(
        "ов", "ова", "ев", "ева", "ина", "ын", "ына", "ский", "ская", "цкий", "цкая", "енко", "ұлы",
        "қызы", "улы", "кызы", "ov", "ova", "ev", "eva", "ina", "sky", "skaya", "enko", "uly", "kyzy",
    )
    private const val MAX_EXTRA_NAME_WORDS = 2
    private const val MIN_NAME_CHARS = 2
    private const val MIN_SURNAME_CHARS = 4
    private val TRIGGER_GAP_CHARS = setOf(" ", "\t", "\u00A0", "'", "\u2019")
    private val INTRO_GAP_CHARS = setOf(" ", "\t", "\u00A0", ":", "\u2014", "\u2013", "-")
    private const val MAX_INTRO_GAP = 4
    private val NAME_GAP_CHARS = setOf(" ", "\t", "\u00A0")
    private const val MAX_NAME_GAP = 2

    private fun charCount(text: String) = text.codePointCount(0, text.length)

    private fun gapOk(gap: String, allowed: Set<String>, limit: Int): Boolean {
        val chars = ScrubTokens.chars(gap)
        return chars.isNotEmpty() && chars.size <= limit && chars.all { it in allowed }
    }

    private fun gap(chars: List<String>, a: Token, b: Token) = ScrubTokens.slice(chars, a.end, b.start)

    private fun matchesTrigger(chars: List<String>, tokens: List<Token>, index: Int, trigger: List<String>): Boolean {
        if (index + trigger.size > tokens.size) return false
        val window = tokens.subList(index, index + trigger.size)
        if (window.withIndex().any { (i, t) -> t.folded != trigger[i] }) return false
        for (i in 1 until window.size) {
            if (!gapOk(gap(chars, window[i - 1], window[i]), TRIGGER_GAP_CHARS, MAX_INTRO_GAP)) return false
        }
        return true
    }

    private fun nameWord(t: Token) = t.kind == Kind.LETTERS && charCount(t.text) >= MIN_NAME_CHARS && t.folded !in STOP_WORDS

    private fun looksLikeSurname(t: Token) =
        charCount(t.text) >= MIN_SURNAME_CHARS && SURNAME_ENDINGS.any { t.folded.endsWith(it) }

    private fun strongName(chars: List<String>, tokens: List<Token>, start: Int): Int {
        if (start >= tokens.size || !nameWord(tokens[start])) return 0
        var count = 1
        while (count <= MAX_EXTRA_NAME_WORDS && start + count < tokens.size) {
            val token = tokens[start + count]
            if (!gapOk(gap(chars, tokens[start + count - 1], token), NAME_GAP_CHARS, MAX_NAME_GAP)) break
            if (!(nameWord(token) && (ScrubTokens.isCapitalised(token.text) || looksLikeSurname(token)))) break
            count += 1
        }
        return count
    }

    private fun weakName(chars: List<String>, tokens: List<Token>, start: Int): Int {
        for (count in listOf(2, 1)) {
            val end = start + count
            if (end >= tokens.size) continue
            val names = tokens.subList(start, end)
            if (!names.all { nameWord(it) && ScrubTokens.isCapitalised(it.text) }) continue
            if (count == 2 && !looksLikeSurname(names[1])) continue
            val gapsOk = (start..end).all { i -> gapOk(gap(chars, tokens[i - 1], tokens[i]), NAME_GAP_CHARS, MAX_NAME_GAP) }
            if (gapsOk && tokens[end].folded in WEAK_FOLLOWERS) return count
        }
        return 0
    }

    private fun nameAt(chars: List<String>, tokens: List<Token>, index: Int): Pair<Int, Int> {
        for (trigger in STRONG_TRIGGERS) {
            if (!matchesTrigger(chars, tokens, index, trigger)) continue
            val start = index + trigger.size
            if (start < tokens.size && gapOk(gap(chars, tokens[start - 1], tokens[start]), INTRO_GAP_CHARS, MAX_INTRO_GAP)) {
                val count = strongName(chars, tokens, start)
                if (count > 0) return start to count
            }
        }
        for (trigger in WEAK_TRIGGERS) {
            if (matchesTrigger(chars, tokens, index, trigger)) {
                var start = index + trigger.size
                if (start + 1 < tokens.size && tokens[start].folded in TITLES) start += 1 // the title stays
                val count = if (start < tokens.size) weakName(chars, tokens, start) else 0
                if (count > 0) return start to count
            }
        }
        return 0 to 0
    }

    fun maskNames(text: String): String {
        val chars = ScrubTokens.chars(text)
        val tokens = ScrubTokens.tokenize(chars)
        val replacements = ArrayList<Replacement>()
        var index = 0
        while (index < tokens.size) {
            val (start, count) = nameAt(chars, tokens, index)
            if (count > 0) {
                replacements.add(Replacement(tokens[start].start, tokens[start + count - 1].end, ScrubTokens.PLACEHOLDERS.getValue("name")))
                index = start + count
            } else {
                index += 1
            }
        }
        return ScrubTokens.applyReplacements(chars, replacements)
    }
}
