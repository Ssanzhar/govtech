package kz.qorgan.core.session

import kz.qorgan.core.model.QorganConfig

class AdviceError(message: String) : IllegalArgumentException(message)

/** Tactic -> advice, port of `site/core/recommend.js`. Texts come from the reviewed config only. */
object Advice {
    fun recommend(tags: List<Tag>, locale: String, confidence: Double?, config: QorganConfig): Recommendation {
        if (locale !in config.locales) throw AdviceError("Unsupported locale $locale")
        if (tags.isEmpty()) return Recommendation.EMPTY
        val advice = config.advice[locale] ?: throw AdviceError("no advice for locale $locale")
        val hard = config.hardSignalIds.toSet()
        val ranked = tags
            .filter { it.id in config.tactics }
            .withIndex()
            .sortedWith(
                compareBy<IndexedValue<Tag>> { if (it.value.id in hard) 0 else 1 }
                    .thenByDescending { it.value.weight }
                    .thenBy { it.index },
            )
            .map { it.value }
        val advices = ranked.mapNotNull { advice.tacticAdvice[it.id] }.distinct()
        if (confidence != null && confidence < config.scoring.lowConfidenceFloor) {
            return Recommendation(emptyList(), advice.verificationQuestions, true, advice.lowConfidenceNote)
        }
        return Recommendation(advices, advice.verificationQuestions, false, null)
    }
}
