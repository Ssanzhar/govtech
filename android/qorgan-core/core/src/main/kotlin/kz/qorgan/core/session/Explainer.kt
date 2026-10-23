package kz.qorgan.core.session

import kz.qorgan.core.model.QorganConfig
import kz.qorgan.core.model.Templates

class ExplainerError(message: String) : IllegalArgumentException(message)

/** Templated, grounded explanation (no LLM prose, ever). */
data class Explanation(
    val reason: String,
    val highlights: List<Span>,
    val tags: List<Tag>,
    val confidence: Double,
    val confidenceLabel: String,
    val caveat: String,
    val humanNote: String,
)

/**
 * Port of `site/core/explain.js` minus the display-only role slot (`role.js`, ADR D63), which this
 * spike does not port. Every highlight is re-validated as a verbatim slice of the transcript.
 */
object Explainer {
    fun explain(result: ScoreResult, transcript: String, locale: String, config: QorganConfig): Explanation {
        if (locale !in config.locales) throw ExplainerError("Unsupported locale $locale")
        if (transcript.isBlank()) throw ExplainerError("transcript must not be empty")
        for (span in result.attributions) {
            if (span.end > transcript.length || transcript.substring(span.start, span.end) != span.text) {
                throw ExplainerError("span is not verbatim: ${span.text}")
            }
        }
        val t = config.templates[locale] ?: throw ExplainerError("no templates for $locale")
        val highlights = result.attributions.sortedBy { it.start }
        val names = result.tags.mapNotNull { config.tactics[it.id]?.get(locale) }.filter { it.isNotEmpty() }
        val calibrated = result.backend in config.scoring.calibratedBackends
        return Explanation(
            reason = renderReason(t, names, highlights.map { it.text }),
            highlights = highlights,
            tags = result.tags,
            confidence = result.rawConfidence,
            confidenceLabel = confidenceBand(t, result.rawConfidence, config),
            caveat = if (calibrated) t.caveat else t.caveat + t.uncalibratedSuffix.replace("{backend}", result.backend),
            humanNote = t.humanNote,
        )
    }

    private fun renderReason(t: Templates, names: List<String>, phrases: List<String>): String {
        if (names.isEmpty()) return t.noSignalReason
        val spans = if (phrases.isEmpty()) t.spansPlaceholder else phrases.joinToString(", ") { "«$it»" }
        return t.reasonTemplate.replace("{tags}", names.joinToString(", ")).replace("{spans}", spans)
    }

    private fun confidenceBand(t: Templates, confidence: Double, config: QorganConfig): String {
        val key = when {
            confidence >= config.scoring.confidenceHigh -> "high"
            confidence >= config.scoring.confidenceMedium -> "medium"
            else -> "low"
        }
        return t.confidence.getValue(key)
    }
}
