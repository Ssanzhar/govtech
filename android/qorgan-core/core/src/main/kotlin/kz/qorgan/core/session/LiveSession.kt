package kz.qorgan.core.session

import kz.qorgan.core.model.QorganConfig
import kz.qorgan.core.model.WindowConfig

/**
 * Per-utterance live pipeline -- port of `site/core/session.js`: utterance -> rolling window
 * (head + tail) -> score -> meter -> evidence -> advice. Pure: `advance` returns a new state.
 */
object LiveSession {
    fun initial(locale: String, config: QorganConfig): SessionState {
        require(locale in config.locales) { "Unsupported locale $locale" }
        return SessionState(locale, emptyList(), MeterState.INITIAL, emptyList(), emptyList())
    }

    fun rollingWindow(utterances: List<String>, w: WindowConfig): String {
        val full = utterances.joinToString(w.join)
        if (full.length <= w.maxChars) return full
        val head = utterances.take(w.headUtterances)
        var budget = w.maxChars - head.joinToString(w.join).length - w.join.length
        val tail = ArrayDeque<String>()
        val rest = utterances.drop(w.headUtterances)
        for (i in rest.indices.reversed()) {
            val cost = rest[i].length + w.join.length
            if (budget - cost < 0 && tail.isNotEmpty()) break
            tail.addFirst(rest[i])
            budget -= cost
        }
        return (head + tail).joinToString(w.join)
    }

    private fun accumulateTags(existing: List<Tag>, fresh: List<Tag>): List<Tag> {
        val weights = LinkedHashMap<String, Double>()
        existing.forEach { weights[it.id] = it.weight }
        fresh.forEach { weights[it.id] = maxOf(weights[it.id] ?: 0.0, it.weight) }
        return weights.map { (id, w) -> Tag(id, w) }
    }

    fun advance(
        state: SessionState,
        text: String,
        confidence: Double,
        score: (String) -> ScoreResult,
        config: QorganConfig,
    ): Advance {
        val utterances = state.utterances + text
        val windowText = rollingWindow(utterances, config.window)
        val result = score(windowText)
        val hard = config.hardSignalIds.toSet()
        val hardSignals = result.tags.filter { it.id in hard }.associate { it.id to it.weight }
        val meter = Meter.update(state.meter, result.risk, confidence, hardSignals, config.meter)
        val band = Meter.band(meter.score, config.meter)
        val tags = accumulateTags(state.tags, result.tags)
        val seen = state.seenEvidence.toSet()
        val newEvidence = result.attributions.filter { it.text !in seen }
        val recommendation = if (band != Band.LOW) {
            Advice.recommend(tags, state.locale, result.rawConfidence, config)
        } else {
            Recommendation.EMPTY
        }
        val next = SessionState(state.locale, utterances, meter, tags, state.seenEvidence + newEvidence.map { it.text })
        return Advance(next, SessionUpdate(meter, band, windowText, result, newEvidence, recommendation))
    }
}
