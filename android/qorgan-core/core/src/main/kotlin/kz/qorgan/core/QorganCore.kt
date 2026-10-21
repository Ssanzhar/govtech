package kz.qorgan.core

import kz.qorgan.core.model.QorganConfig
import kz.qorgan.core.model.Weights
import kz.qorgan.core.scrub.Scrubber
import kz.qorgan.core.session.Advance
import kz.qorgan.core.session.Band
import kz.qorgan.core.session.Embedder
import kz.qorgan.core.session.Explainer
import kz.qorgan.core.session.Explanation
import kz.qorgan.core.session.LiveSession
import kz.qorgan.core.session.Recommendation
import kz.qorgan.core.session.ScoreResult
import kz.qorgan.core.session.Scorer
import kz.qorgan.core.session.SessionState
import kz.qorgan.core.session.Span
import kz.qorgan.core.session.Tag

/**
 * Public entry point of the on-device text core (decision support only: it never blocks, reports or
 * hangs up; the person decides). Text in, assessment out; nothing is persisted and nothing leaves
 * the process. Config and weights are the JSON files the web client ships
 * (`site/core/qorgan-config.json`, `site/models/weights.json`).
 */
class QorganCore(val config: QorganConfig, val weights: Weights, val embedder: Embedder) {
    private val scorer = Scorer(weights, config, embedder)

    /** One-shot verdict on a whole transcript (utterances joined by `config.window.join`). */
    fun analyze(transcript: String, locale: String = config.defaultLocale): Analysis {
        val result = scorer.score(transcript)
        return Analysis(result, Explainer.explain(result, transcript, locale, config), result.risk >= config.scoring.riskThreshold)
    }

    /** A live-call session for one call; drop it when the call ends (its state lives in memory only). */
    fun newSession(locale: String = config.defaultLocale): QorganSession = QorganSession(this, LiveSession.initial(locale, config))

    /** Pure step for callers that keep their own state (e.g. in a ViewModel). */
    fun advance(state: SessionState, utteranceText: String, asrConfidence: Double = 1.0): Advance =
        LiveSession.advance(state, utteranceText, asrConfidence, scorer::score, config)

    /** PII scrub for a report draft the citizen will review (the server re-scrubs regardless). */
    fun scrub(text: String): String = Scrubber.scrub(text)

    companion object {
        fun fromJson(configJson: String, weightsJson: String, embedder: Embedder): QorganCore =
            QorganCore(QorganConfig.parse(configJson), Weights.parse(weightsJson), embedder)
    }
}

data class Analysis(val result: ScoreResult, val explanation: Explanation, val flagged: Boolean)

/** What the UI shows after one committed utterance. */
data class Assessment(
    val turn: Int,
    /** Meter score 0-100 (smoothed, hard-signal floors applied). */
    val score: Double,
    val band: Band,
    /** Alert state with hysteresis (enter 0.59 / exit 0.49 x 100), armed from the 3rd utterance. */
    val latched: Boolean,
    /** Calibrated scam probability of the current window (unsmoothed). */
    val windowRisk: Double,
    /** Tactics seen so far in the call (max weight per tactic). */
    val tactics: List<Tag>,
    val hardSignalIds: List<String>,
    /** Verbatim trigger spans of the current window that were not shown before. */
    val newEvidence: List<Span>,
    /** Templated advice from the reviewed YAML (empty while the band is low). */
    val recommendation: Recommendation,
)

/**
 * Stateful convenience wrapper around the pure [LiveSession.advance]: each [feed] replaces the
 * (immutable) state snapshot. Not thread-safe by design: feed utterances from one thread.
 */
class QorganSession internal constructor(private val core: QorganCore, initial: SessionState) {
    var state: SessionState = initial
        private set

    fun feed(utteranceText: String, asrConfidence: Double = 1.0): Assessment {
        val (next, update) = core.advance(state, utteranceText, asrConfidence)
        state = next
        return Assessment(
            turn = update.meter.turnIndex,
            score = update.meter.score,
            band = update.band,
            latched = update.meter.latched,
            windowRisk = update.result.risk,
            tactics = next.tags,
            hardSignalIds = update.meter.hardSignalIds,
            newEvidence = update.newEvidence,
            recommendation = update.recommendation,
        )
    }

    /** The call transcript so far, for the citizen's own post-call review. */
    fun transcript(): String = state.utterances.joinToString(core.config.window.join)
}
