package kz.qorgan.core.session

import kz.qorgan.core.model.MeterConfig

/** The 0-100 suspicion meter -- port of `site/core/meter.js`. Pure; same arithmetic order. */
object Meter {
    fun band(score: Double, m: MeterConfig): Band {
        require(score >= 0 && score <= m.bands.scoreMax) { "score out of range: $score" }
        return when {
            score <= m.bands.lowMax -> Band.LOW
            score <= m.bands.mediumMax -> Band.MEDIUM
            score <= m.bands.highMax -> Band.HIGH
            else -> Band.CRITICAL
        }
    }

    fun update(
        state: MeterState,
        risk: Double,
        asrConfidence: Double,
        hardSignals: Map<String, Double>,
        m: MeterConfig,
    ): MeterState {
        require(risk >= 0 && risk <= 1) { "risk out of range: $risk" }
        require(asrConfidence >= 0 && asrConfidence <= 1) { "asrConfidence out of range: $asrConfidence" }
        val turn = state.turnIndex + 1
        val target = m.bands.scoreMax * risk
        val alpha = if (target > state.score) {
            m.alphaUp * minOf(1.0, turn.toDouble() / m.shortWindowTurns)
        } else {
            m.alphaDown
        }
        var score = state.score + alpha * asrConfidence * (target - state.score)

        val confident = hardSignals.filterValues { it >= m.hardSignalConfidenceFloor }.keys.toList()
        val accumulated = (state.hardSignalIds + confident).distinct().sorted()
        if (accumulated.size >= 2) score = maxOf(score, m.doubleHardSignalFloor)
        else if (accumulated.size == 1) score = maxOf(score, m.singleHardSignalFloor)
        score = minOf(maxOf(score, 0.0), m.bands.scoreMax)

        val enter = m.enter * m.bands.scoreMax
        val exit = m.exit * m.bands.scoreMax
        // No latch before `minTurnsToArm` utterances unless a hard signal is on record.
        val armed = turn >= m.minTurnsToArm || accumulated.isNotEmpty()
        val latched = if (state.latched) score > exit else armed && score >= enter
        val before = state.hardSignalIds.toSet()
        val event = MeterEvent(
            turnIndex = turn,
            risk = risk,
            asrConfidence = asrConfidence,
            scoreBefore = state.score,
            scoreAfter = score,
            newHardSignalIds = confident.filter { it !in before }.sorted(),
        )
        return MeterState(score, latched, turn, accumulated, state.ledger + event)
    }
}
