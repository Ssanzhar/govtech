package kz.qorgan.core.session

/** A verbatim slice of the analysed text: `text == source.substring(start, end)` (UTF-16 offsets). */
data class Span(val text: String, val start: Int, val end: Int)

/** A tactic tag with its weight in [0, 1]. */
data class Tag(val id: String, val weight: Double)

/** One scoring of a text window (port of the JS `score()` result). */
data class ScoreResult(
    val risk: Double,
    val tags: List<Tag>,
    val attributions: List<Span>,
    val backend: String,
    val rawConfidence: Double,
)

/** Advice for the citizen; `softened` = below the confidence floor, only verification questions. */
data class Recommendation(
    val advices: List<String>,
    val verificationQuestions: List<String>,
    val softened: Boolean,
    val note: String?,
) {
    companion object {
        val EMPTY = Recommendation(emptyList(), emptyList(), false, null)
    }
}

/** One meter step, kept in the in-memory ledger (never persisted by the core). */
data class MeterEvent(
    val turnIndex: Int,
    val risk: Double,
    val asrConfidence: Double,
    val scoreBefore: Double,
    val scoreAfter: Double,
    val newHardSignalIds: List<String>,
)

data class MeterState(
    val score: Double,
    val latched: Boolean,
    val turnIndex: Int,
    val hardSignalIds: List<String>,
    val ledger: List<MeterEvent>,
) {
    companion object {
        val INITIAL = MeterState(0.0, false, 0, emptyList(), emptyList())
    }
}

enum class Band(val wire: String) { LOW("low"), MEDIUM("medium"), HIGH("high"), CRITICAL("critical") }

/** Immutable live-call state; `advance` returns a new one. */
data class SessionState(
    val locale: String,
    val utterances: List<String>,
    val meter: MeterState,
    val tags: List<Tag>,
    val seenEvidence: List<String>,
)

/** What one committed utterance changed. */
data class SessionUpdate(
    val meter: MeterState,
    val band: Band,
    val windowText: String,
    val result: ScoreResult,
    val newEvidence: List<Span>,
    val recommendation: Recommendation,
)

data class Advance(val state: SessionState, val update: SessionUpdate)
